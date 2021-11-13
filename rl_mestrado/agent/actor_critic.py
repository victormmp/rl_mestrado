import datetime as dt
import os
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.random import default_rng
from rl_mestrado.model.simple_network import SimpleNetworkActor
from rl_mestrado.model.simple_network import SimpleNetworkCritic
from rl_mestrado.tools.log import get_logger
from rl_mestrado.tools.path_tools import check_path
from rl_mestrado.tools.stopwatch import StopWatch
from torch.optim import Adam


LOGGER = get_logger('q_learning')


class ActorCriticAgentLearner:

    def __init__(self, assets: str, n_features: int = 20, n_assets: int = 3, gamma: float = 1.,
        actor_lr: float = 1e-4, critic_lr: float = 1e-3, seed: int = None, name_suffix: str = "",
        expr_buffer_size: int = 1000, steps_until_replay: int = 100, expr_replay_size: int = 100,
        lmbda: float = 0.5, replay: bool = True):
        """Simple Actor Critic Algorithm for discrete action space with function approximation.

        Args:
            features (int): [description]
            assets (int): [description]
            learning_rate (float, optional): [description]. Defaults to 1e-4.
            seed (int, optional): [description]. Defaults to None.
        """

        self._n_features = n_features
        self._n_assets = n_assets
        self._assets = assets
        self._actor_lr = actor_lr
        self._critic_lr = critic_lr
        self._seed = seed
        self._execution_date = dt.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        self._agent_name = self._execution_date + ("_" + name_suffix if name_suffix else "")

        self._gamma = gamma

        # This parameter controlls the eligibility traces. Closer to 0 brings a behaviour closer to
        # a TD(0) algorithm. Closer to 1 brings a behaviour closer to a Monte Carlo algorithm.
        self._lambda = lmbda

        self._rng = default_rng(seed=self._seed)

        self._actor = SimpleNetworkActor(features=n_features, outputs=n_assets)
        self._critic = SimpleNetworkCritic(features=n_features + n_assets, outputs=1)
        self._actor_optimizer = Adam(self._actor.parameters(), lr=actor_lr)
        self._critic_optimizer = Adam(self._critic.parameters(), lr=critic_lr)

        self._replay = replay
        self._expr_buffer = []
        self._expr_buffer_size = expr_buffer_size
        self._steps_until_replay = steps_until_replay
        self._expr_replay_size = expr_replay_size
    
    def train(self, data: pd.DataFrame, epochs: int = 1000, window_size: int = 100, result_output_path: str = '.'):
        
        # Disable pytorch profiling tools to improve speed.
        torch.autograd.set_detect_anomaly(False)
        torch.autograd.profiler.profile(False)
        torch.autograd.profiler.emit_nvtx(False)

        LOGGER.info(f"Starting training with {epochs} epochs and {self._actor_lr} learning rate.")

        data = deepcopy(data)

        self._actor.train()
        self._critic.train()
        sw_total = StopWatch()
        last_log = sw_total.read()

        columns_to_drop = [c for c in data.columns if 'TAIL' in c]
        data = data.drop(columns_to_drop, axis=1)
        
        valid_dates = list(data.index)

        values = []

        replay_steps_counter = 0

        actor_losses = []
        delta_losses = []
        critic_losses = []
        expr_replay_losses = []

        for epoch in range(epochs):

            # Storing losses to dump after training
            act_mean_loss   = 0.
            delta_mean_loss = 0.
            crit_mean_loss  = 0.
            
            sw = StopWatch()
            self._actor_optimizer.zero_grad()
            self._critic_optimizer.zero_grad()

            start_index = self._rng.choice(len(valid_dates)-window_size)
            sample_data = data.loc[valid_dates[start_index]: valid_dates[start_index + window_size]]
            port_value = torch.tensor(0.)

            first_date = sample_data.iloc[0, :]
            previous_state = torch.tensor(first_date.values).float()
            with torch.no_grad():
                action = self._actor(previous_state)

            sample_data = sample_data.iloc[1:, :]

            # Initialize Elegibility Trace
            z = dict()
            for name, param in self._critic.named_parameters():
                if param.grad is not None:
                    z[name] = param.grad

            for date, row in sample_data.iterrows():
            
                # Calculate the reward from the previous state
                next_state = torch.tensor(row.values).float()
                next_action = self._actor(next_state.clone())

                reward = torch.dot(action, torch.tensor(np.exp(row[[c + '_logReturns' for c in self._assets]].values)).float())
                reward = torch.log(reward)
                
                # Existe um problema nessa abordagem considerando ativos financeiros. Como o dia ini-
                # cial está sendo definido aleatoriamente, então um dia qualquer pode ser tanto o
                # primeiro dia da troca, quanto o último. Assim, Se ele for o primeiro dia, o valor dele
                # pode ser alto, e se for o último pode ser baixo. Um crítico que tentar aproximar o valor
                # desse estado, dessa forma, vai ter problema, pois esse valor pode assumir esses dois extremos.

                with torch.no_grad():
                    delta = (
                        reward.detach()
                        + self._gamma * self._critic(next_state.clone(), next_action) 
                        - self._critic(previous_state, action)
                    )

                # delta = (
                #     reward.detach().clone()
                #     + self._gamma * self._critic(next_state.clone(), next_action) 
                #     - self._critic(previous_state.clone(), action)
                # )

                ## Calculate actor loss - variation 1
                # action.requires_grad = True
                # action_value = self._critic(previous_state, action)
                # action_value.backward()
                # actor_loss = - torch.dot(self._actor(previous_state), action.grad)
                ## Zero critic gradients to avoid backpropagation on critic from actor_loss
                # self._critic_optimizer.zero_grad()
                
                ## Calculate actor loss - variation 2
                actor_loss = - self._critic(previous_state, self._actor(previous_state))

                # Calculate critic loss with eligibility traces
                critic_loss = - delta * self._critic(previous_state, action.detach().clone())
                # critic_loss = delta ** 2

                actor_loss.backward()
                # Zero critic gradients to avoid backpropagation on critic from actor_loss
                self._critic_optimizer.zero_grad()

                # Run backward pass to calculate the current Q-function gradient relative to the
                # model weights.
                critic_loss.backward()

                # Add eligibility traces to critic gradients
                for name, param in self._critic.named_parameters():
                    param.grad += - delta * self._gamma * self._lambda * z.get(name, 0)
                
                # Update Eligibility traces
                for name, param in self._critic.named_parameters():
                    if param.grad is not None:
                        z[name] = param.grad

                self._actor_optimizer.step()
                self._critic_optimizer.step()

                if self._replay:
                    self._store_experience(previous_state, action, reward, next_state, next_action)

                # Prepare for the next time step
                port_value += reward
                previous_state = next_state.detach().clone()
                action = next_action.detach().clone()

                ## This can be used instead of optimizer.zero_grad()
                # for param in self._actor.parameters():
                #     param.grad=None
                
                # for param in self._critic.parameters():
                #     param.grad=None

                self._actor_optimizer.zero_grad()
                self._critic_optimizer.zero_grad()

                act_mean_loss +=  - actor_loss.item()
                delta_mean_loss += delta.item()
                crit_mean_loss += - critic_loss.item()
            
            if self._replay and (replay_steps_counter >= self._steps_until_replay):
                replay_loss = self.replay()
                if replay_loss:
                    expr_replay_losses.append((epoch, replay_loss))
                replay_steps_counter = 0
            elif self._replay:
                replay_steps_counter += 1

            actor_losses.append((epoch, act_mean_loss/window_size))
            delta_losses.append((epoch, delta_mean_loss/window_size))
            critic_losses.append((epoch, crit_mean_loss /  window_size))
            
            values.append((epoch, np.exp(port_value.item())))
            if sw_total.read() - last_log > dt.timedelta(seconds=3):
                LOGGER.info(f"[{epoch + 1} / {epochs}] Value = {round(np.exp(port_value.item()), 5)} {sw.read()}")
                # LOGGER.info(f"[{epoch + 1} / {epochs}]     last weights = {list(action.numpy())}")
                last_log = sw_total.read()
        
        # Save results
        model_output_path = os.path.join(result_output_path, 'models')
        model_path = self.save(output_path=model_output_path, model_name=self._agent_name)

        values_output_path = os.path.join(result_output_path, 'portfolio_values', self._agent_name+'.csv')
        LOGGER.info(f"Saving portfolio values as {values_output_path}")
        result_df = pd.DataFrame(values,columns=['epoch', self._agent_name]).set_index('epoch')
        result_df.to_csv(values_output_path, index=False)

        losses_output_path = os.path.join(result_output_path, 'losses', 'actor_critic', self._agent_name+'.csv')
        LOGGER.info(f"Saving training losses in {losses_output_path}")
        losses_df = pd.concat([
            pd.DataFrame(actor_losses, columns=['epoch', 'actor loss']).set_index('epoch'),
            pd.DataFrame(delta_losses, columns=['epoch', 'delta']).set_index('epoch'),
            pd.DataFrame(critic_losses, columns=['epoch', 'critic loss']).set_index('epoch'),
        ], axis=1)
        losses_df.to_csv(losses_output_path, index=True)

        LOGGER.info(f"Finished training with {epochs} epochs in {sw_total.read()}.")

        return model_path, result_df

    def _store_experience(self, previous_state, action, reward, next_state, next_action):
        self._expr_buffer.append(
            (
                previous_state.detach().numpy(), 
                action.detach().numpy(), 
                reward.detach().numpy(), 
                next_state.detach().numpy(), 
                next_action.detach().numpy()
            )
        )
        if len(self._expr_buffer) > self._expr_buffer_size:
            del self._expr_buffer[self._rng.integers(self._expr_buffer_size)]
    
    def replay(self):

        if len(self._expr_buffer) < self._expr_buffer_size:
            return None

        replay_losses = []

        for epoch in range(100):

            start = self._rng.integers(self._expr_buffer_size - self._expr_replay_size)
            batch = self._expr_buffer[start: start + self._expr_replay_size]
            
            previous_state      = torch.tensor(np.array([s1 for (s1, a1, r, s2, a2) in batch]))
            action              = torch.tensor(np.array([a1 for (s1, a1, r, s2, a2) in batch]))
            reward              = torch.tensor(np.array([r for (s1, a1, r, s2, a2) in batch])).view(-1,1)
            next_state          = torch.tensor(np.array([s2 for (s1, a1, r, s2, a2) in batch]))
            next_action         = torch.tensor(np.array([a2 for (s1, a1, r, s2, a2) in batch]))

            with torch.no_grad():
                delta = (
                    reward
                    + self._gamma * self._critic(next_state, next_action) 
                    - self._critic(previous_state, action)
                )

            critic_loss = torch.mean(- delta * self._critic(previous_state, action))

            self._critic_optimizer.zero_grad()
            critic_loss.backward()
            self._critic_optimizer.step()

            replay_losses.append(critic_loss.item())
        
        self._critic_optimizer.zero_grad()

        return np.mean(replay_losses)
    
    def act(self, state: np.ndarray) -> np.ndarray:
        self._actor.eval()
        state = deepcopy(state)
        with torch.no_grad():
            state = torch.tensor(state).float()
            weights = self._actor(state)
        
        return weights.detach().numpy()

    def load(self, model_path: str, exclude_params: list = None):

        if not model_path.endswith(".tar"):
            raise AttributeError(f"File with agent model states must be a tar object!")

        LOGGER.info(f"Loading agent model state from {model_path}")

        agent_state = torch.load(model_path)

        actor_state = agent_state.pop("_actor")
        critic_state = agent_state.pop("_critic")
        actor_optimizer_state = agent_state.pop("_actor_optimizer")
        critic_optimizer_state = agent_state.pop("_critic_optimizer")

        # Filter parameters to not overwrite
        if exclude_params:
            for param in exclude_params:
                _ = agent_state.pop(param)

        # Load trader parameters
        self.__dict__.update(agent_state)

        # Rebuilding models
        self._actor: nn.Module = SimpleNetworkActor(features=self._n_features, outputs=self._n_assets)
        self._critic: nn.Module = SimpleNetworkCritic(features=self._n_features * self._n_assets, outputs=self._n_assets)
        self._actor_optimizer = Adam(self._actor.parameters(), lr=self._actor_lr)
        self._critic_optimizer = Adam(self._actor.parameters(), lr=self._critic_lr)

        # Load model states
        self._actor.load_state_dict(actor_state)
        self._critic.load_state_dict(critic_state)
        self._actor_optimizer.load_state_dict(actor_optimizer_state)
        self._critic_optimizer.load_state_dict(critic_optimizer_state)

        self._rng = default_rng(seed=self._seed)

        return self

    def save(self, output_path: str, model_name: str = None):
        output_path = check_path(output_path)
        
        if model_name:
            model_name = model_name + ".tar"
        else:
            model_name = dt.datetime.now().strftime("agent_%Y_%m_%d__%H_%M.tar")
        agent_state = deepcopy(self.__dict__)
        agent_state.update(
             {
                "_actor": self._actor.state_dict(),
                "_critic": self._critic.state_dict(),
                "_actor_optimizer": self._actor_optimizer.state_dict(),
                "_critic_optimizer": self._critic_optimizer.state_dict(),
                "_rng": None
             }
        )

        output_path = os.path.join(output_path, model_name)
        torch.save(agent_state, output_path)

        output_path = os.path.realpath(output_path)
        LOGGER.info(f"Saving trader states in {output_path}")
        
        return output_path
