import datetime as dt
import os
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.random import default_rng
from rl_mestrado.model.simple_network import SimpleNetworkActor
from rl_mestrado.tools.log import get_logger
from rl_mestrado.tools.path_tools import check_path
from rl_mestrado.tools.stopwatch import StopWatch
from torch.optim import Adam


LOGGER = get_logger('q_learning')


class DeepActorAgentLearner:

    def __init__(self, assets: str, n_features: int = 20, n_assets: int = 3, 
        learning_rate: float = 1e-4, seed: int = None, name_suffix: str = ""):
        """Simple QLearning Algorithm for discrete action space with function approximation.

        Args:
            features (int): [description]
            assets (int): [description]
            learning_rate (float, optional): [description]. Defaults to 1e-4.
            seed (int, optional): [description]. Defaults to None.
        """

        self._n_features = n_features
        self._n_assets = n_assets
        self._assets = assets
        self._learning_rate = learning_rate
        self._seed = seed
        self._execution_date = dt.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        self._agent_name = self._execution_date + ("_" + name_suffix if name_suffix else "")

        self._rng = default_rng(seed=self._seed)

        self._actor = SimpleNetworkActor(features=n_features, outputs=n_assets)
        self._optimizer = Adam(self._actor.parameters(), lr=learning_rate)
    
    def train(self, data: pd.DataFrame, epochs: int = 1000, window_size: int = 100, result_output_path: str = '.'):

        LOGGER.info(f"Starting training with {epochs} epochs and {self._learning_rate} learning rate.")

        data = deepcopy(data)

        self._actor.train()
        sw_total = StopWatch()
        last_log = sw_total.read()

        # data = self.data.loc[(~self.data['TLT.O'].isnull()) | (~self.data['TAIL.K'].isnull())]
        columns_to_drop = [c for c in data.columns if 'TAIL' in c]
        data = data.drop(columns_to_drop, axis=1)
        
        valid_dates = list(data.index)

        values = []

        for epoch in range(epochs):
            sw = StopWatch()
            self._optimizer.zero_grad()

            start_index = self._rng.choice(len(valid_dates)-window_size)
            sample_data = data.loc[valid_dates[start_index]: valid_dates[start_index + window_size]]
            port_value = torch.tensor(0.)

            first_date = sample_data.iloc[0, :]
            state = torch.tensor(first_date.values).float()
            weights = self._actor(state)

            sample_data = sample_data.iloc[1:, :]

            for date, row in sample_data.iterrows():
                
                # Calculate the reward from the previous state
                port_value += torch.log(torch.dot(weights, torch.tensor(np.exp(row[[c + '_logReturns' for c in self._assets]].values)).float()))

                # Get the next action for this next state
                state = torch.tensor(row.values).float()
                weights = self._actor(state)
                
            loss =  - port_value / window_size # normalized by the length of the episode
            loss.backward()
            self._optimizer.step()
            
            values.append((epoch, np.exp(port_value.item())))
            if sw_total.read() - last_log > dt.timedelta(seconds=3):
                LOGGER.info(f"[{epoch + 1} / {epochs}] Value = {round(np.exp(port_value.item()), 5)} {sw.read()}")
                last_log = sw_total.read()

        model_output_path = os.path.join(result_output_path, 'models')
        model_path = self.save(output_path=model_output_path, model_name=self._agent_name)

        values_output_path = os.path.join(result_output_path, 'portfolio_values', self._agent_name+'.csv')
        LOGGER.info(f"Saving portfolio values as {values_output_path}")
        result_df = pd.DataFrame(values,columns=['epoch', self._agent_name]).set_index('epoch')
        result_df.to_csv(values_output_path, index=False)

        LOGGER.info(f"Finished training with {epochs} epochs in {sw_total.read()}.")

        return model_path, result_df
    
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
        optimizer_state = agent_state.pop("_optimizer")

        # Filter parameters to not overwrite
        if exclude_params:
            for param in exclude_params:
                _ = agent_state.pop(param)

        # Load trader parameters
        self.__dict__.update(agent_state)

        # Rebuilding models
        self._actor: nn.Module = SimpleNetworkActor(features=self._n_features, outputs=self._n_assets)
        self._optimizer = Adam(self._actor.parameters(), lr=self._learning_rate)

        # Load model states
        self._actor.load_state_dict(actor_state)
        self._optimizer.load_state_dict(optimizer_state)

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
                "_optimizer": self._optimizer.state_dict(),
                "_rng": None
             }
        )

        output_path = os.path.join(output_path, model_name)
        torch.save(agent_state, output_path)

        output_path = os.path.realpath(output_path)
        LOGGER.info(f"Saving trader states in {output_path}")
        
        return output_path
