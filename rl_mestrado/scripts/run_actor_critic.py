import os
from copy import deepcopy

import datetime as dt
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from rl_mestrado.agent.actor_critic import ActorCriticAgentLearner
from rl_mestrado.tools.log import get_logger

#=======================================================| VARIABLES

DATA_PATH = os.path.join('silver', 'daily_feature_set.csv')
NAME_SUFFIX = "daily_replay"
ASSETS = ['SPY', 'TLT.O', 'XLK']
MODEL_OUTPUT_PATH = os.path.join('results')
BACKTEST_OUTPUT_PATH = os.path.join('results', 'backtest')
WEIGHTS_OUTPUT_PATH = os.path.join('results', 'weights')
TRAIN_OUTPUT_PATH = os.path.join('results', 'portfolio_values')
START_OUT_SAMPLE = '2015-01-05'
END_OUT_SAMPLE = '2021-09-30'
N_FEATURES = 14
EPOCHS = 5000

#=======================================================| TRAIN

data = pd.read_csv(DATA_PATH, parse_dates=True, index_col=0)
data = data.loc[(~data['TLT.O_logReturns'].isnull()) | (~data['TAIL.K_logReturns'].isnull())]
data.fillna(0, inplace=True)
print(data.columns)

start_out_samp = pd.Timestamp(START_OUT_SAMPLE)
end_out_samp = pd.Timestamp(END_OUT_SAMPLE)

df_in_sample = data.loc[:start_out_samp, :]
df_out_sample = data.loc[start_out_samp:end_out_samp, :]

actor_lr = 1e-4
critic_lr = actor_lr * 10
n_epochs =  EPOCHS

agent  = ActorCriticAgentLearner(
    actor_lr=actor_lr,
    critic_lr=critic_lr,
    assets=ASSETS,
    n_features=N_FEATURES,
    n_assets=len(ASSETS),
    gamma=1.0,
    name_suffix=f"daily_actor_lr_{actor_lr}_critic_lr_{critic_lr}_epoch_{n_epochs}_gamma_{0.25}"
)

model_path, train_df = agent.train(
    data=df_in_sample, 
    epochs=n_epochs, 
    result_output_path=MODEL_OUTPUT_PATH
)

#=======================================================| BACKTEST
print('\n#====================| PERFORMING BACKTEST |')


port_value = 0.0
backtest_df = []

columns_to_drop = [c for c in df_out_sample.columns if 'TAIL' in c]
df_out_sample = df_out_sample.drop(columns_to_drop, axis=1)

first_date = df_out_sample.iloc[0, :]
state = first_date.values
weights = agent.act(state)

df_out_sample = df_out_sample.iloc[1:, :]

weights_vec = []

for date, row in df_out_sample.iterrows():
    port_value += np.log(np.dot(weights, np.exp(row[[c + '_logReturns' for c in ASSETS]].values)))
    backtest_df.append((date, np.exp(port_value)))

    state = row.values
    weights = agent.act(state)
    weights_vec.append((date, *list(weights)))

backtest_df = pd.DataFrame(backtest_df, columns=['date', agent._agent_name]).set_index('date')

backtest_result_path = os.path.join(BACKTEST_OUTPUT_PATH, agent._agent_name+'.csv')
backtest_df.to_csv(backtest_result_path, index=True)

weights_df = pd.DataFrame(weights_vec, columns=['date']+ASSETS).set_index('date')
weights_result_path = os.path.join(WEIGHTS_OUTPUT_PATH, agent._agent_name+'.csv')
weights_df.to_csv(weights_result_path, index=True)

print(f"Backtest result saved at {backtest_result_path}")

