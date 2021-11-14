import os

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from rl_mestrado.agent.actor_cnn import DeepActorAgentLearner
from rl_mestrado.tools.log import get_logger

#=======================================================| VARIABLES

DATA_PATH = os.path.join('silver', 'daily_feature_set.csv')
ASSETS = ['SPY', 'TLT.O', 'XLK']
MODEL_OUTPUT_PATH = os.path.join('results')
BACKTEST_OUTPUT_PATH = os.path.join('results', 'backtest')
WEIGHTS_OUTPUT_PATH = os.path.join('results', 'weights')
START_OUT_SAMPLE = '2015-01-05'
END_OUT_SAMPLE = '2021-09-30'
N_FEATURES = 14
N_DAYS = 60
EPOCHS = 10000

#=======================================================| TRAIN

data = pd.read_csv(DATA_PATH, parse_dates=True, index_col=0)
data = data.loc[(~data['TLT.O_logReturns'].isnull()) | (~data['TAIL.K_logReturns'].isnull())]
data.fillna(0, inplace=True)

start_out_samp = pd.Timestamp(START_OUT_SAMPLE)
end_out_samp = pd.Timestamp(END_OUT_SAMPLE)

df_in_sample = data.loc[:start_out_samp, :]
df_out_sample = data.loc[start_out_samp:end_out_samp, :]

agent  = DeepActorAgentLearner(
    learning_rate=1e-4,
    assets=ASSETS,
    n_features=N_FEATURES,
    n_assets=len(ASSETS),
    n_days=N_DAYS
)


model_path = agent.train(
    data=df_in_sample, 
    epochs=EPOCHS, 
    result_output_path=MODEL_OUTPUT_PATH
)

#=======================================================| BACKTEST
print('\n\n\n#====================| PERFORMING BACKTEST |')


port_value = 0.0
port_values = []

columns_to_drop = [c for c in df_out_sample.columns if 'TAIL' in c]
df_out_sample = df_out_sample.drop(columns_to_drop, axis=1)

first_state = df_out_sample.iloc[0:N_DAYS, :]
state = first_state.values
weights = agent.act(state)
weights_vec = []

df_out_sample = df_out_sample.iloc[1:, :]

for i in range(N_DAYS, df_out_sample.shape[0]):

    date = pd.Timestamp(df_out_sample.iloc[i, :].name)
    next_state = df_out_sample.iloc[i - N_DAYS : i,:]
    next_assets_returns = next_state.iloc[-1,:][[c + '_logReturns' for c in ASSETS]].values

    port_value += np.log(np.dot(weights, np.exp(next_assets_returns)))
    port_values.append((date, np.exp(port_value)))

    state = next_state.values
    weights = agent.act(state)
    weights_vec.append((date, *list(weights)))

port_values = pd.DataFrame(port_values, columns=['date', 'model_1']).set_index('date')

backtest_result_path = os.path.join(BACKTEST_OUTPUT_PATH, agent._agent_name+'.csv')
port_values.to_csv(backtest_result_path, index=True)

weights_df = pd.DataFrame(weights_vec, columns=['date']+ASSETS).set_index('date')
weights_result_path = os.path.join(WEIGHTS_OUTPUT_PATH, agent._agent_name+'.csv')
weights_df.to_csv(weights_result_path, index=True)

print(f"Backtest result saved at {backtest_result_path}")

