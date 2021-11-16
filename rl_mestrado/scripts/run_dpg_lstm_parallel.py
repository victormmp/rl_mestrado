import os
from copy import deepcopy

import datetime as dt
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from rl_mestrado.agent.actor_lstm import DeepActorAgentLearner
from rl_mestrado.tools.log import get_logger

#=======================================================| VARIABLES

DATA_PATH = os.path.join('silver', 'daily_feature_set.csv')
NAME_SUFFIX = "dpg_lstm"
ASSETS = ['SPY', 'TLT.O', 'XLK']
MODEL_OUTPUT_PATH = os.path.join('results')
BACKTEST_OUTPUT_PATH = os.path.join('results', 'backtest')
WEIGHTS_OUTPUT_PATH = os.path.join('results', 'weights')
TRAIN_OUTPUT_PATH = os.path.join('results', 'portfolio_values')
START_OUT_SAMPLE = '2015-01-05'
END_OUT_SAMPLE = '2021-09-30'
N_FEATURES = 14
N_DAYS = 90
N_LAYERS = 20
EPOCHS = 10000

#=======================================================| TRAIN

data = pd.read_csv(DATA_PATH, parse_dates=True, index_col=0)
data = data.loc[(~data['TLT.O_logReturns'].isnull()) | (~data['TAIL.K_logReturns'].isnull())]
data.fillna(0, inplace=True)
print(data.columns)

start_out_samp = pd.Timestamp(START_OUT_SAMPLE)
end_out_samp = pd.Timestamp(END_OUT_SAMPLE)

df_in_sample = data.loc[:start_out_samp, :]
df_out_sample = data.loc[start_out_samp:end_out_samp, :]


def run(**kwargs):

    learning_rate = kwargs.get('learning_rate', 1e-3)
    n_epochs = kwargs.get('epochs', EPOCHS)
    n_days = kwargs.get('n_days', N_DAYS)
    df_in_sample = deepcopy(kwargs.get('df_in_sample'))
    df_out_sample = deepcopy(kwargs.get('df_out_sample'))

    agent  = DeepActorAgentLearner(
        learning_rate=learning_rate,
        assets=ASSETS,
        n_features=N_FEATURES,
        n_assets=len(ASSETS),
        n_layers=N_LAYERS,
        name_suffix=f"dpg_ltsm_lr_{learning_rate}_epoch_{EPOCHS}"
    )


    model_path, train_df = agent.train(
        data=df_in_sample, 
        epochs=n_epochs, 
        result_output_path=MODEL_OUTPUT_PATH,
        window_size=180
    )

    #=======================================================| BACKTEST
    print('\n\n\n#====================| PERFORMING BACKTEST |')


    port_value = 0.0
    backtest_df = []

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
        backtest_df.append((date, np.exp(port_value)))

        state = next_state.values
        weights = agent.act(state)
        weights_vec.append((date, *list(weights)))

    backtest_df = pd.DataFrame(backtest_df, columns=['date', agent._agent_name]).set_index('date')

    backtest_result_path = os.path.join(BACKTEST_OUTPUT_PATH, agent._agent_name+'.csv')
    backtest_df.to_csv(backtest_result_path, index=True)

    weights_df = pd.DataFrame(weights_vec, columns=['date']+ASSETS).set_index('date')
    weights_result_path = os.path.join(WEIGHTS_OUTPUT_PATH, agent._agent_name+'.csv')
    weights_df.to_csv(weights_result_path, index=True)

    print(f"Backtest result saved at {backtest_result_path}")

    return backtest_df, train_df

configs = [
    (1e-6, EPOCHS, df_in_sample, df_out_sample),
    (1e-5, EPOCHS, df_in_sample, df_out_sample),
    (1e-4, EPOCHS, df_in_sample, df_out_sample),
    (1e-3, EPOCHS, df_in_sample, df_out_sample),
    (1e-2, EPOCHS, df_in_sample, df_out_sample)
]

results = Parallel(n_jobs=-2)(
    delayed(run)(
            learning_rate=lr,
            epochs=e,
            df_in_sample=df_i,
            df_out_sample=df_o
    )
    for lr, e, df_i, df_o in configs
)

df_result_backtest, df_result_train = zip(*results)

print("\n\nFINISHED PARALLEL EXECUTION\n")
result_path = os.path.join(TRAIN_OUTPUT_PATH, dt.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")+f'_aggregated_{NAME_SUFFIX}.csv')
pd.concat(df_result_train, axis=1).to_csv(result_path, index=True)
print(f"Train result saved at {result_path}")

result_path = os.path.join(BACKTEST_OUTPUT_PATH, dt.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")+f'_aggregated_{NAME_SUFFIX}.csv')
pd.concat(df_result_backtest, axis=1).to_csv(result_path, index=True)
print(f"Backtest result saved at {result_path}")


