# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## RegressionBabyModel

# %%
import pandas as pd
import os, sys

sys.path.insert(0, os.path.abspath("../.."))
import datetime


# %%
from lib.data_splitter import split_features_and_target_into_train_and_test_set
from lib.data_util import load_candlesticks
from rl.finrl.model import FinRLExperiment
from finrl.env.env_stocktrading import StockTradingEnv
from finrl.model.models import DRLAgent
from finrl.trade.backtest import backtest_stats

# %%
candlesticks = load_candlesticks("LTCUSDT", "1h", custom_data_path="../../tmp")

# %%
model = FinRLExperiment()
features = pd.DataFrame(index=candlesticks.index)
features = model.generate_features(candlesticks, features)

features = pd.concat([features, candlesticks], axis=1)
features["tic"] = "LTCUSDT"

features["date"] = pd.to_datetime(features["close time"], unit="ms")
features["date"] = features["date"].astype(str)
features = features.sort_values(["date", "tic"], ignore_index=True)
features

# %%
(
    training_features,
    _,
    testing_features,
    _,
) = split_features_and_target_into_train_and_test_set(features, {}, 20)
# training_features = training_features.reset_index(drop=True)
# testing_features = testing_features.reset_index(drop=True)

# %% debugging
dates = features["date"].sort_values().unique()

# %%
information_cols = features.drop(columns=["index", "tic"]).columns
stock_dimension = len(training_features.tic.unique())
state_space = 1 + 2 * stock_dimension + len(information_cols) * stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

# %%

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000,
    "buy_cost_pct": 0.001,
    "sell_cost_pct": 0.001,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": information_cols,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4,
}
e_train_gym = StockTradingEnv(df=training_features, **env_kwargs)

# %%
env_train, _ = e_train_gym.get_sb_env()
print(type(env_train))

# %%
agent = DRLAgent(env=env_train)
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 128,
}
model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)

# %%
trained_ppo = agent.train_model(
    model=model_ppo, tb_log_name="ppo", total_timesteps=10000
)

# %%

e_trade_gym = StockTradingEnv(df=testing_features, **env_kwargs)
# env_trade, obs_trade = e_trade_gym.get_sb_env()

df_account_value, df_actions = DRLAgent.DRL_prediction(
    model=model_ppo, environment=e_trade_gym
)

# %%
print("==============Get Backtest Results===========")
now = datetime.datetime.now().strftime("%Y%m%d-%Hh%M")

perf_stats_all = backtest_stats(account_value=df_account_value)
perf_stats_all = pd.DataFrame(perf_stats_all)
