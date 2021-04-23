# %%
from typing import Any
from pandas.core.frame import DataFrame
import ta
from sklearn.preprocessing import StandardScaler

import pandas as pd
import tensortrade.env.default as default
from tensortrade.env.default import stoppers
from tensortrade.env.default.actions import ManagedRiskOrders
from tensortrade.env.default.renderers import FileLogger, PlotlyTradingChart
from tensortrade.env.default.rewards import RiskAdjustedReturns, SimpleProfit
from tensortrade.feed.core import Stream, DataFeed, NameSpace
from tensortrade.oms.instruments import USDT, BTC, LTC
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.agents import DQNAgent


import os, sys

sys.path.insert(0, os.path.abspath("../.."))

from lib.data_util import load_candlesticks
from rl.tensortrade.position_change_chart import PositionChangeChart


# %%
candlesticks_ltcusdt = load_candlesticks("LTCUSDT", "1h", custom_data_path="../../tmp")

binance_data = candlesticks_ltcusdt

binance_data = ta.add_all_ta_features(
    binance_data, open="open", high="high", low="low", close="close", volume="volume",
)
print(binance_data.columns)
training = binance_data[100:-3000]
testing = binance_data[-3000:]

# %%
def create_env(data: DataFrame, scaler: Any, envs_config=None):

    binance = Exchange("binance", service=execute_order)(
        Stream.source(list(data["close"]), dtype="float").rename("USDT-LTC"),
    )

    features_to_use = data.iloc[:, 6:]
    scaled_data = DataFrame(
        scaler.transform(features_to_use.values), columns=features_to_use.columns
    )

    with NameSpace("binance"):
        straems: Any = [
            Stream.source(list(scaled_data[c]), dtype="float").rename(c)
            for c in scaled_data.columns
        ]

    feed = DataFeed(straems)
    print("DATA PROVIDED BY ONE STEP IN THE FEED:")
    print(feed.next())

    renderer_data = data.rename(columns={"close time": "date"})[
        ["date", "open", "high", "low", "close", "volume"]
    ]
    renderer_streams: Any = [
        Stream.source(list(renderer_data[col])).rename(col)
        for col in renderer_data.columns
    ]

    renderer_feed = DataFeed(renderer_streams)
    # renderer_feed.next()

    portfolio = Portfolio(
        USDT, [Wallet(binance, 10000 * USDT), Wallet(binance, 1 * LTC),],
    )

    action_scheme = ManagedRiskOrders(
        stop=[0.01, 0.02, 0.03, 0.05, 0.08],
        take=[0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15],
        trade_sizes=[1,],
    )

    # reward_scheme = RiskAdjustedReturns(
    #     return_algorithm="sharpe", risk_free_rate=2.5, target_returns=0.0, window_size=3
    # )

    reward_scheme = SimpleProfit(window_size=20)

    env = default.create(
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        feed=feed,
        window_size=40,
        renderer_feed=renderer_feed,
        renderer=PlotlyTradingChart(
            display=False, save_format="html", path="./agents/charts/"
        ),
        max_allowed_loss=0.5,
    )
    return env


# %%
print(training.iloc[:, 6:].columns)
scaler = StandardScaler().fit(
    training.iloc[:, 6:].values
)  # fit scaler on training data
training_env = create_env(training, scaler)
# %%
agent = DQNAgent(training_env)
# agent.train(n_episodes=5, n_steps=25973, save_path="agents/")
agent.train(
    n_episodes=500,
    save_path="agents/DQNAgent",
    save_every=10,
    n_steps=len(training) - 1,
    render_interval=1000,
    learning_rate=0.001,
    discount_factor=0.5,
)
# agent.restore(path="agents/DQNAgent/policy_network__4ab36699-64ea-4c4f-bf8b-f0e8ce4bed8a__002.hdf5")
print(f"Agent saved with id: {agent.id}")

# %% setup testing data
test_env = create_env(testing, scaler)

# %%
episode_reward = 0
done = False
obs: Any = test_env.reset()
first_run = True
end_step = 100
step = 0
info = {}


while not done and step < end_step:
    action = agent.get_action(obs)
    obs, reward, done, info = test_env.step(action)
    episode_reward += reward
    step += 1
    if first_run:
        print("Initial state", info)
        print(obs)
        print(episode_reward)
        first_run = False

print("Final state", info)
print(obs)
print(episode_reward)
test_env.render()
test_env.renderer.save()


# %%
