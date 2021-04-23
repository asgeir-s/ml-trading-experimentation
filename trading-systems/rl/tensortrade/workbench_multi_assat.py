# %%
import ta

import pandas as pd
import tensortrade.env.default as default
from tensortrade.env.default import stoppers
from tensortrade.env.default.renderers import FileLogger, PlotlyTradingChart
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
candlesticks_btcusdt = load_candlesticks(
    "BTCUSDT", "1h", custom_data_path="../../tmp"
).add_prefix("BTC:")
candlesticks_ltcusdt = load_candlesticks(
    "LTCUSDT", "1h", custom_data_path="../../tmp"
).add_prefix("LTC:")

binance_data = pd.concat(
    [candlesticks_btcusdt, candlesticks_ltcusdt], axis=1, join="inner"
)

# %%
binance = Exchange("binance", service=execute_order)(
    Stream.source(list(binance_data["BTC:close"]), dtype="float").rename("USDT-BTC"),
    Stream.source(list(binance_data["LTC:close"]), dtype="float").rename("USDT-LTC"),
)

# %%
binance_btc = binance_data.loc[
    :, [name.startswith("BTC") for name in binance_data.columns]
]

binance_ltc = binance_data.loc[
    :, [name.startswith("LTC") for name in binance_data.columns]
]

# %%
ta.add_all_ta_features(
    binance_btc,
    colprefix="BTC:",
    **{k: "BTC:" + k for k in ["open", "high", "low", "close", "volume"]}
)

ta.add_all_ta_features(
    binance_ltc,
    colprefix="LTC:",
    **{k: "LTC:" + k for k in ["open", "high", "low", "close", "volume"]}
)


# %% Feed features testing how it works
# def macd(
#     price: Stream[float], fast: float, slow: float, signal: float
# ) -> Stream[float]:
#     fm = price.ewm(span=fast, adjust=False).mean()
#     sm = price.ewm(span=slow, adjust=False).mean()
#     md = fm - sm
#     signal = md - md.ewm(span=signal, adjust=False).mean()
#     return signal


# %%

print(binance_btc.columns)


with NameSpace("binance"):
    binance_streams = [
        Stream.source(list(binance_btc[c]), dtype="float").rename(c)
        for c in binance_btc.drop(columns=["BTC:close time"]).columns
    ]
    binance_streams += [
        Stream.source(list(binance_ltc[c]), dtype="float").rename(c)
        for c in binance_ltc.drop(columns=["LTC:close time"]).columns
    ]

    # cp = Stream.select(binance_streams, lambda s: s.name == "binance:/BTC:close")
    # binance_streams += [macd(cp, fast=10, slow=50, signal=5).rename("BTC:custom_macd")]


# %%
feed = DataFeed(binance_streams)
# feed.compile()


# %%
feed.next()

# %%

portfolio = Portfolio(
    USDT,
    [Wallet(binance, 1000 * USDT), Wallet(binance, 1 * BTC), Wallet(binance, 1 * LTC),],
)


# %%
env_train = default.create(
    portfolio=portfolio,
    action_scheme="managed-risk",
    reward_scheme="simple",
    feed=feed,
    window_size=15,
    # renderer=PositionChangeChart(),
    renderer=PlotlyTradingChart(display=False, timestamp_format='%Y-%m-%d %H:%M:%S'),
    enable_logger=False,
)

# %%
agent = DQNAgent(env_train)
agent.train(n_steps=20, n_episodes=2, save_path="agents/", render_interval=2)


# %%
env_test = default.create(
    portfolio=portfolio,
    action_scheme="managed-risk",
    reward_scheme="simple",
    feed=feed,
    window_size=15,
    enable_logger=False,
    # renderers=[PositionChangeChart()],
    renderers=[FileLogger(filename='example.log')],
    max_allowed_loss=0.3,
)

# %%
episode_reward = 0
done = False
obs = env_test.reset()
first_run = True
env_test.action_space


while not done:
    action = agent.get_action(obs)
    obs, reward, done, info = env_test.step(action)
    episode_reward += reward
    if first_run:
        print("Initial state", info)
        print(obs)
        first_run = False
    if done:
        print("Final state", info)

# %%

# %%
