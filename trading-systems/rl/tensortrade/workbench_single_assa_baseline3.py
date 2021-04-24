# %%
from datetime import datetime
from typing import Any
from pandas.core.frame import DataFrame
import ta
from sklearn.preprocessing import StandardScaler

import pandas as pd
import tensortrade.env.default as default
from tensortrade.env.default import stoppers
from tensortrade.env.default.actions import BSH, ManagedRiskOrders
from tensortrade.env.default.renderers import FileLogger, PlotlyTradingChart
from tensortrade.env.default.rewards import RiskAdjustedReturns, SimpleProfit
from tensortrade.env.generic.environment import TradingEnv
from tensortrade.feed.core import Stream, DataFeed, NameSpace
from tensortrade.oms.instruments import USDT, BTC, LTC
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
from stable_baselines3 import DQN as AGENT


import os, sys

sys.path.insert(0, os.path.abspath("../.."))

from lib.data_util import load_candlesticks
from rl.tensortrade.position_change_chart import PositionChangeChart


# %%
candlesticks_ltcusdt = load_candlesticks("LTCUSDT", "1h", custom_data_path="../../tmp")
binance_data = candlesticks_ltcusdt

# %%
# binance_data = ta.add_all_ta_features(
#     binance_data, open="open", high="high", low="low", close="close", volume="volume",
# )
def get_traget(name: str, file_path: str):
    df = pd.read_csv(file_path, index_col="close time")
    df.columns.values[0] = name
    return df


tar = [
    get_traget(
        "classifier_up_down",
        "../../tmp/targets/USDTLTC-1h/2021-04-24-11:06:41-classifier_up_down.csv",
    )
]
tar = tar + [
    get_traget(
        "classifier_sklien_simple",
        "../../tmp/targets/USDTLTC-1h/2021-04-24-11:11:24-classifier_sklien_simple.csv",
    )
]
tar = tar + [
    get_traget(
        "lightgbm_regression_baby_min",
        "../../tmp/targets/USDTLTC-1h/2021-04-24-11:17:51-lightgbm_regression_baby_min.csv",
    )
]
tar = tar + [
    get_traget(
        "lightgbm_regression_baby_max",
        "../../tmp/targets/USDTLTC-1h/2021-04-24-11:20:24-lightgbm_regression_baby_max.csv",
    )
]
tar = tar + [
    get_traget(
        "PricePreditionLSTMModel_close",
        "../../tmp/targets/USDTLTC-1h/2021-04-24-11:30:25-PricePreditionLSTMModel_close.csv",
    )
]
tar = tar + [
    get_traget(
        "PricePreditionLSTMModel_ema",
        "../../tmp/targets/USDTLTC-1h/2021-04-24-11:30:25-PricePreditionLSTMModel_ema.csv",
    )
]
tar = tar + [
    get_traget(
        "PricePreditionLSTMModel_low",
        "../../tmp/targets/USDTLTC-1h/2021-04-24-11:33:25-PricePreditionLSTMModel_low.csv",
    )
]
tar = tar + [
    get_traget(
        "PricePreditionLSTMModel_high",
        "../../tmp/targets/USDTLTC-1h/2021-04-24-11:33:50-PricePreditionLSTMModel_high.csv",
    )
]


# %%
targets = pd.concat(tar, axis=1, join="inner")
targets = targets[:-7]
candlesticks_ltcusdt = candlesticks_ltcusdt[:-7]

# %%
binance_data = targets

print(binance_data.columns)
training_features = binance_data[100:-3000]
testing_features = binance_data[-3000:]
training_candlestics = candlesticks_ltcusdt[100:-3000]
testing_candlestics = candlesticks_ltcusdt[-3000:]

# %%
def create_env(
    features: DataFrame, candlestics: DataFrame, scaler: Any, envs_config=None
):

    binance = Exchange("binance", service=execute_order)(
        Stream.source(list(candlestics["close"]), dtype="float").rename("USDT-LTC"),
    )

    scaled_data = DataFrame(scaler.transform(features.values), columns=features.columns)

    with NameSpace("binance"):
        straems: Any = [
            Stream.source(list(scaled_data[c]), dtype="float").rename(c)
            for c in scaled_data.columns
        ]

    feed = DataFeed(straems)
    print("DATA PROVIDED BY ONE STEP IN THE FEED:")
    print(feed.next())

    renderer_data = candlestics.assign(date=candlestics.index)[
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

    # action_scheme = ManagedRiskOrders(
    #     stop=[0.01, 0.02, 0.03, 0.05, 0.08],
    #     take=[0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15],
    #     trade_sizes=[1,],
    # )

    action_scheme = BSH(
        portfolio.get_wallet(binance.id, USDT), portfolio.get_wallet(binance.id, LTC)
    )

    # reward_scheme = RiskAdjustedReturns(
    #     return_algorithm="sharpe", risk_free_rate=2.5, target_returns=0.0, window_size=3
    # )

    reward_scheme = "risk-adjusted"

    env = default.create(
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        feed=feed,
        window_size=40,
        renderer_feed=renderer_feed,
        renderer=PlotlyTradingChart(
            display=False,
            save_format="html",
            path="./agents/charts/",
            auto_open_html=False,
        ),
        max_allowed_loss=0.3,
    )
    return env


# %%
print(training_features.columns)
scaler = StandardScaler().fit(training_features.values)  # fit scaler on training data
training_env = create_env(training_features, training_candlestics, scaler)
# %%
model = AGENT
today = datetime.today()
agent = model("MlpPolicy", training_env, verbose=1)
agent.learn(total_timesteps=100000)
agent.save("agents/DQNAgent/" + today.strftime("%Y-%m-%d-%H:%M:%S") + ".zip")

# %%

def simulate(agent, env: TradingEnv):
    episode_reward = 0
    done = False
    obs: Any = env.reset()
    first_run = True
    step = 0
    info = {}

    while not done:
        action, _states = agent.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        step += 1
        if first_run:
            print("Initial state", info)
            print(episode_reward)
            first_run = False

    print("Final state", info)
    print(episode_reward)
    env.render()
    env.renderer.save()


# %% run on training evn
print("Simulate on training environment:")
simulate(agent, training_env)


# %% run on training evn
print("Simulate on testing environment:")
test_env = create_env(testing_features, testing_candlestics, scaler)
simulate(agent, test_env)
