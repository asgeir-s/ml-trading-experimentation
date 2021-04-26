# %%
from datetime import datetime
import json
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
from tensortrade.oms.instruments import USDT, BCH
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
from stable_baselines3 import DQN as AGENT
from binance.client import Client
binance_client = Client()


import os, sys

sys.path.insert(0, os.path.abspath("../.."))
from util.simulate_models import simulate_running

from lib.data_util import load_candlesticks
from rl.tensortrade.position_change_chart import PositionChangeChart

BASE_ASSET = USDT
ASSET = BCH
INTERVAL = "15m"
# %%
candlesticks = load_candlesticks(ASSET.symbol + BASE_ASSET.symbol, INTERVAL, custom_data_path="../../tmp", binance_client=binance_client)

predictions_csv_path = f"../../tmp/simulated-predictions/{ASSET.symbol+BASE_ASSET.symbol}-{INTERVAL}"
# %% uild model history
# predictions = simulate_running(candlesticks)
# predictions.to_csv(predictions_csv_path)

# %%
predictions = pd.read_csv(predictions_csv_path, index_col="close time", parse_dates=True)

# %%
raw_features = ta.add_all_ta_features(
    candlesticks,
    open="open",
    high="high",
    low="low",
    close="close",
    volume="volume",
)

# %% add raw data features
binance_data = pd.concat([predictions, raw_features.iloc[:, 5:]], join="inner", axis=1)
# binance_data = predicted

# %%
def create_env(
    features: DataFrame, candlestics: DataFrame, scaler: Any, envs_config=None,
):
    exchange = Exchange("binance", service=execute_order)(
        Stream.source(list(candlestics["close"]), dtype="float").rename(f"{BASE_ASSET.symbol}-{ASSET.symbol}"),
    )

    scaled_data = DataFrame(scaler.transform(features.values), columns=features.columns)

    with NameSpace("binance"):
        straems: Any = [
            Stream.source(list(scaled_data[c]), dtype="float").rename(c)
            for c in scaled_data.columns
        ]

    feed = DataFeed(straems)
    # print("DATA PROVIDED BY ONE STEP IN THE FEED:")
    # print(json.dumps(feed.next(), indent=4))

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
        USDT, [Wallet(exchange, 10000 * BASE_ASSET), Wallet(exchange, 1 * ASSET),],
    )

    # action_scheme = ManagedRiskOrders(
    #     stop=[0.01, 0.02, 0.03, 0.05, 0.08],
    #     take=[0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15],
    #     trade_sizes=[1,],
    # )

    action_scheme = BSH(
        portfolio.get_wallet(exchange.id, BASE_ASSET), portfolio.get_wallet(exchange.id, ASSET)
    )

    # reward_scheme = RiskAdjustedReturns(
    #     return_algorithm="sharpe", risk_free_rate=2.5, target_returns=0.0, window_size=3
    # )

    reward_scheme = "risk-adjusted"
    # reward_scheme = "simple"

    env = default.create(
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        feed=feed,
        window_size=12,
        renderer_feed=renderer_feed,
        renderer=PlotlyTradingChart(
            display=False,
            save_format="html",
            path="./agents/charts/",
            auto_open_html=False,
        ),
        max_allowed_loss=0.003,
    )
    return env


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
    return info


# %% simulate live running with periodically retraining
def simulate_periodically_retrain(
    features: DataFrame,
    candlesticks: DataFrame,
    training_interval=672,
    start_running_index=15000,
    training_window=15000,
    window_for_agent=12,
):
    last_index = start_running_index
    model = AGENT
    predictions = []
    while last_index < len(features):
        start_training_index = last_index-training_window
        stop_training_index = last_index
        print(f"Start of training env: {start_training_index}")
        print(f"Stop of training env: {stop_training_index}")
        training_features = features[start_training_index:stop_training_index]
        training_candlesticks = candlesticks[start_training_index:stop_training_index]
        scaler = StandardScaler().fit(training_features)  # fit scaler on training data

        training_env = create_env(training_features, training_candlesticks, scaler)

        agent = model("MlpPolicy", training_env, verbose=1)
        agent.learn(total_timesteps=100000)

        start_running_index = stop_training_index - window_for_agent + 1 # To not predict the last predition from the last interval like the fist in thi
        stop_running_index = (
            (stop_training_index + training_interval)
            if len(features) > (stop_training_index + training_interval)
            else len(features)
        )
        print(
            f"Start of simulation: {start_running_index} (because the agent wont start emitting preditions before afther {window_for_agent} steps, at step {start_running_index + window_for_agent})"
        )
        print(f"Stop of simulation: {stop_running_index}")
        running_features = features[start_running_index:stop_running_index]
        running_candlesticks = candlesticks[start_running_index:stop_running_index]
        running_env = create_env(running_features, running_candlesticks, scaler)

        res = simulate(agent, running_env)
        predictions = predictions + [res]
        last_index = stop_running_index

    return predictions


# %%
print("Start simulation")
info = simulate_periodically_retrain(binance_data, candlesticks)
for i in info:
    print(i)

