# from strategies import PricePredictor as Strategy
from strategies import PricePredictor2 as Strategy
from lib import data_util
from lib.backtest import Backtest, setup_file_path
from lib.charting import chartTrades
import pandas as pd
import json
from binance.client import Client
import sys

configuration_file_path = sys.argv[1]

candlesticks: pd.DataFrame
trades: pd.DataFrame


def main(binance_client, config):
    TRADING_STRATEGY_INSTANCE_NAME = config["name"]
    ASSET = config["asset"]
    BASE_ASSET = config["baseAsset"]
    CANDLESTICK_INTERVAL = config["candleInterval"]
    MIN_VALUE_ASSET = float(config["minValueAsset"])
    MIN_VALUE_BASE_ASSET = float(config["minValueBaseAsset"])

    TRADE_START_POSITION = config["backtest"]["startPosition"]
    MISSING_TARGETS_AT_THE_END = config["backtest"]["missingTargetsAtTheEnd"]

    print("Trading system name: ", TRADING_STRATEGY_INSTANCE_NAME)
    print("Asset: ", ASSET)
    print("Base Asset: ", BASE_ASSET)
    print("candlesticks interval: ", CANDLESTICK_INTERVAL)
    print("BACKSTEST!!!!")

    strategy = Strategy(
        min_value_asset=MIN_VALUE_ASSET,
        min_value_base_asset=MIN_VALUE_BASE_ASSET,
        backtest=True,
        configurations=config,
    )
    tmp_path = (
        "./strategies/price_predictor/tmp/backtest/"
        + strategy.__class__.__name__
        + "-"
        + BASE_ASSET
        + ASSET
        + "-"
        + CANDLESTICK_INTERVAL
        + "/"
    )
    path_builder = setup_file_path(tmp_path)

    print("Start loading candlesticks")
    candlesticks = data_util.load_candlesticks(
        instrument=ASSET + BASE_ASSET,
        interval=CANDLESTICK_INTERVAL, # binance_client=binance_client
    )
    print("Finished loading candlesticks")

    print("Start generating features")
    features = strategy.generate_features(candlesticks)[:-MISSING_TARGETS_AT_THE_END]
    # targets = strategy._generate_targets(candlesticks, features)
    candlesticks = candlesticks[:-MISSING_TARGETS_AT_THE_END]
    trade_end_position = len(candlesticks)
    print("Finished generating features")

    # TODO: fix this to run with the new refactoring
    trades = Backtest.run(
        strategy=strategy,
        features=features,
        candlesticks=candlesticks,
        start_position=TRADE_START_POSITION,
        end_position=trade_end_position,
        # signals_csv_path=path_builder("signals"),
        trades_csv_path=path_builder("trades"),
    )
    # trades = Backtest.evaluate(
    #     signals, candlesticks, TRADE_START_POSITION, trade_end_position, 0.001
    # )
    # trades.to_csv(path_builder("trades"))

    chartTrades(
        trades,
        candlesticks,
        TRADE_START_POSITION,
        trade_end_position,
        path_builder("chart", extension="html"),
        # "./tmp/PricePredictor/2021-03-22-20-20-chart.html"
    )


if __name__ == "__main__":
    with open(configuration_file_path) as config_file:
        configurations = json.load(config_file)
        client = Client()
        main(client, configurations)
