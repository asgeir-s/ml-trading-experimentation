from strategies import PricePredictor as Strategy
from lib import data_util
from lib.live_runner import LiveRunner
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

    print("Trading system name: ", TRADING_STRATEGY_INSTANCE_NAME)
    print("Asset: ", ASSET)
    print("Base Asset: ", BASE_ASSET)
    print("candlesticks interval: ", CANDLESTICK_INTERVAL)

    strategy = Strategy(
        min_value_asset=MIN_VALUE_ASSET,
        min_value_base_asset=MIN_VALUE_BASE_ASSET,
        configurations=config,
    )

    print("Start loading candlesticks")
    candlesticks = data_util.load_candlesticks(
        instrument=ASSET + BASE_ASSET, interval=CANDLESTICK_INTERVAL, binance_client=binance_client
    )
    print("Finished loading candlesticks")

    print("Start generating features")
    features = strategy.generate_features(candlesticks)
    print("Finished generating features")

    print("Start initiating strategy")
    strategy.init(candlesticks=candlesticks, features=features)
    print("Finished initiating strategy")

    print("Start live trading")

    # TODO: fix this to run with the new refactoring
    runner = LiveRunner(
        trading_strategy_instance_name=TRADING_STRATEGY_INSTANCE_NAME,
        asset=ASSET,
        base_asset=BASE_ASSET,
        candlestick_interval=CANDLESTICK_INTERVAL,
        binance_client=binance_client,
        strategy=strategy,
    )

    runner.start()


if __name__ == "__main__":
    with open(configuration_file_path) as config_file:
        configurations = json.load(config_file)
        with open(configurations["binanceApiSecretPath"]) as secret_file:
            secrets = json.load(secret_file)
            binance_secrets = secrets["binance"]
            client = Client(binance_secrets["apiKey"], binance_secrets["apiSecret"])
            main(client, configurations)
