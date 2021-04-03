from strategies import Third as Strategy
from lib import data_util
from lib.live_runner import LiveRunner
import pandas as pd
import json
from binance.client import Client

TRADING_STRATEGY_INSTANCE_NAME = "test_runner_1"

ASSET = "BTC"
BASE_ASSET = "USDT"

candlestick_interval = "1h"
candlesticks: pd.DataFrame
trades: pd.DataFrame


def main(binance_client):
    strategy = Strategy()

    print("Start loading candlesticks")
    candlesticks = data_util.load_candlesticks(
        instrument=ASSET + BASE_ASSET, interval=candlestick_interval, binance_client=binance_client
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
        candlestick_interval=candlestick_interval,
        binance_client=binance_client,
        strategy=strategy,
    )

    runner.start()


if __name__ == "__main__":
    with open("./secrets.json") as json_file:
        data = json.load(json_file)
        binance_secrets = data["binance"]
        client = Client(binance_secrets["apiKey"], binance_secrets["apiSecret"])
        main(client)
