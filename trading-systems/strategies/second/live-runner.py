from strategies.second.second import Second
from lib.backtest import Backtest
from lib.charting import chartTrades
from lib import data_util
from lib.live_runner import start
import pandas as pd
import json
from binance.client import Client

TRADING_STRATEGY_INSTANCE_NAME = "test_runner_1"

ASSET = "BTC"
BASE_ASSET = "USDT"
TRADINGPAIR = ASSET + BASE_ASSET

candlestick_interval = "1h"
candlesticks: pd.DataFrame
trades: pd.DataFrame


def main(binance_client):
    print("Start loading candlesticks")
    candlesticks = data_util.load_candlesticks(
        instrument=TRADINGPAIR, interval=candlestick_interval, binance_client=binance_client
    )
    print("Finished loading candlesticks")

    print("Start generatinf features")
    features = Second.generate_features(candlesticks)
    print("Finished generating features")

    print("Start initiating strategy")
    strategy = Second(init_features=features)
    print("Finished initiating strategy")

    print("Start live trading")
    start(
        trading_strategy_instance_name=TRADING_STRATEGY_INSTANCE_NAME,
        asset=ASSET,
        base_asset=BASE_ASSET,
        candlestick_interval=candlestick_interval,
        binance_client=binance_client,
        strategy=strategy,
    )
    print("Finished live trading")


if __name__ == "__main__":
    with open("./secrets.json") as json_file:
        data = json.load(json_file)
        binance_secrets = data["binance"]
        client = Client(binance_secrets["apiKey"], binance_secrets["apiSecret"])
        main(client)
