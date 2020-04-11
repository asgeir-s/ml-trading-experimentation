import json
from binance.client import Client
from binance.enums import *
from typing import Any, Optional
from binance.websockets import BinanceSocketManager
from lib.strategy import Strategy
from lib.tradingSignal import TradingSignal
import pandas as pd
from lib import data_util
import time

TRADING_STRATEGY_INSTANCE_NAME = "test_runner_1"

ASSET = "BTC"
BASE_ASSET = "USDT"
TRADINGPAIR = ASSET + BASE_ASSET

candlestick_interval = "1h"
candlesticks: pd.DataFrame
trades: pd.DataFrame


def main(binance_client):
    start(binance_client)


def start(binance_client, strategy: Strategy) -> str:
    print("Start called")
    trades = data_util.load_trades(
        instrument=TRADINGPAIR,
        interval=candlestick_interval,
        trading_strategy_instance_name=TRADING_STRATEGY_INSTANCE_NAME,
    )
    current_position = get_current_position(binance_client)
    assert trades.tail(1)["signal"].values[0] == current_position, "The last trade in the trades dataframe and the current position on the exchange should be the same."
    candlesticks = data_util.load_candlesticks(
        instrument=TRADINGPAIR, interval=candlestick_interval, binance_client=binance_client
    )
    print(f"Current position is (last signal): {current_position}")
    binance_socket_manager = BinanceSocketManager(binance_client)
    # start any sockets here, i.e a trade socket
    connection_key = binance_socket_manager.start_kline_socket(
        TRADINGPAIR,
        setup_message_processor(binance_socket_manager, strategy),
        interval=KLINE_INTERVAL_1HOUR,
    )
    # then start the socket manager
    binance_socket_manager.start()
    return connection_key


def get_current_position(binance_client) -> TradingSignal:
    balance = client.get_asset_balance(asset=INSTRUMENT)
    if balance > 0.01:
        return TradingSignal.BUY
    else:
        return TradingSignal.SELL


def setup_message_processor(binance_socket_manager: Any, strategy: Strategy):
    def process_message(msg: Any):
        signal: Optional[TradingSignal] = None
        if msg["e"] == "error":
            print(f"Error from websocket: {msg['m']}")
            # close and restart the socket
            binance_socket_manager.close()
            start(binance_socket_manager)

        else:
            print("message type: {}".format(msg["e"]))
            print(msg)
            if msg["x"] == True:
                candlesticks = data.add_candle(
                    instrument=TRADINGPAIR,
                    interval=candlestick_interval,
                    new_candle=msg_to_candle(msg),
                )
                signal = strategy.on_candlestick(candlesticks, trades)
            else:
                signal = strategy.on_tick(msg["c"], trades.tail(1)["signal"].values[0])

            if signal is not None:
                place_order(signal, msg["c"])

    return process_message


def place_order(signal: TradingSignal, last_price: float):
    print(f"Placing new order: signal: {signal}")
    order: Any
    if signal == TradingSignal.BUY:
        money = client.get_asset_balance(asset=BASE_ASSET)
        quantity = (money / last_price) * 0.98
        order = client.order_market_buy(symbol=TRADINGPAIR, quantity=quantity)
    elif signal == TradingSignal.SELL:
        quantity = client.get_asset_balance(asset=ASSET)
        order = client.order_market_sell(symbol=TRADINGPAIR, quantity=quantity)

    order = None
    while order["status"] not in ("FILLED", "CANCELED", "REJECTED", "EXPIRED"):
        order = client.get_order(symbol=TRADINGPAIR, orderId=order["orderId"])
        print("current order status: " + order)
        time.sleep(2 * 60)

    if order["status"] == "FILLED":
        print("Order successfully executed! : " + order)

        order_dict = {
            "orderId": order["orderId"],
            "transactTime": order["transactionTime"],
            "price": order["price"],
            "signal": signal,
            "origQty": order["orgQty"],
            "executedQty": order["executedQty"],
            "cummulativeQuoteQty": order["cummulativeQuoteQty"],
            "timeInForce": order["timeInForce"],
            "type": order["type"],
            "side": order["side"],
        }

        trades = data_util.add_trade(
            instrument=TRADINGPAIR,
            interval=candlestick_interval,
            trading_strategy_instance_name=TRADING_STRATEGY_INSTANCE_NAME,
            new_trade_dict=order_dict,
        )
    else:
        print("Order failed! :" + order)


def msg_to_candle(msg: Any):
    assert msg["e"] == "kline", "Should be a candle"
    raw_candle = msg["k"]
    return {
        "open time": raw_candle["t"],
        "open": raw_candle["o"],
        "high": raw_candle["h"],
        "low": raw_candle["l"],
        "close": raw_candle["c"],
        "volume": raw_candle["v"],
        "close time": raw_candle["T"],
        "quote asset volume": raw_candle["q"],
        "number of trades": raw_candle["n"],
        "taker buy base asset volume": raw_candle["V"],
        "taker buy quote asset volume": raw_candle["Q"],
    }


if __name__ == "__main__":
    with open("./secrets.json") as json_file:
        data = json.load(json_file)
        binance_secrets = data["binance"]
        client = Client(binance_secrets["apiKey"], binance_secrets["apiSecret"])
        main(client)
