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
from dataclasses import dataclass
from functools import reduce


@dataclass
class LiveRunner:
    trading_strategy_instance_name: str
    asset: str
    base_asset: str
    candlestick_interval: str
    strategy: Strategy
    binance_client: Client

    tradingpair: Optional[str] = None
    current_position: Optional[TradingSignal] = None
    binance_socket_manager: Any = None
    binance_client: Any = None
    binance_socket_connection_key: Optional[str] = None

    def start(self) -> str:
        self.tradingpair = self.asset + self.base_asset
        print(f"Tradingpair: {self.tradingpair}")
        trades = data_util.load_trades(
            instrument=self.tradingpair,
            interval=self.candlestick_interval,
            trading_strategy_instance_name=self.trading_strategy_instance_name,
        )
        self.current_position = self.get_current_position(
            self.binance_client, self.asset, self.base_asset
        )
        last_trade_signal_saved = None if trades is None else trades.tail(1)
        print(f"Last saved trade is: {last_trade_signal_saved}")
        assert last_trade_signal_saved is None or (
            last_trade_signal_saved["signal"].values[0] == str(self.current_position)
        ), "The last trade in the trades dataframe and the current position on the exchange should be the same."
        self.candlesticks = data_util.load_candlesticks(
            instrument=self.tradingpair,
            interval=self.candlestick_interval,
            binance_client=self.binance_client,
        )
        print(f"Current position is (last signal): {self.current_position}")
        self.binance_socket_manager = BinanceSocketManager(self.binance_client)
        # start any sockets here, i.e a trade socket
        self.binance_socket_connection_key = self.binance_socket_manager.start_kline_socket(
            self.tradingpair, self.process_message, interval=self.candlestick_interval,
        )
        # then start the socket manager
        self.binance_socket_manager.start()
        return self.binance_socket_connection_key

    @staticmethod
    def get_current_position(binance_client: Client, asset: str, base_asset: str) -> TradingSignal:
        asset_balance = binance_client.get_asset_balance(asset=asset)
        base_asset_balance = binance_client.get_asset_balance(asset=base_asset)
        print(
            f"Current position (from exchange): Asset: {asset_balance}, Base asset: {base_asset_balance}"
        )
        if float(asset_balance["free"]) > 0.000001:
            print("Current position (from exchange)(last signal executed): BUY")
            return TradingSignal.BUY
        else:
            print("Current position (from exchange)(last signal executed): SELL")
            return TradingSignal.SELL

    def process_message(self, msg: Any):
        signal: Optional[TradingSignal] = None
        if msg["e"] == "error":
            print(f"Error from websocket: {msg['m']}")
            # close and restart the socket
            self.binance_socket_manager.close()
            self.start()
        else:
            candle_raw = msg["k"]
            current_close_price = candle_raw["c"]
            if candle_raw["x"] == True:
                self.candlesticks = data_util.add_candle(
                    instrument=self.tradingpair,
                    interval=self.candlestick_interval,
                    new_candle=self.msg_to_candle(msg),
                )
                signal = self.strategy.on_candlestick(
                    self.candlesticks,
                    data_util.load_trades(
                        instrument=self.tradingpair,
                        interval=self.candlestick_interval,
                        trading_strategy_instance_name=self.trading_strategy_instance_name,
                    ),
                )
            else:
                signal = self.strategy.on_tick(current_close_price, self.current_position)
            if signal is not None:
                self.place_order(
                    signal=signal, last_price=current_close_price,
                )
            else:
                print(".", end="", flush=True)

    def place_order(
        self, signal: TradingSignal, last_price: float,
    ):
        print(f"Placing new order: signal: {signal}")
        order: Optional[Any] = None
        if signal == TradingSignal.BUY:
            money = self.binance_client.get_asset_balance(asset=self.base_asset)["free"]
            quantity = float(money) / float(last_price) * 0.9995
            quantity = round(quantity, 6)
            print(f"ORDER: Market buy {quantity} of {self.tradingpair}")
            order = self.binance_client.order_market_buy(symbol=self.tradingpair, quantity=quantity)
        elif signal == TradingSignal.SELL:
            quantity = float(self.binance_client.get_asset_balance(asset=self.asset)["free"])
            quantity = f"{quantity:.8f}"[:-2]  # make sure we round down
            print(f"ORDER: Market sell {quantity} of {self.tradingpair}")
            order = self.binance_client.order_market_sell(
                symbol=self.tradingpair, quantity=quantity
            )

        while order["status"] not in ("FILLED", "CANCELED", "REJECTED", "EXPIRED"):
            order = self.binance_client.get_order(symbol=self.tradingpair, orderId=order["orderId"])
            print("ORDER status: " + order)
            time.sleep(2 * 60)

        if order["status"] == "FILLED":
            print("Order successfully executed!:")
            print(order)

            data_util.add_trade(
                instrument=self.tradingpair,
                interval=self.candlestick_interval,
                trading_strategy_instance_name=self.trading_strategy_instance_name,
                new_trade_dict=self.order_to_trade(order, signal),
            )
            self.current_position = signal

        else:
            print("Order failed! :")
            print(order)

    @staticmethod
    def order_to_trade(order: Any, signal: TradingSignal):
        acc_price, acc_quantity = reduce(
            lambda acc, item: (
                acc[0] + float(item["price"]) * float(item["qty"]),
                acc[1] + float(item["qty"]),
            ),
            order["fills"],
            (0.0, 0.0),
        )
        avg_price = acc_price / acc_quantity

        return {
            "orderId": int(order["orderId"]),
            "transactTime": int(order["transactTime"]),
            "price": avg_price,
            "signal": str(signal),
            "origQty": float(order["origQty"]),
            "executedQty": float(order["executedQty"]),
            "cummulativeQuoteQty": float(order["cummulativeQuoteQty"]),
            "timeInForce": str(order["timeInForce"]),
            "type": str(order["type"]),
            "side": str(order["side"]),
        }

    @staticmethod
    def msg_to_candle(msg: Any):
        assert msg["e"] == "kline", "Should be a candle"
        raw_candle = msg["k"]
        return {
            "open time": int(raw_candle["t"]),
            "open": float(raw_candle["o"]),
            "high": float(raw_candle["h"]),
            "low": float(raw_candle["l"]),
            "close": float(raw_candle["c"]),
            "volume": float(raw_candle["v"]),
            "close time": int(raw_candle["T"]),
            "quote asset volume": float(raw_candle["q"]),
            "number of trades": int(raw_candle["n"]),
            "taker buy base asset volume": float(raw_candle["V"]),
            "taker buy quote asset volume": float(raw_candle["Q"]),
        }
