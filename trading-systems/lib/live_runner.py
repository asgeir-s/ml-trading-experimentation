from binance.client import Client
from typing import Any, Optional, Dict, Tuple
from binance.websockets import BinanceSocketManager
import math

from tensorflow.python.ops.gen_spectral_ops import fft
from lib.strategy import Strategy
from lib.tradingSignal import TradingSignal
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
    decimals_asset_quantity: int = 6
    decimals_base_asset_price: int = 6

    __current_position: Optional[TradingSignal] = None
    binance_socket_manager: Any = None
    binance_socket_connection_key: Optional[str] = None

    @property
    def tradingpair(self) -> str:
        return self.asset + self.base_asset

    @property
    def current_position(self) -> TradingSignal:
        if self.__current_position is not None:
            return self.__current_position
        else:
            new_current_position = self.get_current_position(
                self.binance_client, self.asset, self.base_asset
            )
            trades = data_util.load_trades(
                instrument=self.tradingpair,
                interval=self.candlestick_interval,
                trading_strategy_instance_name=self.trading_strategy_instance_name,
            )
            last_trade_signal_saved = None if trades is None else trades.tail(1)
            print(f"Last saved trade is: {last_trade_signal_saved}")
            assert last_trade_signal_saved is None or (
                last_trade_signal_saved["signal"].values[0] == str(new_current_position)
            ), "The last trade in the trades dataframe and the current position on the exchange should be the same."
            self.current_position = new_current_position
            return self.current_position

    @current_position.setter  # noqa: F811
    def current_position(self, new_position: TradingSignal):
        self.__current_position = new_position

    def start(self) -> str:
        print(f"Tradingpair: {self.tradingpair}")
        self.candlesticks = data_util.load_candlesticks(
            instrument=self.tradingpair,
            interval=self.candlestick_interval,
            binance_client=self.binance_client,
        )

        self.decimals_asset_quantity = self.getDecimalsForCoinQuantity(self.asset + self.base_asset)
        self.decimals_base_asset_price = self.getDecimalsForCoinPrice(self.base_asset)

        print(
            f"Number of decimals for asset price (in {self.base_asset}):",
            self.decimals_base_asset_price,
        )
        print(f"Number of decimals for quantity (in {self.asset}):", self.decimals_asset_quantity)

        print(f"Current position is (last signal): {self.current_position}")
        self.binance_socket_manager = BinanceSocketManager(self.binance_client)
        # start any sockets here, i.e a trade socket

        connection_key = self.binance_socket_manager.start_kline_socket(
            self.tradingpair, self.process_message, interval=self.candlestick_interval,
        )
        # then start the socket manager
        self.binance_socket_connection_key = connection_key
        self.binance_socket_manager.start()
        return connection_key

    def getDecimalsForCoinQuantity(self, instrument: str):
        info = self.binance_client.get_symbol_info(instrument)
        step_size = [
            float(_["stepSize"]) for _ in info["filters"] if _["filterType"] == "LOT_SIZE"
        ][0]
        step_size = "%.8f" % step_size
        step_size = step_size.rstrip("0")
        decimals = len(step_size.split(".")[1])
        return decimals

    def getDecimalsForCoinPrice(self, coin: str):
        if coin == "USDT":
            return 2
        info = self.binance_client.get_symbol_info("%sUSDT" % coin)
        step_size = [
            float(_["stepSize"]) for _ in info["filters"] if _["filterType"] == "LOT_SIZE"
        ][0]
        step_size = "%.8f" % step_size
        step_size = step_size.rstrip("0")
        decimals = len(step_size.split(".")[1])
        return decimals

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
        if msg["e"] == "error":
            print("")
            print(f"Error from websocket: {msg['m']}")
            # close and restart the socket
            self.binance_socket_manager.close()
            self.start()
        else:
            signal_tuple: Optional[Tuple[TradingSignal, str]] = None
            candle_raw = msg["k"]
            current_close_price = float(candle_raw["c"])
            if candle_raw["x"] is True:
                self.candlesticks = data_util.add_candle(
                    instrument=self.tradingpair,
                    interval=self.candlestick_interval,
                    new_candle=self.msg_to_candle(msg),
                )
                trades = data_util.load_trades(
                    instrument=self.tradingpair,
                    interval=self.candlestick_interval,
                    trading_strategy_instance_name=self.trading_strategy_instance_name,
                )
                if trades is None or len(trades) == 0:
                    print(
                        "WARNING: there are no trades for this trading system yet. The current position on the exchange needs to match whats specified in the strategy if there are no trades."
                    )
                asset_balance = self.binance_client.get_asset_balance(asset=self.asset)["free"]
                base_asset_balance = self.binance_client.get_asset_balance(asset=self.base_asset)[
                    "free"
                ]

                status = {"asset_balance": asset_balance, "base_asset_balance": base_asset_balance}

                signal_tuple = self.strategy.on_candlestick(self.candlesticks, trades, status)
                print("*", end="", flush=True)
            else:
                signal_tuple = self.strategy.on_tick(current_close_price, self.current_position)

            if signal_tuple is not None:
                self.place_order(
                    signal=signal_tuple[0], last_price=current_close_price, reason=signal_tuple[1]
                )
            else:
                print(".", end="", flush=True)

    def round_quantity(self, quantity: float) -> float:
        return (
            math.floor(quantity * 10 ** self.decimals_asset_quantity)
            / 10 ** self.decimals_asset_quantity
        )

    def round_price(self, price: float) -> float:
        return (
            math.floor(price * 10 ** self.decimals_base_asset_price)
            / 10 ** self.decimals_base_asset_price
        )

    def place_order(self, signal: TradingSignal, last_price: float, reason: str):
        print(f"Placing new order: signal: {signal}")
        print(f"Reason: {reason}")
        order: Dict[str, Any]
        if signal == TradingSignal.BUY:
            # buy as much we can with the available
            quantity_base_asset = float(
                self.binance_client.get_asset_balance(asset=self.base_asset)["free"]
            )
            worst_acceptable_price = float(last_price) * 1.02  # max 2 % up
            # worst_acceptable_price_str = f"{worst_acceptable_price:.8f}"[:-2]
            worst_acceptable_price_str = self.round_price(worst_acceptable_price)
            quantity = float(quantity_base_asset) / worst_acceptable_price
            # make sure we round down by removing the last two digits https://github.com/sammchardy/python-binance/issues/219
            # quantity_str = f"{quantity:.8f}"[:-2]
            quantity_str = self.round_quantity(quantity)

            print(
                f"ORDER: Limit buy! Buy {quantity_str} {self.asset} at a maximum price of {worst_acceptable_price_str} {self.base_asset}"
            )
            order = self.binance_client.order_limit_buy(
                symbol=self.tradingpair,
                quantity=quantity_str,
                timeInForce="GTC",
                price=worst_acceptable_price_str,
            )
        elif signal == TradingSignal.SELL:
            quantity = float(self.binance_client.get_asset_balance(asset=self.asset)["free"])
            # make sure we round down https://github.com/sammchardy/python-binance/issues/219
            # quantity_str = f"{quantity:.8f}"[:-2]
            quantity_str = self.round_quantity(quantity)
            worst_acceptable_price = float(last_price) * 0.97  # max 3 % down
            # worst_acceptable_price_str = f"{worst_acceptable_price:.8f}"[:-2]
            worst_acceptable_price_str = self.round_price(worst_acceptable_price)
            print(
                f"ORDER: Limit sell! Sell {quantity_str} {self.asset} at a minimum price of {worst_acceptable_price_str} {self.base_asset}."
            )
            order = self.binance_client.order_limit_sell(
                symbol=self.tradingpair,
                quantity=quantity_str,
                timeInForce="GTC",
                price=worst_acceptable_price_str,
            )

        assert order is not None, "Order can not be none here."

        sleep_times = 0
        while order["status"] not in ("FILLED", "CANCELED", "REJECTED", "EXPIRED"):
            order = self.binance_client.get_order(symbol=self.tradingpair, orderId=order["orderId"])
            print("ORDER status: " + order["status"])
            if sleep_times >= 3:
                self.binance_client.cancel_order(symbol=self.tradingpair, orderId=order["orderId"])
                print(f"The order was cancled. Because it was not filled within {2*3} min")
                sleep_times = 0
            else:
                time.sleep(2 * 60)  # 2 min
                sleep_times = sleep_times + 1

        if order["status"] == "FILLED":
            print("Order successfully executed!:")
            print(order)

            data_util.add_trade(
                instrument=self.tradingpair,
                interval=self.candlestick_interval,
                trading_strategy_instance_name=self.trading_strategy_instance_name,
                new_trade_dict=self.order_to_trade(order, signal, reason),
            )
            self.current_position = signal

        else:
            print("Order failed! :")
            print(order)

    @staticmethod
    def order_to_trade(order: Any, signal: TradingSignal, reason: str = ""):
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
            "reason": str(reason),
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
