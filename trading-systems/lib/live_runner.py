from lib.position import Position
from binance.client import Client
from typing import Any, Optional, Dict, List
from binance.websockets import BinanceSocketManager
import math
import pandas as pd

from lib.strategy import Strategy
from lib.tradingSignal import TradingSignal
from lib import data_util
import time
from dataclasses import dataclass, field
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
    active_stoploss_and_take_profit_order_ids: List[str] = field(default_factory=list)

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
            new_current_position = self.get_current_position()
            trades = data_util.load_trades(
                instrument=self.tradingpair,
                interval=self.candlestick_interval,
                trading_strategy_instance_name=self.trading_strategy_instance_name,
            )
            last_trade_signal_saved = None if trades is None else trades.tail(1)
            print(f"Last saved trade is: {last_trade_signal_saved}")
            # assert last_trade_signal_saved is None or (
            #     last_trade_signal_saved["signal"].values[0] == str(new_current_position)
            # ), "The last trade in the trades dataframe and the current position on the exchange should be the same."
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

        self.decimals_asset_quantity = self.getDecimalsForCoinQuantity(
            self.asset + self.base_asset
        )
        self.decimals_base_asset_price = self.getDecimalsForCoinPrice(self.base_asset)

        print(
            f"Number of decimals for asset price (in {self.base_asset}):",
            self.decimals_base_asset_price,
        )
        print(
            f"Number of decimals for quantity (in {self.asset}):",
            self.decimals_asset_quantity,
        )

        print(f"Current position is (last signal): {self.current_position}")
        self.binance_socket_manager = BinanceSocketManager(
            self.binance_client, user_timeout=5 * 60
        )
        # start any sockets here, i.e a trade socket

        open_orders = self.binance_client.get_open_orders(symbol=self.tradingpair)

        print("Current open orders:")
        for order in open_orders:
            print(order)

        self.active_stoploss_and_take_profit_order_ids = [
            order["orderId"]
            for order in open_orders
            if ("STOP_LOSS" in order["type"] or "TAKE_PROFIT" in order["type"])
        ]
        print(
            f"There are {len(self.active_stoploss_and_take_profit_order_ids)} open stop-loss/take-profit orders for {self.tradingpair}:"
        )

        connection_key = self.binance_socket_manager.start_kline_socket(
            self.tradingpair, self.process_message, interval=self.candlestick_interval,
        )
        # then start the socket manager
        self.binance_socket_connection_key = connection_key
        self.binance_socket_manager.start()
        return connection_key

    def getDecimalsForCoinQuantity(self, instrument: str):
        info: Any = self.binance_client.get_symbol_info(instrument)
        step_size = [
            float(_["stepSize"])
            for _ in info["filters"]
            if _["filterType"] == "LOT_SIZE"
        ][0]
        step_size_str = "%.8f" % step_size
        step_size_str = step_size_str.rstrip("0")
        decimals = len(step_size_str.split(".")[1])
        return decimals

    def getDecimalsForCoinPrice(self, coin: str):
        if coin == "USDT":
            return 2
        info: Any = self.binance_client.get_symbol_info("%sUSDT" % coin)
        step_size = [
            float(_["stepSize"])
            for _ in info["filters"]
            if _["filterType"] == "LOT_SIZE"
        ][0]
        step_size_str = "%.8f" % step_size
        step_size_str = step_size_str.rstrip("0")
        decimals = len(step_size_str.split(".")[1])
        return decimals

    def get_current_position(self) -> TradingSignal:
        asset_balance_res: Any = self.binance_client.get_asset_balance(asset=self.asset)
        asset_balance = float(asset_balance_res["free"]) + float(
            asset_balance_res["locked"]
        )
        base_asset_balance = self.binance_client.get_asset_balance(
            asset=self.base_asset
        )
        print(
            f"Current position (from exchange): Asset: {asset_balance_res}, Base asset: {base_asset_balance}"
        )
        if float(asset_balance) > self.strategy.min_value_asset:
            print("Current position (from exchange)(last signal executed): BUY")
            return TradingSignal.LONG
        else:
            print("Current position (from exchange)(last signal executed): SELL")
            return TradingSignal.CLOSE

    def process_message(self, msg: Any):
        if msg["e"] == "error":
            print("")
            print(f"Error from websocket: {msg['m']}")
            # close and restart the socket
            self.binance_socket_manager.close()
            self.start()
        else:
            new_position: Optional[Position] = None
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
                        "WARNING: there are no trades for this trading system yet. The current position on the exchange"
                        " needs to match whats specified in the strategy if there are no trades."
                    )
                asset_balance_res: Any = self.binance_client.get_asset_balance(
                    asset=self.asset
                )
                asset_balance = float(asset_balance_res["free"]) + float(
                    asset_balance_res["locked"]
                )
                base_asset_balance_res: Any = self.binance_client.get_asset_balance(
                    asset=self.base_asset
                )
                base_asset_balance = float(base_asset_balance_res["free"])

                print(f"Base asset ({self.base_asset}) balance: {base_asset_balance}")
                print(f"Asset ({self.asset}) balance: {asset_balance}")

                active_take_profit_stop_price: Optional[float] = None
                active_stop_loss_stop_price: Optional[float] = None

                for order_id in self.active_stoploss_and_take_profit_order_ids:
                    order = self.binance_client.get_order(
                        symbol=self.tradingpair, orderId=order_id
                    )
                    order_type = order["type"]
                    if order["status"] == "FILLED":
                        print(f"{order_type} order executed at exchange")
                        print(order)
                        new_trade_dict = self.order_to_trade(
                            order, TradingSignal.CLOSE, f"{order_type} executed"
                        )

                        data_util.add_trade(
                            instrument=self.tradingpair,
                            interval=self.candlestick_interval,
                            trading_strategy_instance_name=self.trading_strategy_instance_name,
                            new_trade_dict=new_trade_dict,
                        )
                        print(
                            f"An active {order_type} order has been filled. Its now added to the trades list."
                        )
                        self.active_stoploss_and_take_profit_order_ids.remove(order_id)
                    else:
                        print("Open order:")
                        print(order)
                        if "STOP_LOSS" in order_type:
                            active_stop_loss_stop_price = float(order["stopPrice"])
                        elif "TAKE_PROFIT" in order_type:
                            active_take_profit_stop_price = float(order["stopPrice"])
                        

                status = {
                    "asset_balance": asset_balance,
                    "base_asset_balance": base_asset_balance,
                    "take_profit_price": active_take_profit_stop_price,
                    "stop_loss_price": active_stop_loss_stop_price
                }

                new_position = self.strategy.on_candlestick(
                    self.candlesticks, trades, status
                )
                print("*", end="", flush=True)
            else:
                new_position = self.strategy.on_tick(
                    current_close_price, self.current_position
                )
                if new_position is not None:
                    time.sleep(10)
                    try:

                        asset_balance_res: Any = self.binance_client.get_asset_balance(
                            asset=self.asset
                        )
                        quantity_asset = float(asset_balance_res["free"])
                        if quantity_asset < self.strategy.min_value_asset:
                            new_position = None
                        else:
                            print(
                                "WARNING!! Still has free balance event though it should have been sold in a stop-loss. Will sell it now!"
                            )
                    except:
                        print(
                            "WARNING: an exception ocurred while checking if we need to execute the stoploss manually."
                            f" Its no big deal (it will try to execute the stoploss)."
                        )

            if new_position is not None:
                self.place_order(
                    new_position=new_position, last_price=current_close_price
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

    def place_order(
        self, new_position: Position, last_price: float,
    ):
        print(f"Taking new position")
        print(new_position)
        order: Dict[str, Any]
        if new_position.signal == TradingSignal.LONG:
            # buy as much we can with the available
            asset_balance_res: Any = self.binance_client.get_asset_balance(
                asset=self.base_asset
            )
            quantity_base_asset = float(asset_balance_res["free"])
            worst_acceptable_price = float(last_price) * 1.02  # max 2 % up
            # worst_acceptable_price_str = f"{worst_acceptable_price:.8f}"[:-2]
            worst_acceptable_price_str = self.round_price(worst_acceptable_price)
            quantity = float(quantity_base_asset) / worst_acceptable_price
            # make sure we round down by removing the last two digits
            # https://github.com/sammchardy/python-binance/issues/219
            # quantity_str = f"{quantity:.8f}"[:-2]
            quantity_str = self.round_quantity(quantity)

            if quantity_str > self.strategy.min_value_asset:
                print(
                    f"ORDER: Limit buy! Buy {quantity_str} {self.asset} at a maximum price of"
                    f" {worst_acceptable_price_str} {self.base_asset}"
                )
                order = self.binance_client.order_limit_buy(
                    symbol=self.tradingpair,
                    quantity=quantity_str,
                    timeInForce="GTC",
                    price=worst_acceptable_price_str,
                )
            else:
                print(
                    f"WARNING: Order not executed!! ERROR: quantity ({quantity_str}) i below min_value_asset"
                    f" ({self.strategy.min_value_asset}). Perhaps the min_value_base_asset should be adjusted in the configs file for the startegy."
                )
        elif new_position.signal == TradingSignal.CLOSE:
            open_orders = self.binance_client.get_open_orders(symbol=self.tradingpair)
            print(f"Closing {len(open_orders)} open orders for {self.tradingpair}")
            for open_order in open_orders:
                self.binance_client.cancel_order(
                    symbol=self.tradingpair, orderId=open_order["orderId"]
                )

            self.wait_for_orders(open_orders)
            print(f"All open orders for {self.tradingpair} closed")
            self.active_stoploss_and_take_profit_order_ids.clear()
            asset_balance_res: Any = self.binance_client.get_asset_balance(
                asset=self.asset
            )
            quantity = float(asset_balance_res["free"])
            # make sure we round down https://github.com/sammchardy/python-binance/issues/219
            # quantity_str = f"{quantity:.8f}"[:-2]
            quantity_str = self.round_quantity(quantity)
            worst_acceptable_price = float(last_price) * 0.97  # max 3 % down
            # worst_acceptable_price_str = f"{worst_acceptable_price:.8f}"[:-2]
            worst_acceptable_price_str = self.round_price(worst_acceptable_price)

            if quantity_str > self.strategy.min_value_asset:
                print(
                    f"ORDER: Limit sell! Sell {quantity_str} {self.asset} at a minimum price of ",
                    f" {worst_acceptable_price_str} {self.base_asset}.",
                )
                order = self.binance_client.order_limit_sell(
                    symbol=self.tradingpair,
                    quantity=quantity_str,
                    timeInForce="GTC",
                    price=worst_acceptable_price_str,
                )
            else:
                print(
                    f"WARNING: Order not executed!! ERROR: quantity ({quantity_str}) i below min_value_asset"
                    f" ({self.strategy.min_value_asset}). This should not be possible."
                )

        assert order is not None, "Order should not be none here."

        order = self.wait_for_orders([order])[0]

        if order["status"] == "FILLED":
            print("Order successfully executed!:")
            print(order)
            new_trade_dict = self.order_to_trade(
                order, new_position.signal, new_position.reason, new_position.data
            )
            print("Trade result:")
            print(new_trade_dict)

            data_util.add_trade(
                instrument=self.tradingpair,
                interval=self.candlestick_interval,
                trading_strategy_instance_name=self.trading_strategy_instance_name,
                new_trade_dict=new_trade_dict,
            )
            self.current_position = new_position.signal

            quantity = (
                self.round_quantity(
                    float(new_trade_dict["executedQty"])
                    - float(new_trade_dict["commission"])
                )
                if new_trade_dict["commissionAsset"] == self.asset
                else self.round_quantity(new_trade_dict["executedQty"])
            )

            if (
                new_position.signal == TradingSignal.LONG
                and new_position.stop_loss_take_profit is not None
            ):
                stop_loss_price = new_position.stop_loss_take_profit.stop_loss
                take_profit_price = new_position.stop_loss_take_profit.take_profit
                if stop_loss_price is not None:
                    order_res = self.binance_client.create_order(
                        symbol=self.tradingpair,
                        side=self.binance_client.SIDE_SELL,
                        type=self.binance_client.ORDER_TYPE_STOP_LOSS_LIMIT,
                        timeInForce=self.binance_client.TIME_IN_FORCE_GTC,
                        quantity=quantity,
                        price=self.round_price(stop_loss_price * 0.97),
                        stopPrice=self.round_price(stop_loss_price),
                    )

                    print("Stoploss order created at exchange:")
                    print(order_res)
                    self.active_stoploss_and_take_profit_order_ids.append(
                        order_res["orderId"]
                    )

                if take_profit_price is not None:
                    order_res = self.binance_client.create_order(
                        symbol=self.tradingpair,
                        side=self.binance_client.SIDE_SELL,
                        type=self.binance_client.ORDER_TYPE_TAKE_PROFIT_LIMIT,
                        timeInForce=self.binance_client.TIME_IN_FORCE_GTC,
                        quantity=quantity,
                        price=self.round_price(take_profit_price * 0.99),
                        stopPrice=self.round_price(take_profit_price),
                    )

                    print("Take profit order created at exchange:")
                    print(order_res)
                    self.active_stoploss_and_take_profit_order_ids.append(
                        order_res["orderId"]
                    )

        else:
            print("Order failed! :")
            print(order)

    def wait_for_orders(
        self,
        orders: List[Any],
        number_of_retry: int = 4,
        sleep_seconds: int = 20,
        accepted_statuses: List[str] = ["FILLED", "CANCELED", "REJECTED", "EXPIRED"],
    ):
        final_orders: List[Any] = []
        for order in orders:
            sleep_times = 0
            while (
                order["status"] not in accepted_statuses
                and sleep_times <= number_of_retry
            ):
                sleep_times = sleep_times + 1
                if sleep_times > number_of_retry:
                    self.binance_client.cancel_order(
                        symbol=self.tradingpair, orderId=order["orderId"]
                    )
                    print(
                        f"The order was cancled. Because it was not filled within {2*3} min"
                    )
                else:
                    time.sleep(sleep_seconds)
                    order = self.binance_client.get_order(
                        symbol=self.tradingpair, orderId=order["orderId"]
                    )
                    print("ORDER status: " + order["status"])
            final_orders = final_orders + [order]

        return final_orders

    @staticmethod
    def order_to_trade(
        order: Any, signal: TradingSignal, reason: str = "", data: Any = {}
    ):
        if order.get("fills", None) is None:
            avg_price = order.get("stopPrice", order.get("price", -1))
            acc_commission = -1.0
            commission_asset = "BNB(?)"
        else:
            acc_price, acc_quantity, acc_commission = reduce(
                lambda acc, item: (
                    acc[0] + float(item["price"]) * float(item["qty"]),
                    acc[1] + float(item["qty"]),
                    acc[2] + float(item["commission"]),
                ),
                order["fills"],
                (0.0, 0.0, 0.0),
            )
            commission_asset = order["fills"][0]["commissionAsset"]
            avg_price = acc_price / acc_quantity

        time = order.get("transactTime", order.get("updateTime", -1))

        return {
            "orderId": int(order.get("orderId", -1)),
            "transactTime": pd.to_datetime(
                time, unit="ms"
            ),  # datetime.datetime.fromtimestamp(float(order["transactTime"]) / 1000.0),
            "price": avg_price,
            "signal": str(signal),
            "origQty": float(order.get("origQty", -1)),
            "executedQty": float(order.get("executedQty", -1)),
            "cummulativeQuoteQty": float(order.get("cummulativeQuoteQty", -1)),
            "timeInForce": str(order.get("timeInForce", -1)),
            "commissionAsset": commission_asset,
            "commission": acc_commission,
            "type": str(order.get("type", -1)),
            "side": str(order.get("side", -1)),
            "reason": str(reason),
            "data": data,
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
