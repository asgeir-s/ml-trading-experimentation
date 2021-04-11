from lib.position import Position
import pandas as pd
from lib.strategy import Strategy
from lib.tradingSignal import TradingSignal
from lib.data_util import create_directory_if_not_exists
from typing import Optional, Any, Dict
from datetime import datetime


class Backtest:
    @staticmethod
    def run(
        strategy: Strategy,
        features: pd.DataFrame,
        candlesticks: pd.DataFrame,
        start_position: int,
        end_position: int,
        signals_csv_path: Optional[str] = None,
        trades_csv_path: Optional[str] = None,
        fee: float = 0.001,
    ) -> pd.DataFrame:
        return Backtest._runWithTarget(
            strategy=strategy,
            features=features,
            targets=None,
            candlesticks=candlesticks,
            start_position=start_position,
            end_position=end_position,
            signals_csv_path=signals_csv_path,
            trades_csv_path=trades_csv_path,
            fee=fee,
        )

    @staticmethod
    def _runWithTarget(
        strategy: Strategy,
        features: pd.DataFrame,
        targets: Optional[Dict[Any, pd.Series]],
        candlesticks: pd.DataFrame,
        start_position: int,
        end_position: int,
        signals_csv_path: Optional[str] = None,
        trades_csv_path: Optional[str] = None,
        fee: float = 0.001,
    ) -> pd.DataFrame:
        """Test trading the target, without prediction. To check if the target is good."""
        init_features = features.iloc[:start_position]
        init_candlesticks = candlesticks.iloc[:start_position]
        strategy.init(candlesticks=init_candlesticks, features=init_features)

        candlesticks_periode = candlesticks.iloc[start_position:end_position]
        start_time = pd.to_datetime(
            candlesticks_periode["open time"].head(1), unit="ms"
        ).values[0]
        end_time = pd.to_datetime(
            candlesticks_periode["close time"].tail(1), unit="ms"
        ).values[0]
        start_price = float(candlesticks_periode["open"].head(1).values[0])
        end_price = float(candlesticks_periode["close"].tail(1).values[0])
        percentage_price_change_in_period = (100.0 / start_price) * (
            end_price - start_price
        )

        trades = pd.DataFrame(
            columns=[
                "open time",
                "close time",
                "time in position",
                "open price",
                "close price",
                "change",
                "change %",
                "open money",
                "close money",
                "open reason",
                "close reason",
                "open data",
                "close data"
            ]
        )

        holding = 0.0
        money = start_price  # todo: could be something else
        open_time = 0
        open_price = 0
        open_money = 0
        open_reason = ""
        open_data = {}

        signals = pd.DataFrame(columns=["transactTime", "signal", "price", "reason", "data"])
        last_signal: TradingSignal = TradingSignal.CLOSE
        stop_loss: Optional[float] = None
        take_profit: Optional[float] = None
        if signals_csv_path is not None:
            print("Signals will be written continuasly to: " + signals_csv_path)
            signals.to_csv(signals_csv_path)
        if trades_csv_path is not None:
            print("Trades will be written continuasly to: " + trades_csv_path)
            trades.to_csv(trades_csv_path)
        for position in range(start_position, end_position):
            period_features = features.iloc[:position]
            period_candlesticks = candlesticks.iloc[:position]

            # go to next candle
            new_position = (
                strategy.on_candlestick_with_features_and_perdictions(
                    candlesticks=period_candlesticks,
                    features=period_features,
                    trades=signals,
                    predictions={
                        key: series[:position].tail(1).values[0]
                        for key, series in targets.items()
                    },
                    status={
                        "asset_balance": holding,
                        "base_asset_balance": money,
                        "take_profit_price": take_profit,
                        "stop_loss_price": stop_loss,
                    },
                )
                if targets is not None
                else strategy.on_candlestick_with_features(
                    candlesticks=period_candlesticks,
                    features=period_features,
                    trades=signals,
                    status={
                        "asset_balance": holding,
                        "base_asset_balance": money,
                        "take_profit_price": take_profit,
                        "stop_loss_price": stop_loss,
                    },
                )
            )
            if position == end_position and last_signal != TradingSignal.CLOSE:
                new_position = Position(
                    signal=TradingSignal.CLOSE,
                    reason=f"There is a open position at the end of the backtest. So we close it.",
                )

            NEXT_PERIODE = candlesticks.iloc[: position + 1].tail(1)
            if new_position is not None:

                if new_position.signal in (
                    TradingSignal.LONG,
                    TradingSignal.CLOSE,
                ):
                    trade_price = float(NEXT_PERIODE["open"].values[0])
                    time = pd.to_datetime(NEXT_PERIODE["open time"], unit="ms").values[0]
                    signal_data = {
                        "transactTime": time,
                        "signal": new_position.signal,
                        "price": trade_price,
                        "reason": new_position.reason,
                        "data": new_position.data
                    }
                    signals = signals.append(signal_data, ignore_index=True,)
                    last_signal = new_position.signal
                    # print(trad)
                    if signals_csv_path is not None:
                        signals.tail(1).to_csv(signals_csv_path, header=False, mode="a")
                    
                    if new_position.signal == TradingSignal.LONG:
                        print("signal is buy")
                        open_time = time
                        open_price = trade_price
                        open_money = money
                        open_reason = new_position.reason
                        open_data = new_position.data
                        holding = (money - (money * fee)) / trade_price
                        money = 0.0
                    elif new_position.signal == TradingSignal.CLOSE:
                        print("signal is sell")
                        money = (holding - (holding * fee)) * trade_price
                        holding = 0.0
                        print("appending trade")
                        trades = trades.append(
                            {
                                "open time": open_time,
                                "close time": time,
                                "time in position": time - open_time,
                                "open price": open_price,
                                "close price": trade_price,
                                "change": trade_price - open_price,
                                "change %": (100.0 / open_price) * (trade_price - open_price),
                                "open money": open_money,
                                "close money": money,
                                "open reason": open_reason,
                                "close reason": new_position.reason,
                                "open data": open_data,
                                "close data": new_position.data
                            },
                            ignore_index=True,
                        )
                        if trades_csv_path is not None:
                            trades.tail(1).to_csv(trades_csv_path, header=False, mode="a")
                    
                
                if new_position.stop_loss_take_profit is not None:
                    if new_position.stop_loss_take_profit.stop_loss is not None:
                        stop_loss = new_position.stop_loss_take_profit.stop_loss
                        print(f"Stop-loss set to: {stop_loss}")
                    if new_position.stop_loss_take_profit.take_profit is not None:
                        stop_loss = new_position.stop_loss_take_profit.take_profit
                        print(f"Take-profit set to: {take_profit}")

            # check if take profit or stop-loss should be executed before getting next periode
            if (
                last_signal == TradingSignal.LONG
                and (
                    stop_loss is not None
                    or take_profit is not None
                )
            ):
                NEXT_PERIOD_LOW: float = NEXT_PERIODE["low"].values[0]
                NEXT_PERIOD_HIGH: float = NEXT_PERIODE["high"].values[0]
                imagened_trade_price = 0
                new_position = None
                time = pd.to_datetime(NEXT_PERIODE["close time"], unit="ms").values[0]
                if stop_loss is not None and NEXT_PERIOD_LOW < stop_loss:
                    print("stoploss executing")
                    imagened_trade_price = stop_loss * 0.998  # slippage 0.5%
                    new_position = Position(
                        signal=TradingSignal.CLOSE,
                        reason=f"Stop-loss: price ({NEXT_PERIOD_LOW}) is below stop-loss ({stop_loss})",
                    )
                    stop_loss = None
                    take_profit = None
                elif (
                    take_profit is not None
                    and NEXT_PERIOD_HIGH > take_profit
                ):
                    print("take profit executing")
                    imagened_trade_price = take_profit * 0.998
                    new_position = strategy.on_tick(imagened_trade_price, last_signal)
                    new_position = Position(
                        signal=TradingSignal.CLOSE,
                        reason=f"Take-profit: price ({NEXT_PERIOD_HIGH}) is above take-profit ({take_profit})",
                    )
                    stop_loss = None
                    take_profit = None

                if new_position is not None:
                    print("creating trade")
                    signal_data = {
                        "transactTime": time,
                        "signal": new_position.signal,
                        "price": imagened_trade_price,
                        "reason": new_position.reason,
                        "data": new_position.data
                    }
                    signals = signals.append(signal_data, ignore_index=True,)
                    last_signal = new_position.signal
                    # print(trad)
                    if signals_csv_path is not None:
                        signals.tail(1).to_csv(signals_csv_path, header=False, mode="a")

                    money = (holding - (holding * fee)) * imagened_trade_price
                    holding = 0
                    print("appending trade")
                    trades = trades.append(
                        {
                            "open time": open_time,
                            "close time": time,
                            "time in position": time - open_time,
                            "open price": open_price,
                            "close price": imagened_trade_price,
                            "change": imagened_trade_price - open_price,
                            "change %": (100.0 / open_price) * (imagened_trade_price - open_price),
                            "open money": open_money,
                            "close money": money,
                            "open reason": open_reason,
                            "close reason": new_position.reason,
                            "open data": open_data,
                            "close data": new_position.data
                        },
                        ignore_index=True,
                    )
                    if trades_csv_path is not None:
                        trades.tail(1).to_csv(trades_csv_path, header=False, mode="a")

            if position % 100 == 0:
                precentage = (
                    100 / (end_position - start_position) * (position - start_position)
                )
                print(
                    f"""Backtest - {precentage:.2f}% done. Position: {position-start_position} of {end_position-start_position}, number of signals: {len(signals)}"""  # noqa:  E501
                )
        
                # print(trades)
        start_money = trades["open money"].head(1).values[0]
        end_money = trades["close money"].tail(1).values[0]

        print(f"Starts with {start_money}$ at {start_time}")
        print(
            f"Ends with {end_money}$ (number of trades: {len(signals)}) at {end_time}"
        )
        print(
            f"Earned {end_money-start_money}$ ({round((100.0/start_money)*(end_money-start_money),2)}%)"
        )
        print(
            f"Percentage price change in period: {round(percentage_price_change_in_period,2)}%"
        )
        return trades

    # @staticmethod
    # def evaluate(
    #     signals: pd.DataFrame,
    #     candlesticks: pd.DataFrame,
    #     start_position: int,
    #     end_position: int,
    #     fee: float,
    # ) -> pd.DataFrame:
    #     candlesticks_periode = candlesticks.iloc[start_position:end_position]
    #     start_time = pd.to_datetime(
    #         candlesticks_periode["open time"].head(1), unit="ms"
    #     ).values[0]
    #     end_time = pd.to_datetime(
    #         candlesticks_periode["close time"].tail(1), unit="ms"
    #     ).values[0]
    #     start_price = candlesticks_periode["open"].head(1).values[0]
    #     end_price = candlesticks_periode["close"].tail(1).values[0]
    #     percentage_price_change_in_period = (100.0 / start_price) * (
    #         end_price - start_price
    #     )

    #     trades = pd.DataFrame(
    #         columns=[
    #             "open time",
    #             "close time",
    #             "time in position",
    #             "open price",
    #             "close price",
    #             "change",
    #             "change %",
    #             "open money",
    #             "close money",
    #             "open reason",
    #             "close reason",
    #         ]
    #     )

    #     holding = 0
    #     money = start_price  # todo: could be something else
    #     open_time = 0
    #     open_price = 0
    #     open_money = 0
    #     open_reason = ""
    #     for index, row in signals.iterrows():
    #         signal = row["signal"]
    #         price = row["price"]
    #         reason = row["reason"]
    #         time = pd.to_datetime(row["transactTime"])
    #         print("signal:", signal)
    #         print("price:", price)
    #         print("reason:", reason)

    #         if signal not in (TradingSignal.LONG, TradingSignal.CLOSE):
    #             if "BUY" in signal:
    #                 signal = TradingSignal.LONG
    #             elif "SELL" in signal:
    #                 signal = TradingSignal.CLOSE

    #         if index == len(signals) and holding != 0:
    #             signal = TradingSignal.CLOSE

    #         if signal == TradingSignal.LONG:
    #             print("signal is buy")
    #             open_time = time
    #             open_price = price
    #             open_money = money
    #             open_reason = reason
    #             holding = (money - (money * fee)) / price
    #             money = 0
    #         elif signal == TradingSignal.CLOSE:
    #             print("signal is sell")
    #             money = (holding - (holding * fee)) * price
    #             holding = 0
    #             print("appending trade")
    #             trades = trades.append(
    #                 {
    #                     "open time": open_time,
    #                     "close time": time,
    #                     "time in position": time - open_time,
    #                     "open price": open_price,
    #                     "close price": price,
    #                     "change": price - open_price,
    #                     "change %": (100.0 / open_price) * (price - open_price),
    #                     "open money": open_money,
    #                     "close money": money,
    #                     "open reason": open_reason,
    #                     "close reason": reason,
    #                 },
    #                 ignore_index=True,
    #             )

    #     print(trades)
    #     start_money = trades["open money"].head(1).values[0]
    #     end_money = trades["close money"].tail(1).values[0]

    #     print(f"Starts with {start_money}$ at {start_time}")
    #     print(
    #         f"Ends with {end_money}$ (number of trades: {len(signals)}) at {end_time}"
    #     )
    #     print(
    #         f"Earned {end_money-start_money}$ ({round((100.0/start_money)*(end_money-start_money),2)}%)"
    #     )
    #     print(
    #         f"Percentage price change in period: {round(percentage_price_change_in_period,2)}%"
    #     )

    #     return trades


def set_up_strategy_tmp_path(strategy_dir: str) -> str:
    print("strategy_dir: " + strategy_dir)
    new_dir = (
        strategy_dir + "/tmp/" + datetime.today().strftime("%Y-%m-%d-%H:%M:%S") + "/"
    )
    create_directory_if_not_exists(new_dir)
    return new_dir


def setup_file_path(path: str):
    today = datetime.today()
    create_directory_if_not_exists(path)

    def csv_file(file_name: str, extension: str = "csv"):
        return (
            path
            + today.strftime("%Y-%m-%d-%H:%M:%S")
            + "-"
            + file_name
            + "."
            + extension
        )

    return csv_file
