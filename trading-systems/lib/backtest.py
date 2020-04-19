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
    ) -> pd.DataFrame:
        return Backtest._runWithTarget(
            strategy=strategy,
            features=features,
            targets=None,
            candlesticks=candlesticks,
            start_position=start_position,
            end_position=end_position,
            signals_csv_path=signals_csv_path,
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
    ) -> pd.DataFrame:
        """Test trading the target, without prediction. To check if the target is good."""
        if signals_csv_path is not None:
            print("Signals will be written continuasly to: " + signals_csv_path)
        init_features = features.iloc[:start_position]
        init_candlesticks = candlesticks.iloc[:start_position]
        strategy.init(candlesticks=init_candlesticks, features=init_features)

        trades = pd.DataFrame(columns=["transactTime", "signal", "price"])
        if signals_csv_path is not None:
            trades.to_csv(signals_csv_path)
        for position in range(start_position, end_position):
            period_features = features.iloc[:position]
            period_candlesticks = candlesticks.iloc[:position]

            # go to next candle
            signal = (
                strategy.on_candlestick_with_features_and_perdictions(
                    candlesticks=period_candlesticks,
                    features=period_features,
                    trades=trades,
                    predictions={
                        key: series[:position].tail(1).values[0] for key, series in targets.items()
                    },
                )
                if targets is not None
                else strategy.on_candlestick_with_features(
                    candlesticks=period_candlesticks, features=period_features, trades=trades
                )
            )
            if signal in (TradingSignal.BUY, TradingSignal.SELL):
                period_candlesticks = candlesticks.iloc[:position]
                NEXT_PERIODE = candlesticks.iloc[: position + 1].tail(1)
                trade_price = NEXT_PERIODE["open"].values[0]
                time = pd.to_datetime(period_candlesticks["close time"].tail(1), unit="ms").values[
                    0
                ]
                trades = trades.append(
                    {"transactTime": time, "signal": signal, "price": trade_price},
                    ignore_index=True,
                )
                if signals_csv_path is not None:
                    trades.tail(1).to_csv(signals_csv_path, header=False, mode="a")
                # check if take profit or stop loss should be executed before getting next periode
                if strategy.need_ticks(signal) and strategy.stop_loss is not None:
                    NEXT_PERIOD_LOW: float = NEXT_PERIODE["low"].values[0]
                    if NEXT_PERIOD_LOW < strategy.stop_loss:
                        imagened_trade_price = strategy.stop_loss * 0.99  # slippage 1%
                        signal = strategy.on_tick(imagened_trade_price, signal)

                        if signal is not None:
                            print(
                                f"Stoploss executed: Trade price: {imagened_trade_price}, low was: {NEXT_PERIOD_LOW}"
                            )
                            trades = trades.append(
                                {"transactTime": time, "signal": signal, "price": trade_price},
                                ignore_index=True,
                            )
                            if signals_csv_path is not None:
                                trades.tail(1).to_csv(signals_csv_path, header=False, mode="a")
            if position % 100 == 0:
                print(
                    f"""Backtest - position: {position-start_position} of {end_position-start_position}, number of signals: {len(trades)}"""  # noqa:  E501
                )
        return trades

    @staticmethod
    def evaluate(
        signals: pd.DataFrame,
        candlesticks: pd.DataFrame,
        start_position: int,
        end_position: int,
        fee: float,
    ) -> pd.DataFrame:
        candlesticks_periode = candlesticks.iloc[start_position:end_position]
        start_time = pd.to_datetime(candlesticks_periode["open time"].head(1), unit="ms").values[0]
        end_time = pd.to_datetime(candlesticks_periode["close time"].tail(1), unit="ms").values[0]
        start_price = candlesticks_periode["open"].head(1).values[0]
        end_price = candlesticks_periode["close"].tail(1).values[0]
        percentage_price_change_in_period = (100.0 / start_price) * (end_price - start_price)

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
            ]
        )

        holding = 0
        money = start_price
        open_time = 0
        open_price = 0
        open_money = 0
        for index, row in signals.iterrows():
            signal = row["signal"]
            price = row["price"]
            time = pd.to_datetime(row["transactTime"])

            if index == len(signals) and holding != 0:
                signal = TradingSignal.SELL

            if signal == TradingSignal.BUY:
                open_time = time
                open_price = price
                open_money = money
                holding = (money - money * fee) / price
                money = 0
            elif signal == TradingSignal.SELL:
                money = (holding - holding * fee) * price
                holding = 0
                trades = trades.append(
                    {
                        "open time": open_time,
                        "close time": time,
                        "time in position": time - open_time,
                        "open price": open_price,
                        "close price": price,
                        "change": price - open_price,
                        "change %": (100.0 / open_price) * (price - open_price),
                        "open money": open_money,
                        "close money": money,
                    },
                    ignore_index=True,
                )

        start_money = trades["open money"].head(1).values[0]
        end_money = trades["close money"].tail(1).values[0]

        print(f"Starts with {start_money}$ at {start_time}")
        print(f"Ends with {end_money}$ (number of trades: {len(signals)}) at {end_time}")
        print(
            f"Earned {end_money-start_money}$ ({round((100.0/start_money)*(end_money-start_money),2)}%)"
        )
        print(f"Percentage price change in period: {round(percentage_price_change_in_period,2)}%")

        return trades


def set_up_strategy_tmp_path(strategy_dir: str) -> str:
    print("strategy_dir: " + strategy_dir)
    new_dir = strategy_dir + "/tmp/" + datetime.today().strftime("%Y-%m-%d-%H:%M:%S") + "/"
    create_directory_if_not_exists(new_dir)
    return new_dir
