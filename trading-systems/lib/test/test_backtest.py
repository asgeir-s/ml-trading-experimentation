from lib.backtest import Backtest
import pandas as pd
import numpy as np
from lib.strategy import Strategy
from lib.data_splitter import split_features_and_target_into_train_and_test_set
from dataclasses import InitVar
from models.xgboost.model import XgboostNovice
from lib.tradingSignal import TradingSignal
from dataclasses import dataclass
import abc
from typing import List

data = [
    [0, 100, 110, 90, 100, 50, 2, 200, 200, 44, 400],
    [3, 100, 110, 90, 110, 50, 4, 200, 200, 44, 400],
    [5, 90, 110, 90, 90, 50, 6, 200, 200, 44, 400],
    [7, 110, 110, 90, 100, 50, 8, 200, 200, 44, 400],
    [9, 100, 110, 90, 100, 50, 10, 200, 200, 44, 400],
    [11, 100, 110, 90, 100, 50, 12, 200, 200, 44, 400],
    [13, 100, 110, 90, 100, 50, 14, 200, 200, 44, 400],
    [15, 100, 110, 90, 110, 50, 16, 200, 200, 44, 400],
    [17, 90, 110, 90, 90, 50, 18, 200, 200, 44, 400],
    [19, 110, 110, 90, 100, 50, 20, 200, 200, 44, 400],
    [21, 100, 110, 90, 100, 50, 22, 200, 200, 44, 400],
]

candlesticks = pd.DataFrame(
    data,
    columns=[
        "open time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close time",
        "quote asset volume",
        "number of trades",
        "taker buy base asset volume",
        "taker buy quote asset volume",
    ],
)

@dataclass
class TestStrategy(Strategy):
    init_candlesticks: InitVar[pd.DataFrame]
    def __post_init__(self, init_features: pd.DataFrame) -> None:
        pass

    def execute(self, candlesticks: pd.DataFrame, trades: pd.DataFrame) -> TradingSignal:
        features = self.generate_features(candlesticks)
        return self.execute_with_features(features, trades)

    def execute_with_features(self, features: pd.DataFrame, signals: pd.DataFrame) -> TradingSignal:
        prediction = 1 if features.tail(1)["close"].values[0] > 100 else -1 if features.tail(1)["close"].values[0] < 100 else 0
        return self._execute(features, signals, [prediction])

    def _execute(
        self, features: pd.DataFrame, signals: pd.DataFrame, predictions: List[float]
    ) -> TradingSignal:
        print(predictions)
        last_signal = (
            TradingSignal.SELL if len(signals) == 0 else signals.tail(1)["signal"].values[0]
        )
        if predictions[0] == 1 and last_signal == TradingSignal.SELL:
            return TradingSignal.BUY
        elif predictions[0] == -1 and last_signal == TradingSignal.BUY:
            return TradingSignal.SELL

    @staticmethod
    def generate_features(candlesticks: pd.DataFrame) -> pd.DataFrame:
        return candlesticks[["open", "close"]]

    def __train(self, features: pd.DataFrame):
        pass

    @staticmethod
    def _generate_target(features: pd.DataFrame) -> pd.DataFrame:
        up_treshold = 100
        down_treshold = 100
        conditions = [
            features["close"] > up_treshold,
            features["close"] < down_treshold,
        ]
        choices = [1, -1]
        return pd.Series(np.select(conditions, choices, default=0))


def test_answer():
    features = TestStrategy.generate_features(candlesticks)
    targets = TestStrategy._generate_target(features)

    assert len(features) == len(targets), "features and target has the same length"

    trade_start_position = 1
    trade_end_position = len(features)

    signals = Backtest._runWithTarget(TestStrategy, features, targets ,candlesticks, trade_start_position, trade_end_position)

    print(signals)

    assert len(signals) == 4, "there should be 4 signals"

    trades = Backtest.evaluate(signals, candlesticks, trade_start_position, trade_end_position, 1000)

    print(trades)

    end_money = trades.tail(1)["close money"].values[0]

    assert len(trades) == 2, "There should be 2 trades."
    assert end_money > 1493 and end_money < 1494, "The ending amout of money should be 1493,82716049382716"
