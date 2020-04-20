from lib.backtest import Backtest
import pandas as pd
import numpy as np
from lib.strategy import Strategy
from lib.tradingSignal import TradingSignal
from dataclasses import dataclass
from typing import Optional, Dict, Any

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


@dataclass  # typing: ignore
class TestStrategy(Strategy):
    def __post_init__(self) -> None:
        pass

    def on_candlestick_with_features(
        self, candlesticks: pd.DataFrame, features: pd.DataFrame, trades: pd.DataFrame
    ) -> Optional[Tuple[TradingSignal, str]]:
        prediction = (
            1.0
            if features.tail(1)["close"].values[0] > 100
            else -1.0
            if features.tail(1)["close"].values[0] < 100
            else 0
        )
        print("pred: " + str(prediction))
        return self.on_candlestick_with_features_and_perdictions(
            candlesticks=candlesticks,
            features=features,
            trades=trades,
            predictions={0: float(prediction)},
        )

    def on_candlestick_with_features_and_perdictions(
        self,
        candlesticks: pd.DataFrame,
        features: pd.DataFrame,
        trades: pd.DataFrame,
        predictions: Dict[Any, float],
    ) -> Optional[TradingSignal]:
        print("predictions:")
        prediction = int(predictions[0])
        print(prediction)
        last_time, last_signal, last_price = self.get_last_trade(trades)
        if last_signal is None:
            last_signal = TradingSignal.SELL
        signal: Optional[TradingSignal] = None
        if prediction == 1 and last_signal == TradingSignal.SELL:
            signal = TradingSignal.BUY
        elif prediction == -1 and last_signal == TradingSignal.BUY:
            signal = TradingSignal.SELL
        return signal

    def generate_features(self, candlesticks: pd.DataFrame) -> pd.DataFrame:
        return candlesticks[["open", "close"]]

    def __train(self, features: pd.DataFrame):
        pass

    def _generate_targets(
        self, candlesticks: pd.DataFrame, features: pd.DataFrame
    ) -> Dict[Any, pd.Series]:
        up_treshold = 100
        down_treshold = 100

        conditions = [
            features["close"] > up_treshold,
            features["close"] < down_treshold,
        ]
        choices = [1, -1]
        return {0: pd.Series(np.select(conditions, choices, default=0))}


def test_backtest():
    strategy = TestStrategy()
    features = strategy.generate_features(candlesticks)
    targets = strategy._generate_targets(candlesticks, features)

    assert len(features) == len(targets[0]), "features and target has the same length"

    trade_start_position = 1
    trade_end_position = len(features)

    signals = Backtest._runWithTarget(
        strategy=strategy,
        features=features,
        targets=targets,
        candlesticks=candlesticks,
        start_position=trade_start_position,
        end_position=trade_end_position,
    )

    print(signals)

    assert len(signals) == 4, "there should be 4 signals"

    trades = Backtest.evaluate(signals, candlesticks, trade_start_position, trade_end_position, 0.0)

    print(trades)

    end_money = trades.tail(1)["close money"].values[0]

    assert len(trades) == 2, "There should be 2 trades."
    assert (
        end_money > 149.3 and end_money < 149.4
    ), "The ending amount of money should be 1493,82716049382716"
