from lib.strategy import Strategy
import pandas as pd
from models.lightgbm import RegressionBabyMinModel, RegressionBabyMaxModel
from lib.tradingSignal import TradingSignal
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple


@dataclass
class Baby(Strategy):
    stop_loss_treshold: float = 0.95

    def __post_init__(self) -> None:
        self.models = (
            RegressionBabyMinModel(),
            RegressionBabyMaxModel(),
        )

    def on_candlestick_with_features_and_perdictions(
        self,
        candlesticks: pd.DataFrame,
        features: pd.DataFrame,
        trades: pd.DataFrame,
        predictions: Dict[Any, float],
    ) -> Optional[Tuple[TradingSignal, str]]:
        last_time, last_signal, last_price = self.get_last_executed_trade(trades)
        if last_signal is None:
            last_signal = TradingSignal.CLOSE

        lowest_min_prediction = predictions[self.models[0]]
        highest_high_prediction = predictions[self.models[1]]

        print(f"lastsignal: {last_signal}")
        print(f"lowestlow: {lowest_min_prediction}")
        print(f"highesthigh: {highest_high_prediction}")

        signal: Optional[Tuple[TradingSignal, str]] = None
        if last_signal == TradingSignal.CLOSE and highest_high_prediction > (
            2.0 * lowest_min_prediction * -1
        ):
            current_price = features.tail(1)["close"].values[0]
            self.stop_loss = current_price * self.stop_loss_treshold
            signal = (
                TradingSignal.LONG,
                "Its predicted that the highest high will be more then two times the lowest low",
            )
        elif (
            last_signal == TradingSignal.LONG
            and (lowest_min_prediction * -1) > highest_high_prediction
        ):
            signal = (
                TradingSignal.CLOSE,
                "Its predicted that the lowest low is larger then the highest high",
            )
        return signal
