from lib.strategy import Strategy
import pandas as pd
from models.xgboost.model import XgboostBaseModel
from lib.tradingSignal import TradingSignal
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple


@dataclass
class First(Strategy):
    def __post_init__(self) -> None:
        self.models = (XgboostBaseModel(),)

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
        prediction = predictions[self.models[0]]

        signal: Optional[Tuple[TradingSignal, str]] = None

        if last_signal == TradingSignal.CLOSE and prediction > 1.5:
            signal = (TradingSignal.LONG, "The boos model predicts up")
        elif last_signal == TradingSignal.LONG and prediction < 0.5:
            signal = (TradingSignal.CLOSE, "the boost model predicts down")
        return signal
