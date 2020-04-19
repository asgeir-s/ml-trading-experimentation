from lib.strategy import Strategy
import pandas as pd
from models.xgboost.model import XgboostBaseModel
from lib.tradingSignal import TradingSignal
from dataclasses import dataclass
from typing import Optional, Dict, Any


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
    ) -> Optional[TradingSignal]:
        last_time, last_signal, last_price = self.get_last_trade(trades)
        if last_signal is None:
            last_signal = TradingSignal.SELL
        prediction = predictions[self.models[0]]

        signal: Optional[TradingSignal] = None

        if last_signal == TradingSignal.SELL and prediction > 1.5:
            signal = TradingSignal.BUY
        elif last_signal == TradingSignal.BUY and prediction < 0.5:
            signal = TradingSignal.SELL
        return signal
