from lib.strategy import Strategy
import pandas as pd
from models.xgboost import ClassifierUpDownModel
from lib.tradingSignal import TradingSignal
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple


@dataclass
class Second(Strategy):
    def __post_init__(self) -> None:
        self.models = (ClassifierUpDownModel(),)

    def on_candlestick_with_features_and_perdictions(
        self,
        candlesticks: pd.DataFrame,
        features: pd.DataFrame,
        trades: pd.DataFrame,
        predictions: Dict[Any, float],
    ) -> Optional[Tuple[TradingSignal, str]]:
        last_time, last_signal, last_price = self.get_last_trade(trades)
        if last_signal is None:
            last_signal = TradingSignal.SELL

        signal: Optional[Tuple[TradingSignal, str]] = None
        if last_signal == TradingSignal.SELL and predictions[self.models[0]] == 1:
            current_price = features.tail(1)["close"].values[0]
            self.stop_loss = current_price * 0.95
            signal = (TradingSignal.BUY, "Classifier Up Down indicates up")
        elif last_signal == TradingSignal.BUY and predictions[self.models[0]] == 0:
            signal = (TradingSignal.SELL, "Classifier Up Down indicates down")
        return signal
