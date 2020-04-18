from lib.strategy import Strategy
import pandas as pd
from models.xgboost import ClassifierUpDownModel
from lib.tradingSignal import TradingSignal
from dataclasses import dataclass
from typing import Optional, Dict, Any


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
    ) -> Optional[TradingSignal]:
        last_time, last_signal, last_price = self.get_last_trade(trades)
        if last_signal is None:
            last_signal = TradingSignal.SELL

        """         print("last signal:")
        print(last_signal)
        print("predictions:")
        print(predictions)
        print("this model:")
        print(self.models[0])
        print("this prediction:")
        print(predictions[self.models[0]]) """

        signal: Optional[TradingSignal] = None
        if last_signal == TradingSignal.SELL and predictions[self.models[0]] == 1:
            current_price = features.tail(1)["close"].values[0]
            self.stop_loss = current_price * 0.95
            signal = TradingSignal.BUY
        elif last_signal == TradingSignal.BUY and predictions[self.models[0]] == 0:
            signal = TradingSignal.SELL
        return signal
