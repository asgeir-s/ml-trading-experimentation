from lib.strategy import Strategy
import pandas as pd
from models.xgboost import RegressionSklienModel
from models.xgboost import ClassifierUpDownModel
from lib.tradingSignal import TradingSignal
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple


@dataclass
class Third(Strategy):
    def __post_init__(self) -> None:
        self.models = (ClassifierUpDownModel(), RegressionSklienModel())

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

        up_down_model_prediction = predictions[self.models[0]]
        sklien_prediction = predictions[self.models[1]]

        signal: Optional[Tuple[TradingSignal, str]] = None
        if (
            last_signal == TradingSignal.SELL
            and sklien_prediction > 1.9
            and up_down_model_prediction == 1
        ):
            current_price = features.tail(1)["close"].values[0]
            self.stop_loss = current_price * 0.95
            signal = (TradingSignal.BUY, "Sklien and up down classifier indicate up")
        elif (
            last_signal == TradingSignal.BUY
            and sklien_prediction < 0
            and up_down_model_prediction == 0
        ):
            signal = (TradingSignal.SELL, "sklien and up down classifier indicate down")
        return signal
