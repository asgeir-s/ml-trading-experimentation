from lib.strategy import Strategy
import pandas as pd
from models.xgboost import ClassifierUpDownModel
from models.xgboost import RegressionSklienModel
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
        last_features = features.tail(1)
        if last_signal is None:
            last_signal = TradingSignal.SELL

        up_down_model_prediction = predictions[self.models[0]]
        sklien_regressor_model_prediction = predictions[self.models[1]]

        signal: Optional[Tuple[TradingSignal, str]] = None
        if (
            last_signal == TradingSignal.SELL
            and up_down_model_prediction == 1
            and last_features["momentum_roc-30"].values[0] > 0
            and sklien_regressor_model_prediction > 2 # 1 is pritty good
        ):
            ##current_price = last_features["close"].values[0]
            ##self.stop_loss = current_price * 0.96
            signal = (TradingSignal.BUY, "Sklien and up down classifier indicate up")
        elif (
            last_signal == TradingSignal.BUY
            and up_down_model_prediction == 0
            and last_features["momentum_roc-30"].values[0] < 0
            and sklien_regressor_model_prediction < -1
        ):
            signal = (TradingSignal.SELL, "sklien and up down classifier indicate down")
        return signal
