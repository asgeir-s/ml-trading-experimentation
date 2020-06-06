from lib.strategy import Strategy
import pandas as pd
from models.xgboost import ClassifierUpDownModel
from models.xgboost import RegressionSklienModel
from lib.tradingSignal import TradingSignal
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from datetime import datetime


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
        momentum_roc_30 = last_features["momentum_roc-30"].values[0]

        now = datetime.now()
        current_time = now.strftime("%d %B %Y, %H:%M:%S")

        print("")
        print("EVALUATION")
        print(f"last signal: {last_signal}")
        print(f"time: {current_time}")
        print(f"up_down_model_prediction: {up_down_model_prediction} - must be 1 to buy - must be 0 to sell")
        print(f"sklien_regressor_model_prediction: {sklien_regressor_model_prediction} - must be above 2.5 to buy")
        print(f"momentum_roc_30: {momentum_roc_30} - must be positive to buy - must be negative to sell")
        print("")

        signal_tuple: Optional[Tuple[TradingSignal, str]] = None
        if (
            last_signal == TradingSignal.SELL
            and up_down_model_prediction == 1
            and momentum_roc_30 > 0
            and sklien_regressor_model_prediction > 2.5  # 2.5 is awesome
        ):
            current_price = float(last_features["close"].values[0])
            self.stop_loss = current_price * 0.95  # 0.95 last
            signal_tuple = (TradingSignal.BUY, "Sklien and up down classifier indicate up")
        elif (
            last_signal == TradingSignal.BUY
            and up_down_model_prediction == 0
            and momentum_roc_30 < 0
            # and sklien_regressor_model_prediction < 3  # best 3
        ):
            signal_tuple = (TradingSignal.SELL, "sklien and up down classifier indicate down")

        return signal_tuple
