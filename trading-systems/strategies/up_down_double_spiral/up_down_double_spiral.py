from lib.strategy import Strategy
import pandas as pd
from models.xgboost import ClassifierUpDownModel
from lib.tradingSignal import TradingSignal
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple


@dataclass
class UpDownDoubleSpiral(Strategy):
    stop_loss_treshold: float = 0.99
    slow_treshold: float = 1.008
    fast_treshold: float = 1

    def __post_init__(self) -> None:
        self.models = (
            ClassifierUpDownModel(treshold=self.fast_treshold),
            ClassifierUpDownModel(treshold=self.slow_treshold),
        )

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

        up_down_model_sell_prediction = predictions[self.models[0]]
        up_down_model_buy_prediction = predictions[self.models[1]]

        signal: Optional[Tuple[TradingSignal, str]] = None
        if (
            last_signal == TradingSignal.SELL
            and up_down_model_buy_prediction == 1
            and up_down_model_sell_prediction == 1
        ):
            current_price = features.tail(1)["close"].values[0]
            self.stop_loss = current_price * self.stop_loss_treshold
            signal = (
                TradingSignal.BUY,
                "The both the buy and sell modell indicates that the market will go up",
            )
        elif (
            last_signal == TradingSignal.BUY
            and up_down_model_sell_prediction == 0
            and up_down_model_buy_prediction == 0
        ):
            signal = (
                TradingSignal.SELL,
                "The both the buy and sell modell indicates that the market will go down",
            )
        return signal
