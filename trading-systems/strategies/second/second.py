from lib.strategy import Strategy
import pandas as pd
from lib.data_splitter import split_features_and_target_into_train_and_test_set
from dataclasses import InitVar
from models.xgboost.model_2.model import XgboostModel2
from lib.tradingSignal import TradingSignal
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Second(Strategy):

    init_features: InitVar[pd.DataFrame] = None
    xgboost_novice: XgboostModel2 = None

    def __post_init__(self, init_features: Tuple[pd.DataFrame]) -> None:
        self.xgboost_novice = XgboostModel2()
        self.__train(init_features)

    def on_candlestick_with_features(
        self, features: pd.DataFrame, trades: pd.DataFrame
    ) -> Optional[TradingSignal]:
        if len(features) % 100 == 0:
            print("Second Strategy - Start retraining.")
            self.__train(features)
            print("Second Strategy - End retraining.")

        prediction = self.xgboost_novice.predict(features[0])
        return self.on_candlestick_with_features_and_perdictions(features, trades, [prediction])

    def on_candlestick_with_features_and_perdictions(
        self, features: Tuple[pd.DataFrame, ...], signals: pd.DataFrame, predictions: List[float],
    ) -> Optional[TradingSignal]:
        last_signal = (
            TradingSignal.SELL if len(signals) == 0 else signals.tail(1)["signal"].values[0]
        )
        prediction = predictions[0]

        signal: Optional[TradingSignal] = None
        if last_signal == TradingSignal.SELL and prediction == 1:
            current_price = features[0].tail(1)["close"].values[0]
            self.stop_loss = current_price * 0.95
            signal = TradingSignal.BUY
        elif last_signal == TradingSignal.BUY and prediction == 0:
            signal = TradingSignal.SELL
        return signal

    @staticmethod
    def generate_features(candlesticks: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
        """Should return a dataframe containing all features needed by this strategy (for all its models etc)"""
        xgboostBaseModelFeatures = XgboostModel2.generate_features(candlesticks)
        return (xgboostBaseModelFeatures,)

    def __train(self, features: Tuple[pd.DataFrame, ...]):
        targets = self._generate_target(features)
        (
            training_set_features,
            training_set_target,
            _,
            _,
        ) = split_features_and_target_into_train_and_test_set(features[0], targets[0], 0)

        self.xgboost_novice.train(training_set_features, training_set_target)

    @staticmethod
    def _generate_target(features: Tuple[pd.DataFrame, ...]) -> Tuple[pd.Series, ...]:
        return (XgboostModel2.generate_target(features[0]),)
