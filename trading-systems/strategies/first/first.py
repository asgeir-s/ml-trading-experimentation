from lib.strategy import Strategy
import pandas as pd
from lib.data_splitter import split_features_and_target_into_train_and_test_set
from dataclasses import InitVar
from models.xgboost.model import XgboostBaseModel
from lib.tradingSignal import TradingSignal
from dataclasses import dataclass
import abc
from typing import List, Optional


@dataclass
class First(Strategy):

    init_candlesticks: InitVar[pd.DataFrame]
    xgboost_novice: XgboostBaseModel = None

    def __post_init__(self, init_features: pd.DataFrame) -> None:
        self.xgboost_novice = XgboostBaseModel()
        self.__train(init_features)

    def on_candlestick_with_features(
        self, features: pd.DataFrame, signals: pd.DataFrame
    ) -> TradingSignal:
        if len(features) % 100 == 0:
            print("First Strategy - Start retraining.")
            self.__train(features)
            print("First Strategy - End retraining.")

        prediction = self.xgboost_novice.predict(features)
        return self.on_candlestick_with_features_and_perdictions(features, signals, [prediction])

    def on_candlestick_with_features_and_perdictions(
        self, features: pd.DataFrame, signals: pd.DataFrame, predictions: List[float]
    ) -> TradingSignal:
        last_signal = (
            TradingSignal.SELL if len(signals) == 0 else signals.tail(1)["signal"].values[0]
        )
        prediction = predictions[0]

        if last_signal == TradingSignal.SELL and prediction > 1.5:
            return TradingSignal.BUY
        elif last_signal == TradingSignal.BUY and prediction < 0.5:
            return TradingSignal.SELL


    @staticmethod
    def generate_features(candlesticks: pd.DataFrame) -> pd.DataFrame:
        """Should return a dataframe containing all features needed by this strategy (for all its models etc)"""
        XgboostBaseModelFeatures = XgboostBaseModel.generate_features(candlesticks)
        return XgboostBaseModelFeatures

    def __train(self, features: pd.DataFrame):
        target = self._generate_target(features)
        (
            training_set_features,
            training_set_target,
            _,
            _,
        ) = split_features_and_target_into_train_and_test_set(features, target, 0)

        self.xgboost_novice.train(training_set_features, training_set_target)

    @staticmethod
    def _generate_target(features: pd.DataFrame) -> pd.DataFrame:
        return XgboostBaseModel.generate_target(features)