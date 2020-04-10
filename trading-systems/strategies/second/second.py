from lib.strategy import Strategy
import pandas as pd
from lib.data_splitter import split_features_and_target_into_train_and_test_set
from dataclasses import InitVar
from models.xgboost.model_2.model_2 import XgboostModel2
from lib.tradingSignal import TradingSignal
from dataclasses import dataclass
import abc
from typing import List, Optional


@dataclass
class Second(Strategy):

    init_features: InitVar[pd.DataFrame] = None
    xgboost_novice: XgboostModel2 = None

    def __post_init__(self, init_features: pd.DataFrame) -> None:
        self.xgboost_novice = XgboostModel2()
        self.__train(init_features)

    def on_candlestick_with_features(
        self, features: pd.DataFrame, trades: pd.DataFrame
    ) -> TradingSignal:
        if len(features) % 100 == 0:
            print("Second Strategy - Start retraining.")
            self.__train(features)
            print("Second Strategy - End retraining.")

        prediction = self.xgboost_novice.predict(features)
        return self.on_candlestick_with_features_and_perdictions(features, trades, [prediction])

    def on_candlestick_with_features_and_perdictions(
        self, features: pd.DataFrame, trades: pd.DataFrame, predictions: List[float]
    ) -> TradingSignal:
        last_time, last_signal, last_price = self.get_last_trade(trades)
        if last_signal is None:
            last_signal = TradingSignal.SELL
        
        prediction = predictions[0]

        if last_signal == TradingSignal.SELL and prediction > 1.5:
            current_price = features.tail(1)["close"].values[0]
            self.stop_loss = current_price * 0.95
            return TradingSignal.BUY
        elif last_signal == TradingSignal.BUY and prediction < 0.5:
            return TradingSignal.SELL

    @staticmethod
    def generate_features(candlesticks: pd.DataFrame) -> pd.DataFrame:
        """Should return a dataframe containing all features needed by this strategy (for all its models etc)"""
        XgboostBaseModelFeatures = XgboostModel2.generate_features(candlesticks)
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
        return XgboostModel2.generate_target(features)
