from lib.strategy import Strategy
import pandas as pd
from lib.data_splitter import split_features_and_target_into_train_and_test_set
from dataclasses import InitVar
from models.xgboost.model_3.model import XgboostModel3
from models.xgboost.model_2.model import XgboostModel2
from lib.tradingSignal import TradingSignal
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Third(Strategy):

    init_features: InitVar[pd.DataFrame] = None
    xgboost_up_down: XgboostModel2 = None
    xgboost_novice: XgboostModel3 = None

    def __post_init__(self, init_features: Tuple[pd.DataFrame]) -> None:
        self.xgboost_up_down = XgboostModel2()
        self.xgboost_novice = XgboostModel3()
        self.__train(init_features)

    def on_candlestick_with_features(
        self, features: Tuple[pd.DataFrame, ...], trades: pd.DataFrame
    ) -> Optional[TradingSignal]:
        if len(features[0]) % 100 == 0:
            print("Third Strategy - Start retraining.")
            self.__train(features)
            print("Third Strategy - End retraining.")

        prediction1 = self.xgboost_novice.predict(features[0])
        prediction2 = self.xgboost_up_down.predict(features[1])
        return self.on_candlestick_with_features_and_perdictions(
            features, trades, [prediction1, prediction2]
        )

    def on_candlestick_with_features_and_perdictions(
        self, features: Tuple[pd.DataFrame, ...], trades: pd.DataFrame, predictions: List[float]
    ) -> Optional[TradingSignal]:
        last_time, last_signal, last_price = self.get_last_trade(trades)
        if last_signal is None:
            last_signal = TradingSignal.SELL

        novice_predition = predictions[0]
        up_down_prediction = predictions[1]

        signal: Optional[TradingSignal] = None
        if last_signal == TradingSignal.SELL and novice_predition > 1.5 and up_down_prediction == 1:
            current_price = features[0].tail(1)["close"].values[0]
            self.stop_loss = current_price * 0.95
            signal = TradingSignal.BUY
        elif last_signal == TradingSignal.BUY and novice_predition < 0 and up_down_prediction == 0:
            signal = TradingSignal.SELL
        return signal

    @staticmethod
    def generate_features(candlesticks: pd.DataFrame) -> pd.DataFrame:
        """Should return a dataframe containing all features needed by this strategy (for all its models etc)"""
        xgboostBaseModelFeatures3 = XgboostModel3.generate_features(candlesticks)
        xgboostBaseModelFeatures2 = XgboostModel2.generate_features(candlesticks)
        return (xgboostBaseModelFeatures3, xgboostBaseModelFeatures2)

    def __train(self, features: Tuple[pd.DataFrame, ...]):
        target = self._generate_target(features)
        (
            training_set_features2,
            training_set_target2,
            _,
            _,
        ) = split_features_and_target_into_train_and_test_set(features[0], target[0], 0)

        (
            training_set_features3,
            training_set_target3,
            _,
            _,
        ) = split_features_and_target_into_train_and_test_set(features[1], target[1], 0)

        self.xgboost_up_down.train(training_set_features2, training_set_target2)
        self.xgboost_novice.train(training_set_features3, training_set_target3)

    @staticmethod
    def _generate_target(features: Tuple[pd.DataFrame, ...]) -> pd.DataFrame:
        return (
            XgboostModel2.generate_target(features[0]),
            XgboostModel3.generate_target(features[1]),
        )
