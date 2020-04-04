from lib.strategy import Strategy
import pandas as pd
from lib.data_splitter import split_features_and_target_into_train_and_test_set
from dataclasses import InitVar
from models.xgboost.model import XgboostNovice
from lib.tradingSignal import TradingSignal


@dataclass
class First(Strategy):

    xgboost_novice: XgboostNovice = None
    init_candlesticks: InitVar[pd.DataFrame]

    def __post_init__(self, init_candlesticks: pd.DataFrame) -> None:
        features = XgboostNovice.generate_features(init_candlesticks)
        target = XgboostNovice.generate_target(features)
        (
            training_set_features,
            training_set_target,
            _,
            _,
        ) = split_features_and_target_into_train_and_test_set(
            features, target, 0
        )

        xgboost_novice = XgboostNovice()
        xgboost_novice.train(training_set_features, training_set_target)

    @abc.abstractmethod
    def execute(self, candlesticks: pd.DataFrame) -> TradingSignal:
        features = XgboostNovice.generate_features(candlesticks)
        prediction = self.xgboost_novice.predict(features)
        if prediction > 0.5:
            return TradingSignal.BUY
        elif prediction < -0.5:
            return TradingSignal.SELL
        else:
            return TradingSignal.HOLD
