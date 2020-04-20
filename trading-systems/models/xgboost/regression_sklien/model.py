from models.xgboost.model import XgboostBaseModel
from features.bukosabino_ta import default_features, macd, roc
import pandas as pd
from dataclasses import dataclass
from targets.regression import trend_force


@dataclass  # type: ignore
class RegressionSklienModel(XgboostBaseModel):
    def __post_init__(self) -> None:
        self.model = xgb.XGBRegressor(  # type: ignore
            objective="reg:squarederror",
            max_depth=15,
            colsample_bytree=0.3,
            learning_rate=0.1,
            alpha=10,
            n_estimators=200,
        )

    @staticmethod
    def generate_features(candlesticks: pd.DataFrame, features_already_computed: pd.DataFrame):
        features = default_features.compute(
            candlesticks.drop(columns=["open time", "close time"]), features_already_computed
        )
        features = macd.compute(candlesticks, features, 100, 30, 20)
        features = macd.compute(candlesticks, features, 300, 100, 50)
        features = macd.compute(candlesticks, features, 15, 5, 3)
        features = roc.compute(candlesticks, features, 2)
        features = roc.compute(candlesticks, features, 3)
        features = roc.compute(candlesticks, features, 3)
        features = roc.compute(candlesticks, features, 5)
        features = roc.compute(candlesticks, features, 10)
        features = roc.compute(candlesticks, features, 15)
        features = roc.compute(candlesticks, features, 20)
        features = roc.compute(candlesticks, features, 30)
        features = roc.compute(candlesticks, features, 50)
        features = roc.compute(candlesticks, features, 80)

        return features

    @staticmethod
    def generate_target(candlesticks: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        return trend_force.generate_target(candlesticks)

    def __hash__(self) -> int:
        return hash(self.__class__.__name__) + hash(self.model)
