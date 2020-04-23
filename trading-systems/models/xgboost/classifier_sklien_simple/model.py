from models.xgboost.model import XgboostBaseModel
from features.bukosabino_ta import default_features, macd, roc
import pandas as pd
import xgboost as xgb
from dataclasses import dataclass
from targets.regression import trend_force


@dataclass  # type: ignore
class ClassifierSklienSimpleModel(XgboostBaseModel):
    def __post_init__(self) -> None:
        self.model = xgb.XGBClassifier(  # type: ignore
            objective="multi:softmax",
            colsample_bytree=0.8613434432877689,
            learning_rate=0.1747803828120286,
            max_depth=8,
            min_child_weight=1,
            n_estimators=492,
            subsample=0.5361675952749418,
            num_class=3,
            eval_metric="merror",
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
        df = trend_force.generate_target(candlesticks)
        df = df.replace([-2, -1, 0, 1, 2], 1)
        df = df.replace([5, 4, 3], 2)
        df = df.replace([-5, -4, -3], 0)

        return df

    def __hash__(self) -> int:
        return hash(self.__class__.__name__) + hash(self.model)
