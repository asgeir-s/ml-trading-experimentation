from models.xgboost.model import XgboostBaseModel
import xgboost as xgb
from features.bukosabino_ta import default_features, macd, roc
from targets.classes import up_down
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class ClassifierUpDownModel(XgboostBaseModel):
    def __post_init__(self) -> None:
        self.model = xgb.XGBClassifier(  # type: ignore
            objective="binary:logistic",
            colsample_bytree=0.8613434432877689,
            learning_rate=0.1747803828120286,
            max_depth=8,
            min_child_weight=1,
            n_estimators=492,
            subsample=0.5361675952749418,
        )

    @staticmethod
    def generate_features(
        candlesticks: pd.DataFrame, features_already_computed: pd.DataFrame
    ) -> pd.DataFrame:
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
        return up_down.generate_target(
            df=features, column="tend_sma_fast", up_treshold=1, down_treshold=1
        )

    def __hash__(self) -> int:
        return hash(self.__class__.__name__) + hash(self.model)
