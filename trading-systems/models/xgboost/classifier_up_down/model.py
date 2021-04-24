from models.xgboost.model import XgboostBaseModel
import xgboost as xgb
from features.bukosabino_ta import default_features, macd, roc
from targets.classes import up_down
import pandas as pd


class ClassifierUpDownModel(XgboostBaseModel):
    target_feature_to_predict: str = "close"
    treshold: float = 1

    def __post_init__(self) -> None:
        self.model = xgb.XGBClassifier(  # type: ignore
            objective="binary:logistic",
            colsample_bytree=0.86,
            learning_rate=0.175,
            max_depth=10,
            min_child_weight=1,
            n_estimators=120,
            subsample=0.5,
            validate_parameters=True,
        )

    def generate_features(
        self, candlesticks: pd.DataFrame, features_already_computed: pd.DataFrame
    ) -> pd.DataFrame:
        features = default_features.compute(
            candlesticks.drop(columns=["open time",]),
            features_already_computed,
        )
        features = macd.compute(candlesticks, features, 100, 30, 20)
        features = macd.compute(candlesticks, features, 300, 100, 50)
        features = macd.compute(candlesticks, features, 15, 5, 3)
        features = macd.compute(candlesticks, features, 10, 4, 2)
        features = macd.compute(candlesticks, features, 7, 3, 2)
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

    def generate_target(
        self, candlesticks: pd.DataFrame, features: pd.DataFrame
    ) -> pd.Series:
        return up_down.generate_target(
            df=features, column=self.target_feature_to_predict, treshold=self.treshold,
        )

    def __hash__(self) -> int:
        return hash(self.__class__.__name__) + hash(self.model)
