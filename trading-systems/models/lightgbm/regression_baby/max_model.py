from models.lightgbm.model import LightGBMBaseModel
from features.bukosabino_ta import default_features, macd, roc
import pandas as pd
from dataclasses import dataclass
from targets.regression import max_over_periods
import lightgbm as lgb


@dataclass  # type: ignore
class RegressionBabyMaxModel(LightGBMBaseModel):
    def __post_init__(self) -> None:
        self.model = lgb.LGBMRegressor(
            objective="regression",
            bagging_fraction=0.75,
            bagging_freq=20,
            bagging_seed=5,
            feature_fraction=0.2319,
            feature_fraction_seed=20,
            learning_rate=0.001,
            max_bin=90,
            max_depth=14,
            min_child_weight=1,
            min_data_in_leaf=10,
            min_sum_hessian_in_leaf=5,
            n_estimators=100,
            num_leaves=25,
            subsample=0.9170884223554043,
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
        return max_over_periods.generate_target(
            candlesticks, column="high", periodes=6, percentage=True
        )

    def __hash__(self) -> int:
        return hash(self.__class__.__name__) + hash(self.model)
