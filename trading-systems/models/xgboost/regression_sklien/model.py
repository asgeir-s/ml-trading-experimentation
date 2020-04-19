from models.xgboost.model import XgboostBaseModel
import xgboost as xgb
from features.bukosabino_ta import default_features, macd, roc
import pandas as pd
from dataclasses import dataclass


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
        res = pd.DataFrame(index=candlesticks.index)
        res["1-over-last"] = candlesticks["close"] < candlesticks["close"].shift(periods=-1)
        res["2-over-last"] = candlesticks["close"].shift(periods=-1) < candlesticks["close"].shift(
            periods=-2
        )
        res["3-over-last"] = candlesticks["close"].shift(periods=-2) < candlesticks["close"].shift(
            periods=-3
        )
        res["4-over-last"] = candlesticks["close"].shift(periods=-3) < candlesticks["close"].shift(
            periods=-4
        )
        res["5-over-last"] = candlesticks["close"].shift(periods=-4) < candlesticks["close"].shift(
            periods=-5
        )

        res["1-over-first"] = candlesticks["close"] < candlesticks["close"].shift(periods=-1)
        res["2-over-first"] = candlesticks["close"] < candlesticks["close"].shift(periods=-2)
        res["3-over-first"] = candlesticks["close"] < candlesticks["close"].shift(periods=-3)
        res["4-over-first"] = candlesticks["close"] < candlesticks["close"].shift(periods=-4)
        res["5-over-first"] = candlesticks["close"] < candlesticks["close"].shift(periods=-5)

        res["1-below-last"] = candlesticks["close"] > candlesticks["close"].shift(periods=-1)
        res["2-below-last"] = candlesticks["close"].shift(periods=-1) > candlesticks["close"].shift(
            periods=-2
        )
        res["3-below-last"] = candlesticks["close"].shift(periods=-2) > candlesticks["close"].shift(
            periods=-3
        )
        res["4-below-last"] = candlesticks["close"].shift(periods=-3) > candlesticks["close"].shift(
            periods=-4
        )
        res["5-below-last"] = candlesticks["close"].shift(periods=-4) > candlesticks["close"].shift(
            periods=-5
        )

        res["one"] = 1
        res["minus-one"] = -1

        res["1-add"] = res["one"][(res["1-over-last"]) & (res["1-over-first"])]
        res["2-add"] = res["one"][(res["2-over-last"]) & (res["2-over-first"])]
        res["3-add"] = res["one"][(res["3-over-last"]) & (res["3-over-first"])]
        res["4-add"] = res["one"][(res["4-over-last"]) & (res["4-over-first"])]
        res["5-add"] = res["one"][(res["5-over-last"]) & (res["5-over-first"])]

        res["1-sub"] = res["minus-one"][(res["1-below-last"]) & (~res["1-over-first"])]
        res["2-sub"] = res["minus-one"][(res["2-below-last"]) & (~res["2-over-first"])]
        res["3-sub"] = res["minus-one"][(res["3-below-last"]) & (~res["3-over-first"])]
        res["4-sub"] = res["minus-one"][(res["4-below-last"]) & (~res["4-over-first"])]
        res["5-sub"] = res["minus-one"][(res["5-below-last"]) & (~res["5-over-first"])]

        res["target"] = res[
            [
                "1-add",
                "2-add",
                "3-add",
                "4-add",
                "5-add",
                "1-sub",
                "2-sub",
                "3-sub",
                "4-sub",
                "5-sub",
            ]
        ].sum(axis=1)

        return res["target"]

    def __hash__(self) -> int:
        return hash(self.__class__.__name__) + hash(self.model)
