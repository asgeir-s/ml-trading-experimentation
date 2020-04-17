from models.xgboost.model import XgboostBaseModel
import xgboost as xgb
from features.bukosabino_ta import default_features, macd, roc
import pandas as pd
from dataclasses import dataclass


@dataclass  # type: ignore
class XgboostModel3(XgboostBaseModel):

    model = xgb.XGBRegressor(  # type: ignore
        objective="reg:squarederror",
        max_depth=15,
        colsample_bytree=0.3,
        learning_rate=0.1,
        alpha=10,
        n_estimators=200,
    )

    @staticmethod
    def generate_features(df: pd.DataFrame):
        default_tas = default_features.createFeatures(df.drop(columns=["open time", "close time"]))
        return pd.concat(
            [
                default_tas,
                macd.compute(df, 100, 30, 20),
                macd.compute(df, 15, 5, 3),
                macd.compute(df, 300, 100, 50),
                roc.compute(df, 2),
                roc.compute(df, 3),
                roc.compute(df, 4),
                roc.compute(df, 5),
                roc.compute(df, 10),
                roc.compute(df, 15),
                roc.compute(df, 20),
                roc.compute(df, 30),
                roc.compute(df, 50),
                roc.compute(df, 80),
            ],
            axis=1,
            sort=False,
        )

    @staticmethod
    def generate_target(df: pd.DataFrame):
        res = pd.DataFrame(index=df.index)
        res["1-over-last"] = df["close"] < df["close"].shift(periods=-1)
        res["2-over-last"] = df["close"].shift(periods=-1) < df["close"].shift(periods=-2)
        res["3-over-last"] = df["close"].shift(periods=-2) < df["close"].shift(periods=-3)
        res["4-over-last"] = df["close"].shift(periods=-3) < df["close"].shift(periods=-4)
        res["5-over-last"] = df["close"].shift(periods=-4) < df["close"].shift(periods=-5)

        res["1-over-first"] = df["close"] < df["close"].shift(periods=-1)
        res["2-over-first"] = df["close"] < df["close"].shift(periods=-2)
        res["3-over-first"] = df["close"] < df["close"].shift(periods=-3)
        res["4-over-first"] = df["close"] < df["close"].shift(periods=-4)
        res["5-over-first"] = df["close"] < df["close"].shift(periods=-5)

        res["1-below-last"] = df["close"] > df["close"].shift(periods=-1)
        res["2-below-last"] = df["close"].shift(periods=-1) > df["close"].shift(periods=-2)
        res["3-below-last"] = df["close"].shift(periods=-2) > df["close"].shift(periods=-3)
        res["4-below-last"] = df["close"].shift(periods=-3) > df["close"].shift(periods=-4)
        res["5-below-last"] = df["close"].shift(periods=-4) > df["close"].shift(periods=-5)

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
