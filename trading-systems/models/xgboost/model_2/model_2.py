from models.xgboost.model import XgboostBaseModel
import xgboost as xgb
from features.bukosabino_ta import default_features
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class XgboostModel2(XgboostBaseModel):

    model: xgb.XGBRegressor = xgb.XGBRegressor(
        objective="multi:softmax",
        colsample_bytree=0.3,
        learning_rate=1,
        max_depth=12,
        alpha=5,
        n_estimators=10,
        num_class=3,
    )

    @staticmethod
    def generate_features(df: pd.DataFrame):
        return default_features.createFeatures(df.drop(columns=["open time", "close time"]))

    @staticmethod
    def generate_target(df: pd.DataFrame):
        up_treshold = 1.02
        down_treshold = 1
        conditions = [
            (df.shift(periods=-2)["open"] / df.shift(periods=-1)["open"] > up_treshold),
            (df.shift(periods=-2)["open"] / df.shift(periods=1)["open"] < down_treshold),
        ]
        choices = [2, 0]
        return pd.Series(np.select(conditions, choices, default=1))