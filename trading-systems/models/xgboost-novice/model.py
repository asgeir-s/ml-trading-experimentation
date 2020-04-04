import xgboost as xgb
import pandas as pd
import numpy as np
from dataclasses import dataclass
from lib.model import Model
from features.bukosabino_ta import default_features
from sklearn.metrics import mean_squared_error


@dataclass
class XgboostNovice(Model):

    model: xgb.XGBRegressor = xgb.XGBRegressor(
            objective="multi:softmax",
            colsample_bytree=0.3,
            learning_rate=1,
            max_depth=12,
            alpha=5,
            n_estimators=10,
            num_class=3,
        )


    def train(self, features: pd.DataFrame, target: pd.Series):
        self.model.fit(features, target)

    def predict(self, df: pd.DataFrame):
        return 1

    def evaluate(self, testSetFeatures: pd.DataFrame, testSetTarget: pd.Series):
        preds = self.model.predict(testSetFeatures)

        rmse = np.sqrt(mean_squared_error(testSetTarget, preds))
        print("RMSE: %f" % (rmse))

    @staticmethod
    def generateFeatures(df: pd.DataFrame):
        return default_features.createFeatures(df)

    @staticmethod
    def generateTarget(df: pd.DataFrame):
        upTreshold = 1.003
        downTreshold = 0.998
        conditions = [
            (
                (1 / df["trend_sma_fast"])
                * (
                    (
                        df.shift(periods=-1)["trend_sma_fast"]
                        + df.shift(periods=-2)["trend_sma_fast"]
                    )
                    / 2
                )
                > upTreshold
            ),
            (
                (1 / df["trend_sma_fast"])
                * (
                    (
                        df.shift(periods=-1)["trend_sma_fast"]
                        + df.shift(periods=-2)["trend_sma_fast"]
                    )
                    / 2
                )
                < downTreshold
            ),
        ]
        choices = [2, 0]
        return pd.Series(np.select(conditions, choices, default=1))
