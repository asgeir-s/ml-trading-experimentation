import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from lib.model import Model
from features.bukosabino_ta import default_features
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection._validation import cross_val_score


@dataclass
class XgboostBaseModel(Model):

    model = xgb.XGBRegressor(  # type: ignore
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
        prediction = self.model.predict(df.tail(1))[0]
        return prediction

    def predict_dataframe(self, df: pd.DataFrame):
        print(
            """Warning: using predict_dataframe (only meant for use in evaluation). This will predict all rows in the
            dataframe."""
            )
        prediction = self.model.predict(df)
        return prediction

    def evaluate(self, test_set_features: pd.DataFrame, test_set_target: pd.Series):
        predictions = self.model.predict(test_set_features)

        rmse = np.sqrt(mean_squared_error(test_set_target, predictions))
        print("RMSE: %f" % (rmse))

        # evaluate predictions
        # accuracy = accuracy_score(test_set_target, predictions)
        # print("Accuracy: %.2f%%" % (accuracy * 100.0))

        # retrieve performance metrics
        kfold = StratifiedKFold(n_splits=10)
        results = cross_val_score(self.model, test_set_features, test_set_target, cv=kfold)
        print("kfold Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

    def print_info(self) -> None:
        xgb.plot_importance(self.model)  # type: ignore
        plt.rcParams["figure.figsize"] = [15, 30]
        plt.show()

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
