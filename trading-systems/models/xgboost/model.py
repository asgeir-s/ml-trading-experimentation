import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from lib.model import Model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection._validation import cross_val_score
from features.bukosabino_ta import roc, macd, default_features


@dataclass
class XgboostBaseModel(Model):

    def __post_init__(self) -> None:
        self.model = xgb.XGBRegressor(  # type: ignore
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

    def predict(self, candlesticks: pd.DataFrame, features: pd.DataFrame) -> float:
        prediction = self.model.predict(features.tail(1))[0]
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
        up_treshold = 1.02
        down_treshold = 1
        conditions = [
            (
                candlesticks.shift(periods=-2)["open"] / candlesticks.shift(periods=-1)["open"]
                > up_treshold
            ),
            (
                candlesticks.shift(periods=-2)["open"] / candlesticks.shift(periods=1)["open"]
                < down_treshold
            ),
        ]
        choices = [2, 0]
        return pd.Series(np.select(conditions, choices, default=1))

    def __hash__(self) -> int:
        return hash(self.__class__.__name__) + hash(self.model)
