from features.bukosabino_ta import default_features, macd, roc
import pandas as pd
from dataclasses import dataclass
from lib.model import Model
from sklearn.svm import LinearSVC
from targets.regression import trend_force
from sklearn.metrics import classification_report


@dataclass  # type: ignore
class ClassifierLinearSVC(Model):
    def __post_init__(self) -> None:
        self.model = LinearSVC(C=5)

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

    @staticmethod
    def generate_target(candlesticks: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        df = trend_force.generate_target(candlesticks)
        df = df.replace([-2, -1, 0, 1, 2], 1)
        df = df.replace([5, 4, 3], 2)
        df = df.replace([-5, -4, -3], 0)

        return df

    def __hash__(self) -> int:
        return hash(self.__class__.__name__) + hash(self.model)

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
        predictions = self.model.predict_dataframe(test_set_features)
        print(classification_report(test_set_target, predictions))

    def print_info(self) -> None:
        print("No info'")