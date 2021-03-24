from features.bukosabino_ta import default_features
import pandas as pd
from dataclasses import dataclass
from lib.model import Model
from sklearn.metrics import classification_report
from tensorflow import keras
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from lib.window_generator import WindowGenerator
from pandas.core.frame import DataFrame


@dataclass  # type: ignore
class PricePreditionLSTMModel(Model):
    target_name: str = "close"
    forward_look_for_target: int = 6
    window_size = 100
    target_col = "not set"
    scaler = None

    def __post_init__(self) -> None:
        self.target_col = f"target-{self.target_name}-next-{self.forward_look_for_target}"

    # @staticmethod
    def generate_features(
        self,
        candlesticks: pd.DataFrame,
        features_already_computed: pd.DataFrame,
        reset_scaler: bool = False,
    ) -> pd.DataFrame:

        raw_input_cols = [
            "open",
            "high",
            "low",
            "close",
            # "volume",
            # "quote asset volume",
            # "number of trades",
            # "taker buy base asset volume",
            # "taker buy quote asset volume",
        ]

        relative_input = (
            (candlesticks[raw_input_cols] / candlesticks[raw_input_cols].shift(-1)) - 1
        ) * 100
        relative_input = relative_input.replace([np.inf, -np.inf], np.nan)
        relative_input = relative_input.fillna(0)
        relative_input = relative_input.clip(-5, 5)

        computed_features = default_features.compute(
            candlesticks[raw_input_cols + ["volume"]], features_already_computed
        )

        if self.scaler is None or reset_scaler:
            print("resetting scalar")
            self.scaler = StandardScaler().fit(computed_features)

        computed_scaled = DataFrame(
            self.scaler.transform(computed_features), columns=[computed_features.columns]
        )
        features = pd.concat([relative_input, computed_scaled], axis=1, join="inner")

        return features

    # @staticmethod
    def generate_target(self, candlesticks: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        candlesticks_copy = candlesticks.copy()
        if self.target_name == "high":
            candlesticks_copy[self.target_col] = (
                candlesticks_copy["high"]
                .rolling(self.forward_look_for_target)
                .max()
                .shift(-self.forward_look_for_target)
            )
        elif self.target_name == "low":
            candlesticks_copy[self.target_col] = (
                candlesticks_copy["low"]
                .rolling(self.forward_look_for_target)
                .min()
                .shift(-self.forward_look_for_target)
            )
        elif self.target_name == "close":
            candlesticks_copy[self.target_col] = candlesticks_copy["close"].shift(
                -self.forward_look_for_target
            )

        relative_target = (
            candlesticks_copy[self.target_col].div(candlesticks_copy["close"], axis=0) - 1
        ) * 100
        relative_target.name = self.target_col
        return relative_target

    def __hash__(self) -> int:
        return hash(self.__class__.__name__) + hash(self.model)

    def train(self, features: pd.DataFrame, target: pd.Series):
        features_copy = features.copy()
        print("training start")
        number_of_inputs = len(features.columns)  # 242
        # print(features.columns)
        # print(features.describe())
        print(target.describe())

        features_copy[self.target_col] = target
        features_copy = features_copy[: -self.forward_look_for_target]

        if self.model is None:
            w1 = WindowGenerator(
                df=features_copy,
                input_width=self.window_size,
                label_width=1,
                shift=0,
                label_columns=[self.target_col],
            )
            print("initialize model")
            keras.backend.clear_session()
            tf.random.set_seed(51)
            np.random.seed(51)

            inputs = keras.Input(shape=(self.window_size, number_of_inputs))
            x = keras.layers.Conv1D(
                64, kernel_size=5, strides=1, padding="causal", activation="relu"
            )(inputs)
            x = keras.layers.LSTM(units=64, return_sequences=True)(x)
            x = keras.layers.LSTM(units=64)(x)
            x = keras.layers.Dense(32, activation="relu")(x)
            x = keras.layers.Dense(12, activation="relu")(x)
            outputs = keras.layers.Dense(1, activation="elu")(x)

            self.model = keras.Model(inputs=inputs, outputs=outputs, name="close_price_prediction")

            self.model.compile(
                loss="mean_absolute_error", optimizer="adam", metrics=["mse", "mae"],
            )
            self.model.fit(w1.dataset, batch_size=64, epochs=6)

        w2 = WindowGenerator(
            df=features_copy.tail(8640),
            input_width=self.window_size,
            label_width=1,
            shift=0,
            label_columns=[self.target_col],
        )

        self.model.fit(w2.dataset, batch_size=64, epochs=1)
        print("training end")

    def predict(self, candlesticks: pd.DataFrame, features: pd.DataFrame) -> float:
        # print("predit start")
        needed_features = features.tail(self.window_size)
        needed_features = needed_features.replace([np.inf, -np.inf], np.nan)
        if needed_features.isnull().values.any():
            count_nan_in_df = needed_features.isnull().sum()
            print("predictins nulls")
            print(count_nan_in_df)

        w1 = WindowGenerator(
            df=needed_features, input_width=self.window_size, label_width=1, shift=0,
        )
        prediction = self.model.predict(w1.features)
        # print("number of predictions:", len(prediction))
        last_predition = prediction[len(prediction) - 1][len(prediction[0]) - 1]
        # print(last_predition)
        # print("prediction end")
        return last_predition

    def predict_dataframe(self, df: pd.DataFrame):
        print(
            """Warning: using predict_dataframe (only meant for use in evaluation). This will predict all rows in the
            dataframe."""
        )
        needed_features = df
        needed_features = needed_features.replace([np.inf, -np.inf], np.nan)
        if needed_features.isnull().values.any():
            count_nan_in_df = needed_features.isnull().sum()
            print("predictins nulls")
            print(count_nan_in_df)

        w1 = WindowGenerator(
            df=needed_features, input_width=self.window_size, label_width=1, shift=0,
        )
        prediction = self.model.predict(w1.features)
        return prediction

    def evaluate(self, test_set_features: pd.DataFrame, test_set_target: pd.Series):
        test_set_features_copy = test_set_features.copy()
        test_set_features_copy[self.target_col] = test_set_target
        w1 = WindowGenerator(
            df=test_set_features_copy,
            input_width=self.window_size,
            label_width=1,
            shift=0,
            label_columns=[self.target_col],
        )
        # Evaluate the model on the test data using `evaluate`
        print("Evaluate on test data")
        results = self.model.evaluate(w1.dataset, batch_size=64)
        print("test loss, test acc:", results)

        # Generate predictions (probabilities -- the output of the last layer)
        # on new data using `predict`
        print("Generate predictions for 3 samples")
        for inputs in w1.features.take(3):
            predictions = self.model.predict(inputs)
            print("predictions shape:", predictions.shape)
            print("predition[0][0]:", predictions[0][0])

    def print_info(self) -> None:
        print("No info'")
