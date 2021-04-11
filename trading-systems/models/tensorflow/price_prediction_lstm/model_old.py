from typing import Optional
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
from targets.regression import ema


@dataclass  # type: ignore
class PricePreditionLSTMModelOld(Model):
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
            (candlesticks[raw_input_cols] / candlesticks[raw_input_cols].shift(1)) - 1 # the cheat
            # (candlesticks[raw_input_cols] / candlesticks[raw_input_cols].shift(-1)) - 1 # the cheat
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
        elif self.target_name == "ema":
            ema_comp = ema.generate_target(candlesticks)
            return (ema_comp.shift(-self.forward_look_for_target).div(ema_comp) - 1) * 100

        relative_target = (
            candlesticks_copy[self.target_col].div(candlesticks_copy["close"], axis=0) - 1
        ) * 100
        relative_target.name = self.target_col
        return relative_target

    def __hash__(self) -> int:
        return hash(self.__class__.__name__) + hash(self.model)

    def create_model(self, number_of_inputs: int, window_size: int):
        inputs = keras.Input(shape=(window_size, number_of_inputs))
        x = keras.layers.Conv1D(64, kernel_size=5, strides=1, padding="causal", activation="relu")(
            inputs
        )
        x = keras.layers.LSTM(units=64, return_sequences=True)(x)
        x = keras.layers.LSTM(units=64)(x)
        # x = keras.layers.Dropout(0.3)(x)
        x = keras.layers.Dense(32, activation="relu")(x)
        # x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(12, activation="relu")(x)
        outputs = keras.layers.Dense(1, activation="elu")(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="price_prediction")

        model.compile(
            loss="mean_absolute_error", optimizer="adam", metrics=["mse", "mae"],
        )
        return model

    def train(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        validation_features: Optional[pd.DataFrame] = None,
        validation_target: Optional[pd.Series] = None,
    ):
        features_copy = features.copy()
        print("training start")
        number_of_inputs = len(features.columns)  # 242
        # print(features.columns)
        # print(features.describe())
        print(target.describe())

        features_copy[self.target_col] = target
        features_copy = features_copy[: -self.forward_look_for_target]

        if self.model is None:
            print("initialize model")
            keras.backend.clear_session()
            tf.random.set_seed(51)
            np.random.seed(51)

            self.model = self.create_model(number_of_inputs, self.window_size)
            w1 = WindowGenerator(
                df=features_copy, input_width=self.window_size, label_width=1, shift=0, label_columns=[self.target_col],
            )
            if validation_features is not None and validation_target is not None:
                val_features_copy = validation_features.copy()
                val_features_copy[self.target_col] = validation_target
                val_features_copy = val_features_copy[: -self.forward_look_for_target]
                w_val = WindowGenerator(
                    df=val_features_copy,
                    input_width=self.window_size,
                    label_width=1,
                    shift=0,
                    label_columns=[self.target_col],
                )
                self.model.fit(w1.dataset, batch_size=64, epochs=6, validation_data=w_val.dataset)
            else:
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
        if self.should_save_model:
            self.save_model()
            print("model save")

    def predict(self, candlesticks: pd.DataFrame, features: pd.DataFrame) -> float:
        # print("predit start")
        needed_features = features.tail(self.window_size)
        needed_features = needed_features.replace([np.inf, -np.inf], np.nan)

        # simulate the bug
        # second_last_row = needed_features.iloc[[-2]]
        # needed_features.loc[needed_features.index[-1], "open"] = second_last_row["open"].values[0]
        # needed_features.loc[needed_features.index[-1], "high"] = second_last_row["high"].values[0]
        # needed_features.loc[needed_features.index[-1], "low"] = second_last_row["low"].values[0]
        # needed_features.loc[needed_features.index[-1], "close"] = second_last_row["close"].values[0]

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

    def save_model(self) -> None:
        """Save the model."""
        self.model.save_weights(self.model_path)

    def load_model(self, number_of_inputs: int) -> None:
        """Load a pre-trained the model."""
        print("loading modell")
        self.model = self.create_model(number_of_inputs, self.window_size)
        self.model.load_weights(self.model_path)

    def print_info(self) -> None:
        print("No info'")
