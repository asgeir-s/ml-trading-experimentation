import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from lib.model import Model
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection._validation import cross_val_score
import lightgbm as lgb

@dataclass
class LightGBMBaseModel(Model):
    def __post_init__(self) -> None:
        self.model = lgb.LGBMRegressor(
            objective="regression",
            num_leaves=5,
            learning_rate=0.05,
            n_estimators=720,
            max_bin=55,
            bagging_fraction=0.8,
            bagging_freq=5,
            feature_fraction=0.2319,
            feature_fraction_seed=9,
            bagging_seed=9,
            min_data_in_leaf=6,
            min_sum_hessian_in_leaf=11,
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
        df = pd.DataFrame(prediction, index=df.index)
        return df

    def evaluate(self, test_set_features: pd.DataFrame, test_set_target: pd.Series):
        predictions = self.model.predict(test_set_features)

        rmse = np.sqrt(mean_squared_error(test_set_target, predictions))
        print("RMSE: %f" % (rmse))

    def print_info(self) -> None:
        lgb.plot_importance(self.model, top_n = 10, measure = "Gain")
        plt.rcParams["figure.figsize"] = [15, 30]
        plt.show()

    def save_model(self) -> None:
        """Save the model."""
        print("WARNING: saving models not implimented")

    def load_model(self, number_of_inputs: int) -> None:
        """Load a pre-trained the model."""
        print("WARNING: loading model not implimented")

    def __hash__(self) -> int:
        return hash(self.__class__.__name__) + hash(self.model)
