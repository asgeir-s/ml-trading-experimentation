import pandas as pd
import abc
from dataclasses import dataclass


@dataclass  # type: ignore
class Model(abc.ABC):
    @abc.abstractmethod
    def train(self, features: pd.DataFrame, target: pd.Series) -> None:
        pass

    @abc.abstractmethod
    def predict(self, df: pd.DataFrame) -> float:
        """"
        The data frame should include data up until the datapont to be predicted.
        The returned value should be a number between -1 and 1. Where -1 means sell
        and 1 means buy. The closer to -1 or 1 the number is the more sure is the signal.
        """

    @abc.abstractmethod
    def evaluate(self, testSetFeatures: pd.DataFrame, testSetTarget: pd.Series) -> None:
        """Evaluate"""

    @abc.abstractmethod
    def print_info(self) -> None:
        """Print or plot information about the current model."""

    @staticmethod
    @abc.abstractmethod
    def generate_features(df: pd.DataFrame) -> pd.DataFrame:
        """Given a dataframe with the candlestick data, it generate and return a
        data frame containing the features."""

    @staticmethod
    @abc.abstractmethod
    def generate_target(df: pd.DataFrame) -> pd.Series:
        """Given a dataframe with the correct features, it return a series
        with the same indexes containing the target."""
