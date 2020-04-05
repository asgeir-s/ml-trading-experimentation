import pandas as pd
import abc
from dataclasses import dataclass
from lib.tradingSignal import TradingSignal
from typing import List


@dataclass
class Strategy(abc.ABC):
    @abc.abstractmethod
    def execute(self, candlesticks: pd.DataFrame, trades: pd.DataFrame) -> TradingSignal:
        """All candlesticks up until the newest (.tail(1))."""
        pass

    def execute_with_features(self, features: pd.DataFrame, trades: pd.DataFrame) -> TradingSignal:
        """Call execute with precomputed features."""
    
    def _execute(self, features: pd.DataFrame, trades: pd.DataFrame, predictions: List[float]) -> TradingSignal:
        """Internal method for calling execute with the prediction. Can be used from outside for testing targets."""

    @staticmethod
    def generate_features(candlesticks: pd.DataFrame) -> pd.DataFrame:
        """Should return a dataframe containing all features needed by this strategy (for all its models etc)"""

    def __train(self, features: pd.DataFrame):
        """Train all models etc. that needs training."""

    @staticmethod
    def _generate_target(features: pd.DataFrame) -> pd.DataFrame:
        """Internal method used genrate features for models used by the strategy. Can be used fro outside for testing target."""