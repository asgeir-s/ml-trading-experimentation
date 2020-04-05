import pandas as pd
import abc
from dataclasses import dataclass
from lib.tradingSignal import TradingSignal


@dataclass
class Strategy(abc.ABC):
    @abc.abstractmethod
    def execute(self, candlesticks: pd.DataFrame, trades: pd.DataFrame) -> TradingSignal:
        """All candlesticks up until the newest (.tail(1))."""
        pass

    def execute_with_features(self, features: pd.DataFrame, trades: pd.DataFrame) -> TradingSignal:
        """Call execute with precomputed features."""

    @staticmethod
    def generate_features(candlesticks: pd.DataFrame) -> pd.DataFrame:
        """Should return a dataframe containing all features needed by this strategy (for all its models etc)"""

    def __train(self, features: pd.DataFrame):
        """Train all models etc. that needs training."""
