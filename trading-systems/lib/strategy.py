import pandas as pd
import abc
from dataclasses import dataclass
from lib.tradingSignal import TradingSignal
from typing import List, Optional


@dataclass
class Strategy(abc.ABC):
    def on_candlestick(self, candlesticks: pd.DataFrame, signals: pd.DataFrame) -> TradingSignal:
        """All or a window of candlesticks up until the newest (.tail(1)) and all earlyer signals."""
        features = self.generate_features(candlesticks)
        return self.on_candlestick_with_features(features, signals)

    @abc.abstractmethod
    def on_candlestick_with_features(self, features: pd.DataFrame, signals: pd.DataFrame) -> TradingSignal:
        """Called with precomputed features."""
    
    @abc.abstractmethod
    def on_candlestick_with_features_and_perdictions(self, features: pd.DataFrame, signals: pd.DataFrame, predictions: List[float]) -> TradingSignal:
        """(mostly) Internal method for calling with the features and the predictions. 
        Can be used from outside for testing targets."""

    def on_shorter_candlestick(self, last_candlestick: pd.Series, signals: pd.DataFrame) -> Optional[TradingSignal]:
        """Can act upon a shorter time frame."""
        pass

    @staticmethod
    @abc.abstractmethod
    def generate_features(candlesticks: pd.DataFrame) -> pd.DataFrame:
        """Should return a dataframe containing all features needed by this strategy (for all its models etc.)"""

    def __train(self, features: pd.DataFrame):
        """Train all models etc. that needs training."""

    @staticmethod
    @abc.abstractmethod
    def _generate_target(features: pd.DataFrame) -> pd.DataFrame:
        """Internal method used genrate features for models used by the strategy. Can be used fro outside for testing target."""