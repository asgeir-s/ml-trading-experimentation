import pandas as pd
import abc
from dataclasses import dataclass
from lib.tradingSignal import TradingSignal
from typing import List, Optional, Dict, Any


@dataclass
class Strategy(abc.ABC):
    #data_training_window: int
    #data_execution_window: int

    stop_loss: Optional[float] = None

    def on_candlestick(self, candlesticks: pd.DataFrame, trades: pd.DataFrame) -> TradingSignal:
        """All or a window of candlesticks up until the newest (.tail(1)) and all earlyer signals."""
        features = self.generate_features(candlesticks)
        return self.on_candlestick_with_features(features, trades)

    @abc.abstractmethod
    def on_candlestick_with_features(
        self, features: pd.DataFrame, trades: pd.DataFrame
    ) -> TradingSignal:
        """Called with precomputed features."""

    @abc.abstractmethod
    def on_candlestick_with_features_and_perdictions(
        self, features: pd.DataFrame, signals: pd.DataFrame, predictions: List[float]
    ) -> TradingSignal:
        """(mostly) Internal method for calling with the features and the predictions. 
        Can be used from outside for testing targets."""

    def on_tick(self, price: float, last_signal: TradingSignal) -> Optional[TradingSignal]:
        if (
            self.stop_loss is not None
            and last_signal == TradingSignal.BUY
            and price <= self.stop_loss
        ):
            return TradingSignal.SELL

    def need_ticks(self, last_signal: TradingSignal):
        """This should be overwritten with a function returning false is a stoploss or trailing stoploss is not used."""
        return last_signal == TradingSignal.BUY and self.stop_loss

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

    @staticmethod
    def get_last_trade(
        trades: pd.DataFrame,
    ) -> (Optional[Any], Optional[TradingSignal], Optional[float]):
        if len(trades) == 0:
            return None, None, None
        else:
            last = trades.tail(1)
            time = last["time"].values[0]
            signal = last["signal"].values[0]
            price = last["price"].values[0]
            return time, signal, price
