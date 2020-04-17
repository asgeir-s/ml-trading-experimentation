import pandas as pd
import abc
from dataclasses import dataclass
from lib.tradingSignal import TradingSignal
from typing import List, Optional, Any, Tuple
from dataclasses import InitVar


@dataclass  # type: ignore
class Strategy(abc.ABC):
    stop_loss: Optional[float] = None
    init_features: InitVar[Tuple[pd.DataFrame, ...]] = None

    def on_candlestick(
        self, candlesticks: pd.DataFrame, trades: pd.DataFrame
    ) -> Optional[TradingSignal]:
        """All or a window of candlesticks up until the newest (.tail(1)) and all earlyer signals."""
        features = self.generate_features(candlesticks)
        return self.on_candlestick_with_features(features, trades)

    @abc.abstractmethod
    def on_candlestick_with_features(
        self, features: Tuple[pd.DataFrame, ...], trades: pd.DataFrame
    ) -> Optional[TradingSignal]:
        """Called with precomputed features."""

    @abc.abstractmethod
    def on_candlestick_with_features_and_perdictions(
        self, features: Tuple[pd.DataFrame, ...], signals: pd.DataFrame, predictions: List[float],
    ) -> Optional[TradingSignal]:
        """
        (mostly) Internal method for calling with the features and the predictions.
        Can be used from outside for testing targets.
        """

    def on_tick(self, price: float, last_signal: TradingSignal) -> Optional[TradingSignal]:
        """
        Checks if the stoploss should be executed. Should be called on every tick in live mode.
        Be carful overriding this as it will not be possible to backtest when it is changed.
        """
        if (
            self.stop_loss is not None
            and last_signal == TradingSignal.BUY
            and price <= self.stop_loss
        ):
            return TradingSignal.SELL
        else:
            return None

    def need_ticks(self, last_signal: TradingSignal) -> bool:
        """
        This should be overwritten with a function returning false is a stoploss or trailing stoploss is not used.
        This is used for optimizing the backtest.
        """
        return last_signal == TradingSignal.BUY and self.stop_loss is not None

    @staticmethod
    @abc.abstractmethod
    def generate_features(candlesticks: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
        """
        Should return a dataframe containing all features needed by this strategy (for all its models etc.).
        """

    def __train(self, features: Tuple[pd.DataFrame, ...]):
        """
        Train all models etc. that needs training.
        """

    @staticmethod
    @abc.abstractmethod
    def _generate_target(features: Tuple[pd.DataFrame, ...]) -> Tuple[pd.Series, ...]:
        """
        Internal method used genrate features for models used by the strategy. Can be used fro outside for backtesting
        (raw) target (CHEATING).
        """

    @staticmethod
    def get_last_trade(
        trades: pd.DataFrame,
    ) -> Tuple[Optional[Any], Optional[TradingSignal], Optional[float]]:
        if len(trades) == 0:
            return None, None, None
        else:
            last = trades.tail(1)
            time = last["transactTime"].values[0]
            signal = last["signal"].values[0]
            price = last["price"].values[0]
            return time, signal, price
