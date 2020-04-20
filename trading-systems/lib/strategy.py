import pandas as pd
import abc
from dataclasses import dataclass
from lib.tradingSignal import TradingSignal
from typing import Optional, Any, Tuple, Dict
from lib.model import Model
from lib.data_splitter import split_features_and_target_into_train_and_test_set


@dataclass  # type: ignore
class Strategy(abc.ABC):
    stop_loss: Optional[float] = None
    models: Tuple[Model, ...] = ()

    @abc.abstractmethod
    def __post_init__(self) -> None:
        """
        The models should be initiated here.
        """

    def init(self, candlesticks: pd.DataFrame, features: pd.DataFrame) -> None:
        """
        Should be called when sitting up a strategy.
        """
        self.__train(candlesticks, features)

    def on_candlestick(
        self, candlesticks: pd.DataFrame, trades: pd.DataFrame
    ) -> Optional[Tuple[TradingSignal, str]]:
        """All or a window of candlesticks up until the newest (.tail(1)) and all earlyer signals."""
        features = self.generate_features(candlesticks)
        return self.on_candlestick_with_features(candlesticks, features, trades)

    def on_candlestick_with_features(
        self, candlesticks: pd.DataFrame, features: pd.DataFrame, trades: pd.DataFrame
    ) -> Optional[Tuple[TradingSignal, str]]:
        """
        It calls the __train method every 100th execution.

        It also calls predict on every model and calls on_candlestick_with_features_and_perdictions with the
        predictions.
        """
        if len(features) % 100 == 0:
            print("Strategy - Start retraining.")
            self.__train(candlesticks, features)
            print("Strategy - End retraining.")

        predictions = {}
        for model in self.models:
            predictions[model] = model.predict(candlesticks, features)

        return self.on_candlestick_with_features_and_perdictions(
            candlesticks, features, trades, predictions
        )

    @abc.abstractmethod
    def on_candlestick_with_features_and_perdictions(
        self,
        candlesticks: pd.DataFrame,
        features: pd.DataFrame,
        trades: pd.DataFrame,
        predictions: Dict[Any, float],
    ) -> Optional[Tuple[TradingSignal, str]]:
        """
        (mostly) Internal method for calling with the features and the predictions.
        Can be used from outside for testing targets.
        """

    def on_tick(self, price: float, last_signal: TradingSignal) -> Optional[Tuple[TradingSignal, str]]:
        """
        Checks if the stoploss should be executed. Should be called on every tick in live mode.
        Be carful overriding this as it will not be possible to backtest when it is changed.
        """
        if (
            self.stop_loss is not None
            and last_signal == TradingSignal.BUY
            and price <= self.stop_loss
        ):
            return (TradingSignal.SELL, f"Stop loss: price ({price}) is below stop loss ({self.stop_loss})")
        else:
            return None

    def need_ticks(self, last_signal: TradingSignal) -> bool:
        """
        This should be overwritten with a function returning false is a stoploss or trailing stoploss is not used.
        This is used for optimizing the backtest.
        """
        return last_signal == TradingSignal.BUY and self.stop_loss is not None

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

    def generate_features(self, candlesticks: pd.DataFrame) -> pd.DataFrame:
        """Should return a dataframe containing all features needed by this strategy (for all its models etc)"""
        features = pd.DataFrame(index=candlesticks.index)
        for model in self.models:
            features = model.generate_features(candlesticks, features)

        return features

    def __train(self, candlesticks: pd.DataFrame, features: pd.DataFrame) -> None:
        """
        Train all models etc. that needs training.
        """
        targets = self._generate_targets(candlesticks, features)
        (
            training_set_features,
            training_set_targets,
            _,
            _,
        ) = split_features_and_target_into_train_and_test_set(features, targets, 0)

        for model in self.models:
            model.train(training_set_features, training_set_targets[model])

    def _generate_targets(
        self, candlesticks: pd.DataFrame, features: pd.DataFrame
    ) -> Dict[Any, pd.Series]:
        """
        Internal method used genrate features for models used by the strategy. Can be used fro outside for backtesting
        (raw) target (CHEATING).
        """
        targets = {}

        for model in self.models:
            targets[model] = model.generate_target(candlesticks, features)

        return targets
