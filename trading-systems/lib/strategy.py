import pandas as pd
import abc
from dataclasses import dataclass, field
from lib.tradingSignal import TradingSignal
from typing import Optional, Any, Tuple, Dict
from lib.model import Model
from lib.data_splitter import split_features_and_target_into_train_and_test_set


@dataclass  # type: ignore
class Strategy(abc.ABC):
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    models: Tuple[Model, ...] = ()
    min_value_asset: float = 0.0002
    min_value_base_asset: float = 0.0002
    backtest: bool = False
    configurations: Dict[str, str] = field(default_factory=dict)
    first_run: bool = True

    @abc.abstractmethod
    def __post_init__(self) -> None:
        """
        The models should be initiated here.
        """

    def init(self, candlesticks: pd.DataFrame, features: pd.DataFrame) -> None:
        """
        Should be called when sitting up a strategy.
        """
        print("min_value_base_asset:", self.min_value_base_asset)
        print("min_value_asset:", self.min_value_asset)
        self.__train(candlesticks, features)

    def on_candlestick(
        self, candlesticks: pd.DataFrame, trades: pd.DataFrame, status: Dict = {}
    ) -> Optional[Tuple[TradingSignal, str, Optional[float]]]:
        """All or a window of candlesticks up until the newest (.tail(1)) and all earlyer signals."""
        features = self.generate_features(candlesticks)
        return self.on_candlestick_with_features(candlesticks, features, trades, status)

    def on_candlestick_with_features(
        self,
        candlesticks: pd.DataFrame,
        features: pd.DataFrame,
        trades: pd.DataFrame,
        status: Dict = {},
    ) -> Optional[Tuple[TradingSignal, str, Optional[float]]]:
        """
        It calls the __train method every nth execution.

        It also calls predict on every model and calls on_candlestick_with_features_and_perdictions with the
        predictions.
        """
        if len(candlesticks) % 720 == 0:
            if len(candlesticks) > len(features):
                features = self.generate_features(
                    candlesticks
                )  # Added here to recompute all features only when traing is starting, if only the latest features are computed (for optimization)
            print("Strategy - Start retraining.")
            self.__train(candlesticks, features)
            print("Strategy - End retraining.")

        if len(candlesticks) > len(features):
            candlesticks = candlesticks.tail(
                len(features)
            )  # make sure we limit the omout of candles to the same as the features (if optimized)

        predictions = {}
        for model in self.models:
            predictions[model] = model.predict(candlesticks, features)

        # print(predictions)

        return self.on_candlestick_with_features_and_perdictions(
            candlesticks, features, trades, predictions, status
        )

    @abc.abstractmethod
    def on_candlestick_with_features_and_perdictions(
        self,
        candlesticks: pd.DataFrame,
        features: pd.DataFrame,
        trades: pd.DataFrame,
        predictions: Dict[Any, float],
        status: Dict = {},
    ) -> Optional[Tuple[TradingSignal, str, Optional[float]]]:
        """
        (mostly) Internal method for calling with the features and the predictions.
        Can be used from outside for testing targets.
        """

    def on_tick(
        self, price: float, last_signal: TradingSignal
    ) -> Optional[Tuple[TradingSignal, str, Optional[float]]]:
        """
        Checks if the stoploss should be executed. Should be called on every tick in live mode.
        Be carful overriding this as it will not be possible to backtest when it is changed.

        This is used as a sefety if the stoploss order added to the exchange when buying is not working or or is cancled or something
        """
        # if not self.backtest:
        #     return None
        if (
            self.stop_loss is not None
            and last_signal == TradingSignal.BUY
            and price <= self.stop_loss
        ):
            return (
                TradingSignal.SELL,
                f"Stop loss: price ({price}) is below stop loss ({self.stop_loss})",
                None
            )
        elif (
            self.take_profit is not None
            and last_signal == TradingSignal.BUY
            and price >= self.take_profit
        ):
            return (
                TradingSignal.SELL,
                f"Take profit: price ({price}) is above take profit ({self.take_profit})",
                None
            )
        else:
            return None

    def need_ticks(self, last_signal: TradingSignal) -> bool:
        """
        This should be overwritten with a function returning false is a stoploss or trailing stoploss is not used.
        This is used for optimizing the backtest.
        """
        return last_signal == TradingSignal.BUY and (
            self.stop_loss is not None or self.take_profit is not None
        )

    @staticmethod
    def get_last_trade(
        trades: Optional[pd.DataFrame],
    ) -> Tuple[Optional[int], Optional[TradingSignal], Optional[float]]:
        if trades is None or len(trades) == 0:
            return None, None, None
        else:
            last = trades.tail(1)
            time = int(last["transactTime"].values[0])
            signal = (
                TradingSignal.BUY if "BUY" in str(last["signal"].values[0]) else TradingSignal.SELL
            )
            price = float(last["price"].values[0])
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
        if (
            self.configurations.get("loadModelFromPath", "False") == "True" and self.first_run
        ):  # and is first run
            for model in self.models:
                model.load_model(len(training_set_features.columns))
        else:
            for model in self.models:
                model.train(training_set_features, training_set_targets[model])
        self.first_run = False

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
