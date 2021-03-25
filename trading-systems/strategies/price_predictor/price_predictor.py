from lib.strategy import Strategy
import pandas as pd
from models.tensorflow.price_prediction_lstm import PricePreditionLSTMModel
from lib.tradingSignal import TradingSignal
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np


@dataclass
class PricePredictor(Strategy):
    def __post_init__(self) -> None:
        self.models = (
            PricePreditionLSTMModel(target_name="close", forward_look_for_target=1),
            PricePreditionLSTMModel(target_name="low"),
            PricePreditionLSTMModel(target_name="high"),
        )

    def on_candlestick(
        self, candlesticks: pd.DataFrame, trades: pd.DataFrame
    ) -> Optional[Tuple[TradingSignal, str]]:
        candlestics_to_use = candlesticks.tail(300).reset_index().drop(columns=["index"])
        features = self.generate_features(
            candlestics_to_use
        )  # Added here to not recompute all features only the last 1000
        return self.on_candlestick_with_features(candlesticks, features, trades)

    def on_candlestick_with_features_and_perdictions(
        self,
        candlesticks: pd.DataFrame,
        features: pd.DataFrame,
        trades: pd.DataFrame,
        predictions: Dict[Any, float],
    ) -> Optional[Tuple[TradingSignal, str]]:
        last_time, last_signal, last_price = self.get_last_trade(trades)

        if last_signal is None:
            # TODO: find a way to set this automatic on first run. For now this must be adjusted to fit the position on the exchange on the first run.
            last_signal = TradingSignal.SELL
            print(
                f"WARNING: theire are no previous trades for this tradingsystem. The position on the exchange needs to be {last_signal}"
            )

        close_prediction = predictions[self.models[0]]
        lowest_min_prediction = predictions[self.models[1]]
        highest_high_prediction = predictions[self.models[2]]

        print("close_preditcion:", close_prediction)
        print("lowest_min_prediction:", lowest_min_prediction)
        print("highest_high:", highest_high_prediction)
        print("current position (last signal):", last_signal)

        if np.nan in (close_prediction, lowest_min_prediction, highest_high_prediction):
            print("THE PREDITED VALUE IS NAN!!")
            print(f"lastsignal: {last_signal}")
            print(f"last_trade_price: {last_price}")
            print(f"close prediction: {close_prediction}")
            print(f"lowestlow: {lowest_min_prediction}")
            print(f"highesthigh: {highest_high_prediction}")
            print(features.last())
            print(candlesticks.last())

        signal: Optional[Tuple[TradingSignal, str]] = None
        if (
            last_signal == TradingSignal.SELL
            and highest_high_prediction > 1
            and (
                highest_high_prediction > (2.0 * (lowest_min_prediction * -1))
                or lowest_min_prediction > 0
            )
            and close_prediction > 0
        ):
            current_price = candlesticks.tail(1)["close"].values[0]
            self.stop_loss = current_price * ((100 + lowest_min_prediction * 1.05)) / 100
            # self.take_profit = current_price * ((100 + highest_high_prediction * 0.95)) / 100
            print(f"Buy signal at: {current_price}")
            print("stoploss: ", self.stop_loss)
            print("take_profit: ", self.take_profit)
            print(f"close prediction: {close_prediction}")
            print(f"lowestlow: {lowest_min_prediction}")
            print(f"highesthigh: {highest_high_prediction}")
            signal = (
                TradingSignal.BUY,
                "Its predicted that the highest high will be more then two times the lowest low and the close price is expected to be highter",
            )
        elif (
            last_signal == TradingSignal.BUY
            and close_prediction < -0.1
            # and (lowest_min_prediction * -1) > highest_high_prediction * 1.2
            # and lowest_min_prediction < -0.6
        ):
            current_price = candlesticks.tail(1)["close"].values[0]
            print(f"Sell signal at: {current_price}")
            print(f"close prediction: {close_prediction}")
            print(f"lowestlow: {lowest_min_prediction}")
            print(f"highesthigh: {highest_high_prediction}")
            signal = (
                TradingSignal.SELL,
                "The close price is predicted to go down more then 0.1%",
            )
        return signal
