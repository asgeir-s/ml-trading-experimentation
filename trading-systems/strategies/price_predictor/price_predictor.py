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
        self, candlesticks: pd.DataFrame, trades: pd.DataFrame, status: Dict = {}
    ) -> Optional[Tuple[TradingSignal, str]]:
        candlestics_to_use = candlesticks.tail(300).reset_index().drop(columns=["index"])
        features = self.generate_features(
            candlestics_to_use
        )  # Added here to not recompute all features only the last 1000
        return self.on_candlestick_with_features(candlesticks, features, trades, status)

    def on_candlestick_with_features_and_perdictions(
        self,
        candlesticks: pd.DataFrame,
        features: pd.DataFrame,
        trades: pd.DataFrame,
        predictions: Dict[Any, float],
        status: Dict = {},
    ) -> Optional[Tuple[TradingSignal, str]]:
        asset_balance = status.get("asset_balance", None)
        base_asset_balance = status.get("base_asset_balance", None)
        last_time, last_signal, last_price = self.get_last_trade(trades)

        # if we ar in
        if self.backtest:
            last_time, last_signal, last_price = self.get_last_trade(trades)
            asset_balance = 1 if last_signal == TradingSignal.BUY else 0
            base_asset_balance = 1 if last_signal == TradingSignal.SELL else 0
        else:
            asset_balance = status.get("asset_balance", None)
            base_asset_balance = status.get("base_asset_balance", None)

        close_prediction = predictions[self.models[0]]
        lowest_min_prediction = predictions[self.models[1]]
        highest_high_prediction = predictions[self.models[2]]

        print("Close_preditcion:", close_prediction)
        print("Lowest_min_prediction:", lowest_min_prediction)
        print("Highest_high:", highest_high_prediction)
        print(f"Base asset balance:", base_asset_balance)
        print(f"Asset balance:", asset_balance)

        if np.nan in (close_prediction, lowest_min_prediction, highest_high_prediction):
            print("THE PREDITED VALUE IS NAN!!")

        signal: Optional[Tuple[TradingSignal, str]] = None
        if (
            base_asset_balance > self.min_value_base_asset
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
        elif asset_balance > self.min_value_asset and close_prediction < -0.1:
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
