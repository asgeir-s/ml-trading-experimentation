from lib.strategy import Strategy
import pandas as pd
from models.tensorflow.price_prediction_lstm import PricePreditionLSTMModel
from lib.tradingSignal import TradingSignal
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np


@dataclass
class PricePredictor(Strategy):
    highest_high_buy_threshold: float = 1
    close_prediction_sell_threshold: float = 1

    def __post_init__(self) -> None:
        self.models = (
            PricePreditionLSTMModel(
                target_name="close",
                forward_look_for_target=1,
                model_path=self.configurations["closeModelPath"],
                should_save_model=(self.configurations["saveModelToPath"] == "True"),
            ),
            PricePreditionLSTMModel(
                target_name="low",
                model_path=self.configurations["lowModelPath"],
                should_save_model=(self.configurations["saveModelToPath"] == "True"),
            ),
            PricePreditionLSTMModel(
                target_name="high",
                model_path=self.configurations["highModelPath"],
                should_save_model=(self.configurations["saveModelToPath"] == "True"),
            ),
        )
        self.highest_high_buy_threshold = float(self.configurations["highestHighBuyThreshold"])
        self.close_prediction_sell_threshold = float(
            self.configurations["closePredictionSellThreshold"]
        )
        print("Highest High buy threshold:", self.highest_high_buy_threshold)
        print("Close prediction sell threshold:", self.close_prediction_sell_threshold)

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
        if self.backtest:
            last_time, last_signal, last_price = self.get_last_trade(trades)
            asset_balance = 10.0 if last_signal == TradingSignal.BUY else 0.0
            base_asset_balance = (
                10.0 if (last_signal == TradingSignal.SELL or last_signal is None) else 0.0
            )
        else:
            asset_balance = float(status["asset_balance"])
            base_asset_balance = float(status["base_asset_balance"])

        close_prediction = predictions[self.models[0]]
        lowest_min_prediction = predictions[self.models[1]]
        highest_high_prediction = predictions[self.models[2]]

        if not self.backtest:
            print("Close_preditcion:", close_prediction)
            print("Lowest_min_prediction:", lowest_min_prediction)
            print("Highest_high:", highest_high_prediction)
            print(f"Base asset balance:", base_asset_balance)
            print(f"Asset balance:", asset_balance)
            print("min_value_base_asset:", self.min_value_base_asset)
            print("highest_high_buy_threshold:", self.highest_high_buy_threshold)

        if np.nan in (close_prediction, lowest_min_prediction, highest_high_prediction):
            print("THE PREDITED VALUE IS NAN!!")

        signal: Optional[Tuple[TradingSignal, str]] = None
        if (
            base_asset_balance > self.min_value_base_asset
            and highest_high_prediction > self.highest_high_buy_threshold
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
            print("Stoploss: ", self.stop_loss)
            print("Take_profit: ", self.take_profit)
            signal = (
                TradingSignal.BUY,
                "Its predicted that the highest high will be more then two times the lowest low and the close price is expected to be highter",
            )
        elif (
            asset_balance > self.min_value_asset
            and close_prediction < self.close_prediction_sell_threshold
        ):
            current_price = candlesticks.tail(1)["close"].values[0]
            print(f"Sell signal at: {current_price}")
            signal = (
                TradingSignal.SELL,
                "The close price is predicted to go down more then 0.1%",
            )
        return signal
