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
    ) -> Optional[Tuple[TradingSignal, str, Optional[float]]]:
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
    ) -> Optional[Tuple[TradingSignal, str, Optional[float]]]:
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

        current_price = candlesticks.tail(1)["close"].values[0]
        # stop_loss_new = current_price * ((100 + lowest_min_prediction * 1.05)) / 100

        # if self.stop_loss is not None and stop_loss_new > self.stop_loss:
        #     self.stop_loss = stop_loss_new

        if not self.backtest:
            print("Close preditcion:", close_prediction)
            print("Lowest min prediction:", lowest_min_prediction)
            print("Highest high:", highest_high_prediction)
            print(f"Base asset balance:", base_asset_balance)
            print(f"Asset balance:", asset_balance)

        if np.nan in (close_prediction, lowest_min_prediction, highest_high_prediction):
            print("THE PREDITED VALUE IS NAN!!")

        signal: Optional[Tuple[TradingSignal, str, Optional[float]]] = None
        if (
            base_asset_balance > self.min_value_base_asset
            and highest_high_prediction > self.highest_high_buy_threshold
            and (
                highest_high_prediction > (2.0 * (lowest_min_prediction * -1))
                or lowest_min_prediction > 0
            )
            and close_prediction > 0
        ):
            self.stop_loss = current_price * ((100 + lowest_min_prediction * 1.05)) / 100
            # self.take_profit = current_price * ((100 + highest_high_prediction * 0.95)) / 100
            print(f"Buy signal at: {current_price}")
            print("Stoploss: ", self.stop_loss)
            print("Take_profit: ", self.take_profit)
            signal = (
                TradingSignal.BUY,
                "Its predicted that the highest high will be more then two times the lowest low and the close price is expected to be highter",
                self.stop_loss
            )
        elif (
            asset_balance > self.min_value_asset
            and close_prediction < self.close_prediction_sell_threshold
        ):
            current_price = candlesticks.tail(1)["close"].values[0]
            print(f"Sell signal at: {current_price}")
            signal = (
                TradingSignal.SELL,
                f"The close price is predicted to go down more then {self.close_prediction_sell_threshold}%",
                None
            )
            self.stop_loss = None
        return signal
