from lib.position import Position, Stop_loss_Take_profit
from lib.strategy import Strategy
import pandas as pd
from models.tensorflow.price_prediction_lstm import (
    PricePreditionLSTMModelOld as PricePreditionLSTMModel,
)
from lib.tradingSignal import TradingSignal
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class PricePredictorV2(Strategy):
    highest_high_buy_threshold: float = 1
    close_prediction_sell_threshold: float = 1

    def __post_init__(self) -> None:
        self.models = (
            PricePreditionLSTMModel(
                target_name="close",
                forward_look_for_target=2,
                model_path=self.configurations["closeModelPath"],
                should_save_model=(self.configurations["saveModelToPath"] == "True"),
            ),
            PricePreditionLSTMModel(
                target_name="low",
                # forward_look_for_target=5,
                model_path=self.configurations["lowModelPath"],
                should_save_model=(self.configurations["saveModelToPath"] == "True"),
            ),
            PricePreditionLSTMModel(
                target_name="high",
                # forward_look_for_target=5,
                model_path=self.configurations["highModelPath"],
                should_save_model=(self.configurations["saveModelToPath"] == "True"),
            ),
        )
        self.highest_high_buy_threshold = float(
            self.configurations["highestHighBuyThreshold"]
        )
        self.close_prediction_sell_threshold = float(
            self.configurations["closePredictionSellThreshold"]
        )
        print("Highest High buy threshold:", self.highest_high_buy_threshold)
        print("Close prediction sell threshold:", self.close_prediction_sell_threshold)

    def on_candlestick(
        self, candlesticks: pd.DataFrame, trades: pd.DataFrame, status: Dict = {}
    ) -> Optional[Position]:
        candlestics_to_use = (
            candlesticks.tail(300).reset_index().drop(columns=["index"])
        )
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
    ) -> Optional[Position]:
        asset_balance = status["asset_balance"]
        base_asset_balance = status["base_asset_balance"]
        take_profit_price = status["take_profit_price"]
        stop_loss_price = status["stop_loss_price"]

        close_prediction = predictions[self.models[0]]
        lowest_min_prediction = predictions[self.models[1]]
        highest_high_prediction = predictions[self.models[2]]

        current_price = candlesticks.tail(1)["close"].values[0]

        # if not self.backtest:
        print("Close preditcion:", close_prediction)
        print("Lowest min prediction:", lowest_min_prediction)
        print("Highest high:", highest_high_prediction)

        if np.nan in (close_prediction, lowest_min_prediction, highest_high_prediction):
            print("THE PREDITED VALUE IS NAN!!")

        new_position: Optional[Position] = None
        if (
            base_asset_balance > self.min_value_base_asset
            and highest_high_prediction > self.highest_high_buy_threshold
            and (
                highest_high_prediction > (1.6 * (lowest_min_prediction * -1))
                or lowest_min_prediction > 0
            )
            and close_prediction > 0.02
        ):
            stop_loss_price = (
                current_price * ((100 + lowest_min_prediction * 1.05)) / 100
            )
            print(f"Buy signal at: {current_price}")
            print("Stoploss: ", stop_loss_price)
            new_position = Position(
                signal=TradingSignal.LONG,
                reason="Its predicted that the highest high will be more then two times the lowest low and the close price is"
                " expected to be highter",
                stop_loss_take_profit=Stop_loss_Take_profit(stop_loss=stop_loss_price),
                data={
                    "close_prediction": close_prediction,
                    "lowest_min_prediction": lowest_min_prediction,
                    "highest_high_prediction": highest_high_prediction,
                },
            )
        elif asset_balance > self.min_value_asset and (
            close_prediction < self.close_prediction_sell_threshold
            or (
                ((lowest_min_prediction * -1) > highest_high_prediction)
                and close_prediction < 0
            )
        ):
            current_price: float = candlesticks.tail(1)["close"].values[0]
            print(f"Sell signal at: {current_price}")
            new_position = Position(
                signal=TradingSignal.CLOSE,
                reason=f"The close price is predicted to go down more then {self.close_prediction_sell_threshold}%",
                data={
                    "close_prediction": close_prediction,
                    "lowest_min_prediction": lowest_min_prediction,
                    "highest_high_prediction": highest_high_prediction,
                },
            )
        return new_position
