from lib.position import Position, Stop_loss_Take_profit
from lib.strategy import Strategy
import pandas as pd
from models.tensorflow.price_prediction_lstm import PricePreditionLSTMModel
from lib.tradingSignal import TradingSignal
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class PricePredictor2(Strategy):
    highest_high_buy_threshold: float = 1
    close_prediction_sell_threshold: float = 1

    def __post_init__(self) -> None:
        self.models = (
            PricePreditionLSTMModel(
                target_name="ema",
                forward_look_for_target=1,
                window_size=10,
                model_path=self.configurations["closeModelPath"],
                should_save_model=(self.configurations["saveModelToPath"] == "True"),
            ),
            PricePreditionLSTMModel(
                target_name="ema",
                forward_look_for_target=3,
                window_size=10,
                model_path=self.configurations["lowModelPath"],
                should_save_model=(self.configurations["saveModelToPath"] == "True"),
            ),
            PricePreditionLSTMModel(
                target_name="high",
                forward_look_for_target=3,
                window_size=10,
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
        self,
        candlesticks: pd.DataFrame,
        trades: pd.DataFrame,
        status: Dict[str, Any] = {},
    ) -> Optional[Position]:
        candlestics_to_use = (
            candlesticks.tail(100).reset_index().drop(columns=["index"])
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
        status: Dict[str, Any] = {},
    ) -> Optional[Position]:
        asset_balance = status["asset_balance"]
        base_asset_balance = status["base_asset_balance"]
        take_profit_price = status["take_profit_price"]
        stop_loss_price = status["stop_loss_price"]

        close_t1 = predictions[self.models[0]]
        close_t2 = predictions[self.models[1]]
        high_pred = predictions[self.models[2]]

        current_price: float = candlesticks.tail(1)["close"].values[0]

        if not self.backtest:
            print("Close t1 preditcion:", close_t1)
            print("Close t2 prediction:", close_t2)
            print("High prediction:", high_pred)

        if close_t1 is None:  # close_t2, high_pred):
            print("THE PREDICTED VALUE IS NAN!!")

        new_position: Optional[Position] = None
        if (
            base_asset_balance > self.min_value_base_asset
            and close_t2 > 0.5
            and close_t1 > 0.3
            and high_pred > 1.2
        ):
            stop_loss_price = current_price * 0.98  # 1% ned
            print(f"Buy signal at: {current_price}")
            print("Stoploss: ", stop_loss_price)
            new_position = Position(
                signal=TradingSignal.LONG,
                reason="All close prices are positive and bigger the the one before",
                stop_loss_take_profit=Stop_loss_Take_profit(stop_loss=stop_loss_price),
                data={
                    "close_t1": close_t1,
                    "close_t2": close_t2,
                    "high_pred": high_pred,
                },
            )
        elif asset_balance > self.min_value_asset and close_t1 < 0.1:
            current_price: float = candlesticks.tail(1)["close"].values[0]
            print(f"Sell signal at: {current_price}")
            new_position = Position(
                signal=TradingSignal.CLOSE,
                reason=f"The close price is predicted to go down more then {self.close_prediction_sell_threshold}%",
                data={
                    "close_t1": close_t1,
                    "close_t2": close_t2,
                    "high_pred": high_pred,
                },
            )
        return new_position
