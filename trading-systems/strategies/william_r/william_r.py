from lib.position import Position, Stop_loss_Take_profit
from lib.strategy import Strategy
import pandas as pd
from lib.tradingSignal import TradingSignal
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class WilliamR(Strategy):

    def __post_init__(self) -> None:
        pass

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

        feature_name = [name for name in features.columns if "trend_aroon_ind" in name][0]
        william_r = features.tail(1)[feature_name].values[0] * -1
        current_price: float = candlesticks.tail(1)["close"].values[0]
        # if not self.backtest:

        # if close_t1 is None:  # close_t2, high_pred):
        #     print("THE PREDICTED VALUE IS NAN!!"r

        new_position: Optional[Position] = None
        if (
            base_asset_balance > self.min_value_base_asset
            and william_r > 25
        ):
            stop_loss_price = current_price * 0.98  # 1% ned
            print(f"Buy signal at: {current_price}")
            print("Stoploss: ", stop_loss_price)
            new_position = Position(
                signal=TradingSignal.LONG,
                reason="William R < 10",
                stop_loss_take_profit=Stop_loss_Take_profit(stop_loss=stop_loss_price),
                data={
                    "william_r": william_r,
                },
            )
        elif asset_balance > self.min_value_asset and william_r < 50:
            current_price: float = candlesticks.tail(1)["close"].values[0]
            print(f"Sell signal at: {current_price}")
            new_position = Position(
                signal=TradingSignal.CLOSE,
                reason=f"William R > 50",
                data={
                    "william_r": william_r,
                },
            )
        return new_position
