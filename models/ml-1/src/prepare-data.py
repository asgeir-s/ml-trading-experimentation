import pandas as pd
import numpy as np
import ta

from sklearn.model_selection import train_test_split


def loadData() -> pd.DataFrame:
    df = pd.read_csv("../../data-loading/binance/data/candlestick-BTCUSDT-1h.csv").drop(
        columns=["ignore"]
    )
    return df


def createTarget(df: pd.DataFrame) -> pd.DataFrame:
    upTreshold = 1.04
    downTreshold = 0.98
    conditions = [
        ((1 / df["close"]) * df.shift(periods=-3)["close"]) > upTreshold,
        ((1 / df["close"]) * df.shift(periods=-3)["close"]) < downTreshold,
    ]
    choices = [1, -1]
    df["target"] = np.select(conditions, choices, default=0)
    return df


def createFeatures(df: pd.DataFrame) -> pd.DataFrame:
    # Add ta features filling NaN values
    df = ta.add_all_ta_features(
        df,
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
        fillna=True,
    )

    return df


candlesticks = loadData()
candlesticsWithFeatures = createFeatures(candlesticks)
candlesticsAndFeaturesWithTarget = createTarget(candlesticsWithFeatures)
size = candlesticsAndFeaturesWithTarget.size
print(f"{size}")
print(candlesticsAndFeaturesWithTarget.describe())
candlesticsAndFeaturesWithTarget.to_csv("./data/candlesticsAndFeaturesWithTarget.csv")
