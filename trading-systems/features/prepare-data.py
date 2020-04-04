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
    upTreshold = 1.003
    downTreshold = 0.998
    conditions = [
        ((1 / df["trend_sma_fast"]) * ((df.shift(periods=-1)["trend_sma_fast"] + df.shift(periods=-2)["trend_sma_fast"])/2) > upTreshold),
        ((1 / df["trend_sma_fast"]) * ((df.shift(periods=-1)["trend_sma_fast"] + df.shift(periods=-2)["trend_sma_fast"])/2) < downTreshold),
    ]
    choices = [2,0]
    df["target"] = np.select(conditions, choices, default=1)
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
