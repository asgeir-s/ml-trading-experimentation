from ta.trend import EMAIndicator
import pandas as pd


def generate_target(candlestics: pd.DataFrame) -> pd.Series:
    return EMAIndicator(close=candlestics["close"], window=12, fillna=True).ema_indicator()

