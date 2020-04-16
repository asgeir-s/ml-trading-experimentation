import pandas as pd
import ta


def compute(df: pd.DataFrame, n_slow: int, n_fast: int, n_signal: int) -> pd.DataFrame:
    """
    default from lib: n_slow: int = 26, n_fast: int = 12, n_sign: int = 9
    """
    indicator = ta.trend.MACD(
        close=df["close"], n_slow=n_slow, n_fast=n_fast, n_sign=n_signal, fillna=False
    )

    prefix_name = str(n_slow) + "_" + str(n_fast) + "_" + str(n_signal)

    return pd.DataFrame(
        {
            "trend_macd-" + prefix_name: indicator.macd(),
            "trend_macd_diff-" +prefix_name: indicator.macd_diff(),
            "trend_macd_signal-" +prefix_name: indicator.macd_signal(),
        }
    )

