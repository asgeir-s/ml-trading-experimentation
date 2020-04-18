import pandas as pd
import ta
from lib.feature_util import get_name


def compute(
    candlesticks: pd.DataFrame,
    features_already_computed: pd.DataFrame,
    n_slow: int,
    n_fast: int,
    n_signal: int,
) -> pd.DataFrame:
    """
    default from lib: n_slow: int = 26, n_fast: int = 12, n_sign: int = 9
    """
    prefix = "trend"
    postfix = str(n_slow) + "_" + str(n_fast) + "_" + str(n_signal)

    indicator = ta.trend.MACD(
        close=candlesticks["close"], n_slow=n_slow, n_fast=n_fast, n_sign=n_signal, fillna=False
    )

    indicator_names = ["macd", "macd_diff", "macd_signal"]

    for indicator_name in indicator_names:
        indicator_full_name = get_name(prefix, indicator_name, postfix)
        if indicator_full_name not in features_already_computed.columns:
            features_already_computed[indicator_full_name] = getattr(indicator, indicator_name)()

    return features_already_computed
