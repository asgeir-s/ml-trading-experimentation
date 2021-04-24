import pandas as pd
import ta
from ta.trend import MACD
from lib.feature_util import get_name


def compute(
    candlesticks: pd.DataFrame,
    features_already_computed: pd.DataFrame,
    window_slow: int,
    window_fast: int,
    window_signal: int,
) -> pd.DataFrame:
    """
    default from lib: window_slow: int = 26, window_fast: int = 12, window_sign: int = 9
    """
    prefix = "trend"
    postfix = str(window_slow) + "_" + str(window_fast) + "_" + str(window_signal)
    
    indicator = MACD(
        close=candlesticks["close"], window_slow=window_slow, window_fast=window_fast, window_sign=window_signal, fillna=False
    )

    indicator_names = ["macd", "macd_diff", "macd_signal"]

    for indicator_name in indicator_names:
        indicator_full_name = get_name(prefix, indicator_name, postfix)
        if indicator_full_name not in features_already_computed.columns:
            features_already_computed[indicator_full_name] = getattr(indicator, indicator_name)()

    return features_already_computed
