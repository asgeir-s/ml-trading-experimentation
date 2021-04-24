import pandas as pd
from ta.momentum import ROCIndicator


def compute(
    candlesticks: pd.DataFrame, features_already_computed: pd.DataFrame, window: int
) -> pd.DataFrame:
    """
    default from lib: n = 12
    """
    prefix = "momentum"
    name = "roc"
    post_name = str(window)
    full_name = prefix + "_" + name + "-" + post_name

    if full_name in features_already_computed.columns:
        return features_already_computed
    else:
        indicator = ROCIndicator(close=candlesticks["close"], window=window, fillna=False)
        features_already_computed[full_name] = indicator.roc()
        return features_already_computed
