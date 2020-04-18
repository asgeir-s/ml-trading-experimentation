import pandas as pd
import ta


def compute(
    candlesticks: pd.DataFrame, features_already_computed: pd.DataFrame, n: int
) -> pd.DataFrame:
    """
    default from lib: n = 12
    """
    prefix = "momentum"
    name = "roc"
    post_name = str(n)
    full_name = prefix + "_" + name + "-" + post_name

    if full_name in features_already_computed.columns:
        return features_already_computed
    else:
        indicator = ta.momentum.ROCIndicator(close=candlesticks["close"], n=n, fillna=False)
        features_already_computed[full_name] = indicator.roc()
        return features_already_computed
