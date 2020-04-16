import pandas as pd
import ta


def compute(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    default from lib: n = 12
    """
    indicator = ta.momentum.ROCIndicator(close=df["close"], n=n, fillna=False)
    prefix = "momentum"
    name = "roc"
    post_name = str(n)

    full_name = prefix + "_" + name + "-" + post_name

    return pd.DataFrame({full_name: indicator.roc(),})
