import pandas as pd
from typing import Union


def load_candlesticks(granularity: str) -> pd.DataFrame:
    return pd.read_csv(f"../data-loading/binance/data/candlestick-BTCUSDT-{granularity}.csv").drop(
        columns=["ignore"]
    )
