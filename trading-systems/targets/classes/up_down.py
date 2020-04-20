import pandas as pd
import numpy as np


def generate_target(
    df: pd.DataFrame, column: str, up_treshold: float, down_treshold: float
) -> pd.Series:
    conditions = [
        (
            df.shift(periods=-2)[column] / df.shift(periods=-1)[column]
            > up_treshold
        ),
        (
            df.shift(periods=-2)[column] / df.shift(periods=-1)[column]
            < down_treshold
        ),
    ]
    choices = [1, 0]
    return pd.Series(np.select(conditions, choices, default=1))
