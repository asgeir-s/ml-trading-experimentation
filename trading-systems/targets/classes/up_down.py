import pandas as pd
import numpy as np


def generate_target(
    df: pd.DataFrame, column: str, treshold: float
) -> pd.Series:
    conditions = [
        (
            df.shift(periods=-2)[column] / df.shift(periods=-1)[column]
            >= treshold
        ),
        (
            df.shift(periods=-2)[column] / df.shift(periods=-1)[column]
            < treshold
        ),
    ]
    choices = [1, 0]
    return pd.Series(np.select(conditions, choices, default=1))
