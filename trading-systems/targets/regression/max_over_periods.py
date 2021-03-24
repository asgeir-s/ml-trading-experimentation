import pandas as pd


def generate_target(
    df: pd.DataFrame, column: str, periodes: int, min: bool = False, percentage: bool = False
) -> pd.Series:
    """
    For each row; takes the max value in the 'column' over the next 'periods' periodes and compares it
    to the current rows 'column' value. It returns the % or absolute diff (based on 'percentage').
    """
    res = (
        df[column][::-1].rolling(periodes, 1).min().shift(1)[::-1]
        if min
        else df[column][::-1].rolling(periodes, 1).max().shift(1)[::-1]
    )

    res = (res - df[column]) / df[column] if percentage else res

    return res
