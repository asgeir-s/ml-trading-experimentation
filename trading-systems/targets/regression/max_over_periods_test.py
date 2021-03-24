import pandas as pd
import numpy as np
from .max_over_periods import generate_target


def test_generate_target_max():
    column_name_1 = "column1"
    column_name_2 = "column2"
    df = pd.DataFrame(
        columns=[column_name_1, column_name_2],
        data=[
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [5, 6],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [6, 7],
            [1, 2],
            [1, 2],
        ],
    )
    first_test = generate_target(df, column_name_1, 3)
    assert (first_test[:-1] == [1, 1, 5, 5, 5, 1, 1, 1, 6, 6, 6, 1, 1]).all()
    second_test = generate_target(df, column_name_2, 10)
    assert (second_test[:-1] == [6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 2, 2]).all()


def test_generate_target_min():
    column_name_1 = "column1"
    column_name_2 = "column2"
    df = pd.DataFrame(
        columns=[column_name_1, column_name_2],
        data=[
            [7, 2],
            [7, 2],
            [7, 2],
            [7, 2],
            [7, 2],
            [5, 6],
            [7, 2],
            [7, 2],
            [7, 2],
            [7, 2],
            [7, 2],
            [6, 7],
            [7, 2],
            [7, 2],
        ],
    )
    first_test = generate_target(df, column_name_1, 3, min=True)
    assert (first_test[:-1] == [7, 7, 5, 5, 5, 7, 7, 7, 6, 6, 6, 7, 7]).all()
    second_test = generate_target(df, column_name_2, 10, min=True)
    assert (second_test[:-1] == [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]).all()


def test_generate_target_max_percentage():
    column_name_1 = "column1"
    column_name_2 = "column2"
    df = pd.DataFrame(
        columns=[column_name_1, column_name_2],
        data=[
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [5, 6],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [6, 9],
            [6, 9],
            [6, 9],
        ],
    )
    first_test = generate_target(df, column_name_1, 3, percentage=True)
    assert (first_test[:-1] == [0, 0, 4, 4, 4, -0.8, 0, 0, 5, 5, 5, 0, 0]).all()
    second_test = generate_target(df, column_name_2, 10, percentage=True)
    assert (second_test[:-1] == [2, 3.5, 3.5, 3.5, 3.5, 0.5, 3.5, 3.5, 3.5, 3.5, 3.5, 0, 0]).all()
