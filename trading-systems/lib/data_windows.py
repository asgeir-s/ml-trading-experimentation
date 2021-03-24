import pandas as pd
import numpy as np
from pandas import DataFrame
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf


def create_windows(df: DataFrame, window_size: int):
    df_in = df.copy()
    # Put your inputs into a single list
    new_df = pd.DataFrame(df_in.apply(tuple, axis=1).apply(list), columns=["single_input_vector"])
    # Double-encapsulate list so that you can sum it in the next step and keep time steps as separate elements
    new_df["single_input_vector"] = new_df["single_input_vector"].apply(lambda x: [list(x)])
    # Use .cumsum() to include previous row vectors in the current row list of vectors
    new_df["cumulative_input_vectors"] = new_df["single_input_vector"].cumsum()

    # Pad your sequences so they are the same length
    padded_sequences = pad_sequences(
        new_df["cumulative_input_vectors"].tolist(), window_size
    ).tolist()
    new_df["padded_input_vectors"] = pd.Series(padded_sequences).apply(np.asarray)

    # Extract your training data
    data = np.asarray(new_df["padded_input_vectors"])

    # Use hstack to and reshape to make the inputs a 3d vector
    array_of_windows = np.hstack(data).reshape(len(new_df), window_size, len(df.columns))

    return array_of_windows


def windowed_dataset(series, window_size: int, batch_size: int, shuffle_buffer: int):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[-1][3]))  # element 3 is close price
    # ds = ds.filter(lambda x, y: not (tf.math.is_nan(y) or tf.math.is_inf(y)))
    # return ds
    return ds.batch(batch_size).prefetch(1)
