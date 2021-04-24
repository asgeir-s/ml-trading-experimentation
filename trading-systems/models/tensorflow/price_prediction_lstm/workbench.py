# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## RegressionBabyModel

# %%
from numpy.lib.histograms import _ravel_and_check_weights
import os, sys

import matplotlib.pyplot as plt
import seaborn as sns

from pandas.core.frame import DataFrame

sys.path.insert(0, os.path.abspath("../../.."))


# %%
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from lib.data_util import load_candlesticks
from lib.data_windows import create_windows, windowed_dataset
from sklearn.metrics import mean_squared_error, make_scorer, mean_absolute_error
from features.bukosabino_ta import default_features, macd, roc
from sklearn.preprocessing import StandardScaler
from lib.window_generator import WindowGenerator
from lib.backtest import setup_file_path

ASSET = "LTC"
BASE_ASSET = "USDT"
CANDLESTICK_INTERVAL = "1h"

tmp_path = (
    "../../../tmp/targets/"
    + BASE_ASSET
    + ASSET
    + "-"
    + CANDLESTICK_INTERVAL
    + "/"
)
path_builder = setup_file_path(tmp_path)

# %%
candlesticks = load_candlesticks(ASSET + BASE_ASSET, CANDLESTICK_INTERVAL, custom_data_path="../../../tmp")

candlesticks

# %%
raw_input_cols = [
    "open",
    "high",
    "low",
    "close",
    # "volume",
    # "quote asset volume",
    # "number of trades",
    # "taker buy base asset volume",
    # "taker buy quote asset volume",
]

# %% Compute raw targets
forward_look_for_target = 6
candlesticks[f"target-max-high-next-{forward_look_for_target}"] = (
    candlesticks["high"].rolling(forward_look_for_target).max().shift(-forward_look_for_target)
)
candlesticks[f"target-min-low-next-{forward_look_for_target}"] = (
    candlesticks["low"].rolling(forward_look_for_target).min().shift(-forward_look_for_target)
)
candlesticks[f"target-close-in-{forward_look_for_target}"] = (
    candlesticks["close"].shift(-forward_look_for_target)
)
target_cols = [col for col in candlesticks.columns if "target" in col]
# %% Compute relative inputs and target
relative_input = ((candlesticks[raw_input_cols] / candlesticks[raw_input_cols].shift(-1)) - 1) * 100
relative_input = relative_input.replace([np.inf, -np.inf], np.nan)
relative_input = relative_input.fillna(0)
relative_input = relative_input.clip(-5, 5)

relative_target = (candlesticks[target_cols].div(candlesticks["close"], axis=0) - 1)* 100

data = pd.concat([relative_input, relative_target], join="inner", axis=1)

# %% Compute TA features TODO: add some features
zero = pd.DataFrame(index=candlesticks.index)
computed_features = default_features.compute(candlesticks[raw_input_cols + ["volume"]], zero)
# computed_features = computed_features[[col for col in computed_features.columns if "trend" in col]]

# %%
start_index = 10000
split_index = 25000
# x_train = candlesticks[input_cols][:split_index]
# y_train = candlesticks[:split_index]["close"].shift(-1)
# x_test = candlesticks[input_cols][split_index:-1]  # remove the last element due to the target being nan
# y_test = candlesticks[split_index:]["close"].shift(-1)[:-1]

train_computed = computed_features[start_index:split_index]
test_computed = computed_features[split_index:-forward_look_for_target]
train_relative = data[start_index:split_index].reset_index().drop(columns=["index"])
test_relative = data[split_index:-forward_look_for_target].reset_index().drop(columns=["index"])


# %% TODO: add later for scaling TA
scaler = StandardScaler().fit(train_computed)

train_computed_scaled = DataFrame(
    scaler.transform(train_computed), columns=[computed_features.columns]
)
test_computed_scaled = DataFrame(
    scaler.transform(test_computed), columns=[computed_features.columns]
)

# %%
train = pd.concat([train_relative, train_computed_scaled], axis=1, join="inner")
test = pd.concat([test_relative, test_computed_scaled], axis=1, join="inner")

# %%
plt.figure(figsize=(12, 6))
ax = sns.violinplot(data=test)
_ = ax.set_xticklabels(test.keys(), rotation=90)

# %%
target = target_cols[2]
window_size = 100
number_of_inputs = len(raw_input_cols) + len(test_computed.columns)

w_train = WindowGenerator(
    df=train,
    input_width=window_size,
    label_width=1,
    shift=0,
    label_columns=[target],
)
w_test = WindowGenerator(
    df=test,
    input_width=window_size,
    label_width=1,
    shift=0,
    label_columns=[target],
)


# %%
for example_inputs, example_labels in w_test.dataset.take(1):
    print(f"Inputs shape (batch, time, features): {example_inputs.shape}")
    print(f"Labels shape (batch, time, features): {example_labels.shape}")

# %%
for example_inputs, example_labels in w_train.dataset.take(1):
    print(f"Inputs shape (batch, time, features): {example_inputs.shape}")
    print(f"Labels shape (batch, time, features): {example_labels.shape}")

# %%
# w1.plot(plot_col=target)

# %% create windows old
# window_size = 200
# shuffle_buffer = 1000
# batch_size = 100
# number_of_inputs = len(train.columns)
# train_windows_ds = windowed_dataset(train, window_size, batch_size, shuffle_buffer)
# test_windows_ds = windowed_dataset(test, window_size, batch_size, shuffle_buffer)

# %%
# for e in test_windows_ds.take(3):
#     print(e[0])
#     print(e[1])

# %% [markdown]
# #### Naive forcast (baseline)
# The forcasted value is the same as the current.
# Results raw candles candles (mse: 28379, mae: 77)
# Results relative candles: (mse: 4.7679e-05, mae: 0.004189)

naive_forecast = test[target]
naive_forecast = pd.DataFrame(0, index=np.arange(len(test)), columns=[target])
# naive_forecast = naive_forecast[1:]

# y = test.shift(1)["close"]
# y = y[1:]

print(f"mse: {mean_squared_error(test[target], naive_forecast)}")
print(f"mae: {mean_absolute_error(test[target], naive_forecast)}")

# %%
# ## Scaling the data.

# from sklearn.preprocessing import MinMaxScaler

# input_scaler = MinMaxScaler()
# x_train_scaled = input_scaler.fit_transform(x_train)
# x_test_scaled = input_scaler.transform(x_test)

# output_scaler = MinMaxScaler()
# y_train_scaled = output_scaler.fit_transform(y_train.to_numpy().reshape(-1, 1))
# y_test_scaled = output_scaler.transform(y_test.to_numpy().reshape(-1, 1))

# %%
# %%
keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)

inputs = keras.Input(shape=(window_size, number_of_inputs))
x = keras.layers.Conv1D(64, kernel_size=5, strides=1, padding="causal", activation="relu")(inputs)
x = keras.layers.LSTM(units=64, return_sequences=True)(x)
x = keras.layers.LSTM(units=64)(x)
x = keras.layers.Dense(32, activation="relu")(x)
x = keras.layers.Dense(12, activation="relu")(x)
outputs = keras.layers.Dense(1, activation="elu")(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="close_price_prediction")
model.summary()

model.compile(
    loss="mean_absolute_error", optimizer="adam", metrics=["mse", "mae"],
)

history = model.fit(w_train.dataset, batch_size=64, epochs=10, validation_data=w_test.dataset)

# %%
history = model.fit(w_train.dataset, batch_size=64, epochs=10, validation_data=w_test.dataset)
# predictions = model.predict(test)

# # naive_tf_forcast = output_scaler.inverse_transform(predictions)

# print(f"mse: {mean_squared_error(y_test, predictions)}")
# print(f"mae: {mean_absolute_error(y_test, predictions)}")
# %%

# %%
