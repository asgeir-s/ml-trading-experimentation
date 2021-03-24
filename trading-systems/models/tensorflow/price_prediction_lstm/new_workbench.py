# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## RegressionBabyModel

# %%
import os, sys

sys.path.insert(0, os.path.abspath("../../.."))


# %%
from models.tensorflow.price_prediction_lstm import PricePreditionLSTMModel
import pandas as pd
from lib.data_splitter import split_features_and_target_into_train_and_test_set
from lib.data_util import load_candlesticks

# %%
candlesticks = load_candlesticks("BTCUSDT", "1h", custom_data_path="../../../tmp")

candlesticks


# %%
forward_look_for_target = 6
model = PricePreditionLSTMModel(target_name="high", forward_look_for_target=forward_look_for_target)
features = pd.DataFrame(index=candlesticks.index)

features = model.generate_features(candlesticks, features)
target = model.generate_target(candlesticks, features)

features


# %%
target.describe()
target.value_counts()


# %%
(
    training_set_features,
    training_set_targets,
    test_set_features,
    test_set_targets,
) = split_features_and_target_into_train_and_test_set(features, {0: target}, 20)

# remove the last rows where there are no target
test_set_features = test_set_features[: -forward_look_for_target]
test_set_targets[0] = test_set_targets[0][: -forward_look_for_target]

# %%
model.train(training_set_features, training_set_targets[0])

# %%
model.evaluate(test_set_features, test_set_targets[0])

# %%
predictions = model.predict_dataframe(test_set_features)
print(classification_report(test_set_targets[0], predictions))




# %%

        targets = self._generate_targets(candlesticks, features)
        (
            training_set_features,
            training_set_targets,
            _,
# %%
start_index = 10000
split_index = 25000

train_candles = candlesticks[start_index:split_index]
test_candles = candlesticks[split_index:].reset_index().drop(columns=["index"])

train_features = model.generate_features(train_candles, reset_scaler=True)
test_features = model.generate_features(test_candles, reset_scaler=False)
train_target = model.generate_target(train_candles)
test_target = model.generate_target(test_candles)
# remove last rows where the target is nan
test_features = test_features[:-forward_look_for_target]
test_target = test_target[:-forward_look_for_target]

# %%

test_features.head(2000)


# %%
test_target.describe()


# %%
# (
#     training_set_features,
#     training_set_targets,
#     test_set_features,
#     test_set_targets,
# ) = split_features_and_target_into_train_and_test_set(features, {0: target}, 20)


# %%
# scatte = scatter_matrix(test_set_features.iloc[:, -5:], c=test_set_target.iloc[:], s=40, hist_kwds={"bins": 15}, figsize=(20,20))

# %%
model.train(train_features, train_target)

# %%
model.evaluate(test_features, test_target)

# %%
model.print_info()

# %%
prediction = model.predict(test_candles, test_features)

prediction
# prediction
# %%
