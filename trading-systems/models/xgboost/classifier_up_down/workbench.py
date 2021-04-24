# %% [markdown]
# ## ClassifierUpDownModel

# %%
import os, sys

sys.path.insert(0, os.path.abspath("../../.."))

# %%
from models.xgboost import ClassifierUpDownModel
import pandas as pd
from lib.data_splitter import split_features_and_target_into_train_and_test_set
from lib.data_util import load_candlesticks
from sklearn.metrics import classification_report
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
model = ClassifierUpDownModel()
features = pd.DataFrame(index=candlesticks.index)

features = model.generate_features(candlesticks, features)
target = model.generate_target(candlesticks, features)

features


# %%
target.describe()
target.value_counts()

# %%
target.to_csv(path_builder("classifier_up_down"))


# %%
start_running_index = 10000 # end of initial training
training_interval = 500
last_index = start_running_index
done = False
predictions = None

# %% initial training
model.train(features[0:start_running_index], target[0:start_running_index])
# %%
while last_index < len(features):
    start_index = last_index
    stop_index = (
        last_index + training_interval
        if len(features) > (last_index + training_interval)
        else len(features)
    )
    new_pred = model.predict_dataframe(features[start_index:stop_index])
    if predictions is None:
        predictions = new_pred
    else:
        predictions = predictions.append([new_pred])
    model.train(features[:stop_index], target[:stop_index])
    last_index = stop_index

# %%
predictions.to_csv(path_builder("classifier_up_down"))

# %%
print(classification_report(target[start_running_index:], predictions))

# %%
df = pd.DataFrame({"target": test_set_targets[0], "prediction": predictions})
df.to_csv("./tmp/target_predictions.csv")


# %%
predictions = pd.Series(predictions)

predictions.describe()
predictions.value_counts()


# %%
score = df["target"] == df["prediction"]
score


# %%
model.print_info()


# %%
xg_boost = RegressionSklienModel(
    model=xgb.XGBRegressor(
        objective="reg:squarederror",
        colsample_bytree=0.716549113595231,
        learning_rate=0.05488437390157618,
        max_depth=3,
        min_child_weight=2,
        n_estimators=181,
        subsample=0.33243402437274977,
    )
)
xg_boost.train(training_set_features, training_set_targets[0])
xg_boost.evaluate(test_set_features, test_set_targets[0])


# %%
predictions = xg_boost.predict_dataframe(test_set_features)
test_set_target
df = pd.DataFrame({"target": test_set_target, "prediction": predictions})
df.to_csv("./tmp/target_predictions.csv")

