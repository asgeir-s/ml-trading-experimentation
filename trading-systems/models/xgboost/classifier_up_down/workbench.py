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


# %%
candlesticks = load_candlesticks("BTCUSDT", "1h", custom_data_path="../../../tmp")

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
(
    training_set_features,
    training_set_targets,
    test_set_features,
    test_set_targets,
) = split_features_and_target_into_train_and_test_set(features, {0: target}, 20)


# %%
# scatte = scatter_matrix(test_set_features.iloc[:, -5:], c=test_set_target.iloc[:], s=40, hist_kwds={"bins": 15}, figsize=(20,20))


# %%
model.train(training_set_features, training_set_targets[0])

# %%
model.evaluate(test_set_features, test_set_targets[0])

# %%
predictions = model.predict_dataframe(test_set_features)
print(classification_report(test_set_targets[0], predictions))

# %%
df = pd.DataFrame({"target": test_set_targets[0], "prediction": predictions})
df.to_csv("./tmp/target_predictions.csv")


# %%
predictions = pd.Series(predictions)

predictions.describe()
predictions.value_counts()


# %%
score = df[df["target"] == df["prediction"]]
score


# %%
model.print_info()


# %%
xg_boost = RegressionSklienModel(model=xgb.XGBRegressor(
    objective="reg:squarederror",
    colsample_bytree= 0.716549113595231,
    learning_rate= 0.05488437390157618,
    max_depth= 3,
    min_child_weight= 2,
    n_estimators= 181,
    subsample= 0.33243402437274977
    ))
xg_boost.train(training_set_features, training_set_targets[0])
xg_boost.evaluate(test_set_features, test_set_targets[0])


# %%
predictions = xg_boost.predict_dataframe(test_set_features)
test_set_target
df = pd.DataFrame({"target": test_set_target, "prediction": predictions})
df.to_csv("./tmp/target_predictions.csv")

