# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## ClassifierSklienSimpleModel

# %%
import os, sys
from sklearn.metrics import classification_report
from models.xgboost import ClassifierSklienSimpleModel
import pandas as pd
from lib.data_splitter import split_features_and_target_into_train_and_test_set
from lib.data_util import load_candlesticks

sys.path.insert(0, os.path.abspath("../../.."))

# %%
candlesticks = load_candlesticks("BTCUSDT", "1h", custom_data_path="../../../tmp")

candlesticks


# %%
model = ClassifierSklienSimpleModel()

features = model.generate_features(
    candlesticks, pd.DataFrame(index=candlesticks.index)
)
target = model.generate_target(candlesticks, features)

target.head(20)


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
model.train(training_set_features, training_set_targets[0])

# %%
model.evaluate(test_set_features, test_set_targets[0])

# %%
predictions = model.predict_dataframe(test_set_features)
print(classification_report(test_set_targets[0], predictions))


# %%
""" raw_model = xgb.XGBRegressor(objective="reg:squarederror")

param_dist = {
    "n_estimators": stats.randint(150, 1000),
    "learning_rate": stats.uniform(0.01, 0.6),
    "subsample": stats.uniform(0.3, 0.9),
    "max_depth": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 25],
    "colsample_bytree": stats.uniform(0.5, 0.9),
    "min_child_weight": [1, 2, 3, 4, 5, 10],
}
rmse = make_scorer(mean_squared_error, greater_is_better=False)

r = RandomizedSearchCV(raw_model, param_distributions=param_dist, scoring=rmse, n_iter=3, n_jobs=2)
r.fit(training_set_features, training_set_targets[0])

r.score(test_set_features, test_set_targets[0]) """


# %%
# r.best_params_

model.train(training_set_features, training_set_targets[0])

# %%

print(classification_report(test_set_targets[0], predictions))
