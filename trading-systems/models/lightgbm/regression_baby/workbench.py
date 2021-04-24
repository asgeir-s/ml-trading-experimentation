# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## RegressionBabyModel

# %%
import os, sys

sys.path.insert(0, os.path.abspath("../../.."))


# %%
from models.lightgbm import RegressionBabyMaxModel, RegressionBabyMinModel
import pandas as pd
from lib.data_splitter import split_features_and_target_into_train_and_test_set
from lib.data_util import load_candlesticks
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, make_scorer
from scipy import stats
from sklearn.model_selection import RandomizedSearchCV
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
model = RegressionBabyMaxModel()
# model = RegressionBabyMinModel()
features = pd.DataFrame(index=candlesticks.index)

features = model.generate_features(candlesticks, pd.DataFrame(index=candlesticks.index))
target = model.generate_target(candlesticks, features)

features.head(2000)


# %%
target.describe()

# %%
target.to_csv(path_builder("lightgbm_regression_baby_max"))

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
model.evaluate(test_set_features[:-1], test_set_targets[0][:-1])

# %%
model.print_info()

# %% Hyper parameter search
raw_model = lgb.LGBMRegressor(
    objective="regression",
    num_leaves=5,
    learning_rate=0.05,
    n_estimators=720,
    max_bin=55,
    bagging_fraction=0.8,
    bagging_freq=5,
    feature_fraction=0.2319,
    feature_fraction_seed=9,
    bagging_seed=9,
    min_data_in_leaf=6,
    min_sum_hessian_in_leaf=11,
)

param_dist = {
    "subsample": stats.uniform(0.3, 0.9),
    "max_depth": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 25],
    "min_child_weight": [1, 2, 3, 4, 5, 10],
    "num_leaves": [5, 10, 15, 20, 25],
    "learning_rate": [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.6],
    "n_estimators": [100, 250, 500, 720, 800, 1000],
    "max_bin": [10, 25, 55, 70, 90, 120],
    "bagging_fraction": [0.1, 0.25, 0.5, 0.75, 1],
    "bagging_freq": [1, 5, 10, 20],
    "feature_fraction": [0.2319, 0.5, 0.7, 0.8],
    "feature_fraction_seed": [1, 5, 9, 20],
    "bagging_seed": [1, 5, 9, 25],
    "min_data_in_leaf": [5, 10, 15, 20],
    "min_sum_hessian_in_leaf": [5, 11, 20, 30, 50],
}
rmse = make_scorer(mean_squared_error, greater_is_better=False)

r = RandomizedSearchCV(raw_model, param_distributions=param_dist, scoring=rmse, n_iter=3, n_jobs=2)
r.fit(training_set_features, training_set_targets[0])

r.score(test_set_features[:-1], test_set_targets[0][:-1])

# %% hyperparameter search result
# 'bagging_fraction': 0.75,
#  'bagging_freq': 20,
#  'bagging_seed': 5,
#  'feature_fraction': 0.2319,
#  'feature_fraction_seed': 20,
#  'learning_rate': 0.001,
#  'max_bin': 90,
#  'max_depth': 14,
#  'min_child_weight': 1,
#  'min_data_in_leaf': 10,
#  'min_sum_hessian_in_leaf': 5,
#  'n_estimators': 100,
#  'num_leaves': 25,
#  'subsample': 0.9170884223554043


# %%
r.best_params_

