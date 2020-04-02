import xgboost as xgb
from sklearn.metrics import accuracy_score, mean_squared_error
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection._validation import cross_val_score


def generteTrainAndTestSet(
    candlesticsAndFeaturesWithTarget: pd.DataFrame, percentage: int
):
    testSetSize = int(len(candlesticsAndFeaturesWithTarget) * (percentage / 100))
    trainSetSize = int(len(candlesticsAndFeaturesWithTarget) - testSetSize)

    candlesticsAndFeaturesWithTarget = candlesticsAndFeaturesWithTarget.drop(
        columns=["open time", "close time"]
    )
    candlesticsAndFeaturesWithTarget = candlesticsAndFeaturesWithTarget.drop(
        candlesticsAndFeaturesWithTarget.columns[1], axis=1
    )

    X, y = (
        candlesticsAndFeaturesWithTarget.iloc[:, :-1],
        candlesticsAndFeaturesWithTarget.iloc[:, -1],
    )

    X_train, y_train = (X.iloc[:trainSetSize], y.iloc[:trainSetSize])

    X_test, y_test = (
        X.iloc[trainSetSize : len(candlesticsAndFeaturesWithTarget)],
        y.iloc[trainSetSize : len(candlesticsAndFeaturesWithTarget)],
    )

    return X_train, y_train, X_test, y_test


def generateTrainSet(candlesticsAndFeaturesWithTarget: pd.DataFrame):
    candlesticsAndFeaturesWithTarget = candlesticsAndFeaturesWithTarget.drop(
        columns=["open time", "close time"]
    )
    candlesticsAndFeaturesWithTarget = candlesticsAndFeaturesWithTarget.drop(
        candlesticsAndFeaturesWithTarget.columns[0], axis=1
    )
    X_train, y_train = (
        candlesticsAndFeaturesWithTarget.iloc[:, :-1],
        candlesticsAndFeaturesWithTarget.iloc[:, -1],
    )
    return X_train, y_train


def setupAndTrainModel(X_train, y_train):

    xg_reg = xgb.XGBRegressor(
        objective="multi:softmax",
        colsample_bytree=0.3,
        learning_rate=1,
        max_depth=12,
        alpha=5,
        n_estimators=10,
        num_class=3,
    )
    xg_reg.fit(X_train, y_train)

    return xg_reg


def benchSetupAndTrainModel(X_train, y_train, X_test, y_test):
    xg_reg = xgb.XGBRegressor(
        objective="multi:softmax",
        colsample_bytree=0.3,
        learning_rate=1,
        max_depth=12,
        alpha=5,
        n_estimators=10,
        num_class=3,
    )

    xg_reg.fit(
        X_train,
        y_train,
        eval_metric=["error", "logloss"],
        verbose=True,
    )

    y_pred = xg_reg.predict(X_test)
    predictions = y_pred
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("accuracy: %.2f%%" % (accuracy * 100.0))

    ## retrieve performance metrics
#
    kfold = StratifiedKFold(n_splits=10)
    results = cross_val_score(xg_reg, X_test, y_test, cv=kfold)
    print("kfold Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

    return xg_reg


def continueTraining(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def evaluate(xg_reg, X_test, y_test):
    preds = xg_reg.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print("RMSE: %f" % (rmse))


def showInfo(xg_reg):
    xgb.plot_importance(xg_reg)
    plt.rcParams["figure.figsize"] = [5, 5]
    plt.show()
