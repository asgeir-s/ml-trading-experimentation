import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def generteTrainAndTestSet(candlesticsAndFeaturesWithTarget: pd.DataFrame):

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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    return X_train, X_test, y_train, y_test

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
        objective="reg:squarederror",
        colsample_bytree=0.3,
        learning_rate=0.8,
        max_depth=20,
        alpha=5,
        n_estimators=10,
    )

    xg_reg.fit(X_train, y_train)

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
