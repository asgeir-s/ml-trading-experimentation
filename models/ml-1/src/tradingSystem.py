import pandas as pd
import numpy as np
from model import (
    generteTrainAndTestSet,
    setupAndTrainModel,
    evaluate,
    showInfo,
    continueTraining,
    generateTrainSet,
    benchSetupAndTrainModel,
)


def prepearModel(candlesticsAndFeaturesWithTargetHistory: pd.DataFrame):
    # X_train, X_test, y_train, y_test = generteTrainAndTestSet(
    #    candlesticsAndFeaturesWithTargetHistory
    # )
    X_train, y_train = generateTrainSet(candlesticsAndFeaturesWithTargetHistory)
    xg_reg = setupAndTrainModel(X_train, y_train)
    # evaluate(xg_reg, X_test, y_test)
    # showInfo(xg_reg)
    return xg_reg


def benchPrepearModel(
    candlesticsAndFeaturesWithTargetHistory: pd.DataFrame, percentageTestSet: int
):
    X_train, y_train, X_test, y_test = generteTrainAndTestSet(
        candlesticsAndFeaturesWithTargetHistory, percentageTestSet
    )
    xg_reg = benchSetupAndTrainModel(X_train, y_train, X_test, y_test)
    evaluate(xg_reg, X_test, y_test)
    showInfo(xg_reg)
    return xg_reg


def fitNewData(model, newData: pd.DataFrame):
    X_train, y_train = generateTrainSet(newData)
    return continueTraining(model, X_train, y_train)


def generateSignal(model, period: pd.DataFrame) -> pd.DataFrame:
    """
    The input periode should be data up to the and including the row to be predicted.
    """
    newRow: pd.DataFrame = period.tail(1).copy()

    newRow = newRow.drop(columns=["open time", "close time"])
    newRow = newRow.drop(newRow.columns[0], axis=1)  # drop index
    newRow["pred"] = model.predict(newRow)

    newRow["signal"] = "none"

    if newRow["pred"].values[0] == 2:
        newRow["signal"] = "BUY"
    elif newRow["pred"].values[0] == 0:
        newRow["signal"] = "SELL"
    newRow["open time"] = period.tail(1)["open time"]
    newRow["close time"] = period.tail(1)["close time"]

    return newRow


def benchGenerateSignal(model, period: pd.DataFrame) -> pd.DataFrame:
    """
    The input periode should be data up to the and including the row to be predicted.
    """
    periode = periode.drop(columns=["open time", "close time"])
    periode = periode.drop(periode.columns[0], axis=1)  # drop index
    periode["pred"] = model.predict(periode)

    upTreshold = 0.003
    downTreshold = -0.002
    periode["signal"] = "none"

    conditions = [period["pred"] > upTreshold, period["pred"] < downTreshold]
    choices = ["BUY", "SELL"]
    period["trade"] = np.select(conditions, choices, default="none")
    period["trade price"] = period.shift(periods=-1)["open"]

    period["open time"] = period.tail(1)["open time"]
    period["close time"] = period.tail(1)["close time"]

    return period
