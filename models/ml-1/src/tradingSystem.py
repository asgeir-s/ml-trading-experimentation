import pandas as pd
import numpy as np
from model import (
    generteTrainAndTestSet,
    setupAndTrainModel,
    evaluate,
    showInfo,
    continueTraining,
    generateTrainSet,
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


def fitNewData(model, newData: pd.DataFrame):
    X_train, y_train = generateTrainSet(newData)
    return continueTraining(model, X_train, y_train)


def generateSignal(model, period: pd.DataFrame) -> pd.DataFrame:
    """
    The input periode should be data up to the and including the row to be predicted.
    """
    newRow = period.tail(1).copy()

    newRow = newRow.drop(columns=["open time", "close time"])
    newRow = newRow.drop(newRow.columns[0], axis=1)  # drop index
    newRow["pred"] = model.predict(newRow)

    upTreshold = 0.3
    downTreshold = -0.1
    newRow["signal"] = "none"

    if newRow["pred"].values[0] > upTreshold:
        newRow["signal"] = "BUY"
    elif newRow["pred"].values[0] < downTreshold:
        newRow["signal"] = "SELL"
    newRow["open time"] = period.tail(1)["open time"]
    newRow["close time"] = period.tail(1)["close time"]

    return newRow
