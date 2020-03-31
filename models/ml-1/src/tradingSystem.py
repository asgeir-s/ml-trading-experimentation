import pandas as pd
import numpy as np
from model import generteTrainAndTestSet, setupAndTrainModel, evaluate, showInfo

candlesticsAndFeaturesWithTarget = pd.read_csv(
    "./data/candlesticsAndFeaturesWithTarget.csv"
)

myTestSize = 1000
myTestSet = candlesticsAndFeaturesWithTarget.tail(myTestSize)
print(myTestSet)
candlesticsAndFeaturesWithTarget = candlesticsAndFeaturesWithTarget.drop(
    candlesticsAndFeaturesWithTarget.tail(myTestSize).index
)

candlesticsAndFeatures = (candlesticsAndFeaturesWithTarget.iloc[:, :-1],)
X_train, X_test, y_train, y_test = generteTrainAndTestSet(
    candlesticsAndFeaturesWithTarget
)
xg_reg = setupAndTrainModel(X_train, y_train)
# evaluate(xg_reg, X_test, y_test)
# showInfo(xg_reg)

result = pd.DataFrame(myTestSet.copy())

myTestSet = myTestSet.drop(columns=["open time", "close time"])
myTestSet = myTestSet.drop(myTestSet.columns[1], axis=1).iloc[:, :-1]

result["pred"] = xg_reg.predict(myTestSet)
upTreshold = 0.6
downTreshold = -0.2
conditions = [result["pred"] > upTreshold, result["pred"] < downTreshold]
choices = ["BUY", "SELL"]
result["trade"] = np.select(conditions, choices, default="none")
result["trade price"] = result.shift(periods=-1)["open"]

startCash = 10000
cash = startCash
holding = 0
status = "NONE"
numberOfTrades = 0
startTime = result["open time"].head(1)
startTime = pd.to_datetime(startTime, unit="ms").values[0]
endTime = result["close time"].tail(1)
endTime = pd.to_datetime(endTime, unit="ms").values[0]
startPrice = result["open"].head(1).values[0]
endPrice = result["close"].tail(1).values[0]
percentagePriceChange = (100.0 / startPrice) * (endPrice - startPrice)

print(f"Starts with {cash}$ at {startTime}")
for index, row in result.iterrows():
    signal = row["trade"]
    tradePrice = row["trade price"]

    if status == "NONE" and signal == "BUY":
        holding = cash / tradePrice
        cash = 0
        status = "LONG"
        numberOfTrades = numberOfTrades + 1
    elif status == "LONG" and signal == "SELL":
        cash = holding * tradePrice
        holding = 0
        status = "NONE"
        numberOfTrades = numberOfTrades + 1

if status == "LONG":
    cash = holding * tradePrice
    holding = 0
    status = "NONE"
    numberOfTrades = numberOfTrades + 1

print(f"Ends with {cash}$ (number of trades: {numberOfTrades}) at {endTime}")
print(f"Earned {cash-startCash}$ ({round((100.0/startCash)*(cash-startCash),2)}%)")
print(f"Percentage price change in period: {round(percentagePriceChange,2)}%")

result.to_csv("data/result.csv")
