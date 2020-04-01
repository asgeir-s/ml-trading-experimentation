import pandas as pd
from tradingSystem import prepearModel, generateSignals


def createHistoryAndTestPeriodes(
    testSetSize: int, data: pd.DataFrame
) -> (pd.DataFrame, pd.DataFrame):
    """
    Devides the history data into a testperiod and history.
    """
    testPeriode = data.tail(testSetSize)
    history = data.drop(data.tail(testSetSize).index)
    return testPeriode, history


def simulateTrades(signals: pd.DataFrame, startCash: int):
    trades = pd.DataFrame(
        columns=[
            "open time",
            "close time",
            "open price",
            "close price",
            "change %",
            "open cash",
            "close cash",
        ]
    )
    cash = startCash
    holding = 0
    status = "NONE"
    openTime = 0
    closeTime = 0
    openPrice = 0
    closePrice = 0
    openCash = 0
    itemnumber = 0

    for index, row in signals.iterrows():
        itemnumber += 1
        signal = row["trade"]
        tradePrice = row["trade price"]

        if status == "NONE" and signal == "BUY":
            openTime = row["open time"]
            openPrice = tradePrice
            openCash = cash
            holding = cash / tradePrice
            cash = 0
            status = "LONG"
        elif (status == "LONG" and signal == "SELL") or (
            status == "LONG" and itemnumber == len(signals)
        ):
            closeTime = row["close time"]
            closePrice = tradePrice
            cash = holding * tradePrice
            trades = trades.append(
                {
                    "open time": openTime,
                    "close time": closeTime,
                    "open price": openPrice,
                    "close price": closePrice,
                    "change %": (100.0 / openPrice) * (closePrice - openPrice),
                    "open cash": openCash,
                    "close cash": cash,
                },
                ignore_index=True,
            )
            openTime = 0
            closeTime = 0
            openPrice = 0
            closePrice = 0
            holding = 0
            openCash = 0
            status = "NONE"

    trades["open time"] = pd.to_datetime(trades["open time"], unit="ms")
    trades["close time"] = pd.to_datetime(trades["close time"], unit="ms")

    return trades


def printResults(trades: pd.DataFrame):
    startTime = trades["open time"].head(1).values[0]
    endTime = trades["close time"].tail(1).values[0]
    startPrice = trades["open price"].head(1).values[0]
    endPrice = trades["close price"].tail(1).values[0]
    percentagePriceChange = (100.0 / startPrice) * (endPrice - startPrice)
    startCash = trades["open cash"].head(1).values[0]
    endCash = trades["close cash"].tail(1).values[0]
    print(f"Starts with {startCash}$ at {startTime}")
    print(f"Ends with {endCash}$ (number of trades: {len(trades)}) at {endTime}")
    print(
        f"Earned {endCash-startCash}$ ({round((100.0/startCash)*(endCash-startCash),2)}%)"
    )
    print(f"Percentage price change in period: {round(percentagePriceChange,2)}%")


allCandlesticsAndFeaturesWithTarget = pd.read_csv(
    "./data/candlesticsAndFeaturesWithTarget.csv"
)

testPeriode, history = createHistoryAndTestPeriodes(
    2000, allCandlesticsAndFeaturesWithTarget
)

model = prepearModel(history)

tradingSignals = generateSignals(model, testPeriode)
tradingSignals.to_csv("data/tradingSignals.csv")

# remove last row as it does not have a "trade price"
tradingSignals = tradingSignals.drop(tradingSignals.tail(1).index)

trades = simulateTrades(tradingSignals, 1000)

trades.to_csv("data/trades.csv")

printResults(trades)
