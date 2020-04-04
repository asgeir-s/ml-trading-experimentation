import pandas as pd
from tradingSystem import benchPrepearModel, benchGenerateSignal

allCandlesticsAndFeaturesWithTarget: pd.DataFrame = pd.read_csv(
    "./data/candlesticsAndFeaturesWithTarget.csv"
)

startPosition = 20000
# take first 1000 into training set
firstPart = allCandlesticsAndFeaturesWithTarget.iloc[0:startPosition]

# train the modell
model = benchPrepearModel(firstPart, 20)