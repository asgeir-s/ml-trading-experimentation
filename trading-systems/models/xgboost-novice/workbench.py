from model import XgboostNovice
import pandas as pd
from lib.data_splitter import generteTrainAndTestSet


def loadData() -> pd.DataFrame:
    df = pd.read_csv("../data-loading/binance/data/candlestick-BTCUSDT-1h.csv").drop(
        columns=["ignore"]
    )
    return df


df = loadData()

xg_boost = XgboostNovice()

features = xg_boost.generateFeatures(df)
print(features)

target = xg_boost.generateTarget(features)

print(target)


trainingSetFeatures, trainingSetTarget, testSetFeatures, testSetTarget = generteTrainAndTestSet(
    features, target, 20, ["close time", "open time"]
)

xg_boost.train(trainingSetFeatures, trainingSetTarget)
xg_boost.evaluate(testSetFeatures, testSetTarget)
