import pandas as pd
import ta

def loadData():
    df = pd.read_csv("../../data-loading/binance/data/candlestick-BTCUSDT-1h.csv").drop(
        columns=["ignore"]
    )
    size = df.size
    print(f"{size}")
    print(df.describe())

    # Add ta features filling NaN values
    df = ta.add_all_ta_features(
        df, open="open", high="high", low="low", close="close", volume="volume", fillna=True)
    
    print(df)


loadData()
