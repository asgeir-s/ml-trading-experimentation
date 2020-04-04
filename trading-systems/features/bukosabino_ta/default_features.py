import pandas as pd
import ta

def createFeatures(df: pd.DataFrame) -> pd.DataFrame:
    return ta.add_all_ta_features(
        df,
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
        fillna=True,
    )