import pandas as pd
import ta


def compute(candlesticks: pd.DataFrame, features_already_computed: pd.DataFrame) -> pd.DataFrame:
    if "trend_visual_ichimoku_a" in features_already_computed.columns:
        return features_already_computed
    else:
        return pd.concat(
            [
                features_already_computed,
                ta.add_all_ta_features(
                    candlesticks,
                    open="open",
                    high="high",
                    low="low",
                    close="close",
                    volume="volume",
                    fillna=True,
                ),
            ],
            axis=1,
        )
