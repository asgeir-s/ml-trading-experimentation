import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Dict, Any, Optional

candlesticks = {}
trades = {}
tmp_path = "tmp/"


def read_csv_if_exists(file_path: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(file_path)

    except FileNotFoundError:
        return None


def create_directory_if_not_exists(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_candlesticks(instrument: str, interval: str, binance_client: Any):
    """
    Returns all candlesticks up until NOW and persists it to the csv.
    """
    if candlesticks.get(instrument) is None:
        candlesticks[instrument] = {}

    if candlesticks[instrument].get(interval) is None:
        candlesticks[instrument][interval] = read_csv_if_exists(
            f"{tmp_path}/data/binance/candlestick-{instrument}-{interval}.csv"
        )
        new_csv = candlesticks[instrument].get(interval) is None
        last_candle_close: Union[str, int] = (
            int(candlesticks[instrument][interval].tail(1)["close time"].values[0])
            if not new_csv
            else "1 Jan, 2017"
        )
        print(f"Closetime of newest candle is {last_candle_close}")
        new_raw_candles_raw = binance_client.get_historical_klines(
            instrument, interval, last_candle_close
        )

        new_candles = pd.DataFrame.from_records(
            new_raw_candles_raw,
            columns=[
                "open time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close time",
                "quote asset volume",
                "number of trades",
                "taker buy base asset volume",
                "taker buy quote asset volume",
                "ignore",
            ],
        ).drop(columns=["ignore"])
        new_candles = new_candles.astype(
            {
                "open time": int,
                "open": float,
                "high": float,
                "low": float,
                "close": float,
                "volume": float,
                "close time": int,
                "quote asset volume": float,
                "number of trades": int,
                "taker buy base asset volume": float,
                "taker buy quote asset volume": float,
            }
        )

        create_directory_if_not_exists(f"{tmp_path}/data/binance/")

        new_candles.to_csv(
            f"{tmp_path}/data/binance/candlestick-{instrument}-{interval}.csv",
            mode="w" if new_csv else "a",
            header=True if new_csv else False,
            index=False,
        )
        if candlesticks[instrument][interval] is None:
            candlesticks[instrument][interval] = new_candles
        else:
            candlesticks[instrument][interval] = candlesticks[instrument][interval].append(
                new_candles, ignore_index=True
            )

    return candlesticks[instrument][interval]


def add_candle(instrument: str, interval: str, new_candle_dict: Dict):
    candlesticks[instrument][interval] = candlesticks[instrument][interval].append(
        new_candle_dict, ignore_index=True
    )
    new_candle = candlesticks[instrument][interval].tail(0)
    new_candle.to_csv(
        f"{tmp_path}/data/binance/candlestick-{instrument}-{interval}.csv",
        mode="a",
        header=False,
        index=False,
    )
    return candlesticks[instrument][interval]


def load_trades(instrument: str, interval: str, trading_strategy_instance_name: str):
    """
    Returns all trades up until NOW from csv.
    """
    name = f"{trading_strategy_instance_name}-{instrument}-{interval}"
    if trades.get(name) is None:
        trades[name] = read_csv_if_exists(f"{tmp_path}/trades/" + name + ".csv")

    return trades[name]


def add_trade(
    instrument: str, interval: str, trading_strategy_instance_name: str, new_trade_dict: Dict
):
    name = f"{trading_strategy_instance_name}-{instrument}-{interval}"
    new_csv = trades[name] is None
    if new_csv:
        print("Adding first trade to the csv.")
        trades[name] = pd.DataFrame(
            columns=[
                "orderId",
                "transactTime",
                "price",
                "signal",
                "origQty",
                "executedQty",
                "cummulativeQuoteQty",
                "timeInForce",
                "type",
                "side",
            ],
        )
    trades[name] = trades[name].append(new_trade_dict, ignore_index=True)

    create_directory_if_not_exists(f"{tmp_path}/trades")
    trades[name].tail(1).to_csv(
        f"{tmp_path}/trades/" + name + ".csv",
        mode="w" if new_csv else "a",
        header=True if new_csv else False,
        index=False,
    )

    return trades[name]