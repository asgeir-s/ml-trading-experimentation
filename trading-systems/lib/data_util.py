from pandas.core.frame import DataFrame
from pandas.core.series import Series
from lib.model import Model
import pandas as pd
from pathlib import Path
from typing import Union, Dict, Any, Optional
from datetime import datetime


candlesticks: Dict[str, Dict[str, pd.DataFrame]] = {}
trades: Dict[str, Any] = {}
tmp_path = "tmp/"


def read_csv_if_exists(file_path: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        return None


def create_directory_if_not_exists(dir_path: str) -> None:
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def load_candlesticks(
    instrument: str,
    interval: str,
    binance_client: Optional[Any] = None,
    custom_data_path: Optional[str] = None,
    convert_timestamp_to_date: bool = True,
):
    """
    Returns all candlesticks up until NOW and persists it to the csv.
    """
    if custom_data_path is not None:
        tmp_path = custom_data_path
    else:
        tmp_path = "tmp/"

    if candlesticks.get(instrument) is None:
        candlesticks[instrument] = {}

    if candlesticks[instrument].get(interval) is None:
        candlesticks[instrument][interval] = read_csv_if_exists(
            f"{tmp_path}/data/binance/candlestick-{instrument}-{interval}.csv"
        )
        new_csv = candlesticks[instrument].get(interval) is None
        last_candle_close: Union[str, int] = (
            int(candlesticks[instrument][interval].tail(1)["close time"].values[0]) + 1
            if not new_csv
            else "1 Jan, 2017"
        )
        print(f"Closetime of newest candle is {last_candle_close}")
        if binance_client is not None:
            print("Geting new candlesticks from Binance.")
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
                candlesticks[instrument][interval] = candlesticks[instrument][
                    interval
                ].append(new_candles, ignore_index=True)

        else:
            print("Only using data on file. Will not download new data from Binance.")

    if convert_timestamp_to_date:
        candlesticks[instrument][interval]["open time"] = pd.to_datetime(
            candlesticks[instrument][interval]["open time"], unit="ms"
        )
        candlesticks[instrument][interval]["close time"] = pd.to_datetime(
            candlesticks[instrument][interval]["close time"], unit="ms"
        )

    return (
        candlesticks[instrument][interval]
        .drop_duplicates(subset=["open time"])
        .set_index("close time")
    )


def add_candle(instrument: str, interval: str, new_candle: Dict):
    timestemp = int(new_candle["close time"]) / 1000
    time_formated = datetime.fromtimestamp(timestemp, tz=None).strftime("%c")

    print(f"\n[{time_formated}] New candle for {instrument} @ {interval}:")
    print(new_candle)
    candlesticks[instrument][interval] = candlesticks[instrument][interval].append(
        new_candle, ignore_index=True
    )
    new_candle_row = candlesticks[instrument][interval].tail(1)
    new_candle_row.to_csv(
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
        if trades[name] is not None:
            trades[name] = trades[name].astype(
                {
                    "orderId": int,
                    "transactTime": str,
                    "price": float,
                    "signal": str,
                    "origQty": float,
                    "executedQty": float,
                    "cummulativeQuoteQty": float,
                    "timeInForce": str,
                    "commissionAsset": str,
                    "commission": float,
                    "type": str,
                    "side": str,
                    "reason": str,
                    "data": any,
                }
            )

    return trades[name]


def add_trade(
    instrument: str,
    interval: str,
    trading_strategy_instance_name: str,
    new_trade_dict: Dict,
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
                "commissionAsset",
                "commission",
                "type",
                "side",
                "reason",
                "data",
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
