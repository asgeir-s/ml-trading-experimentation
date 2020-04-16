import numpy as np
import pandas as pd
from bokeh.layouts import gridplot
from bokeh.plotting import figure, output_file, show


def chartTrades(
    trades: pd.DataFrame, candlesticks: pd.DataFrame, start_position: int, end_position: int, output_file_path: str
):
    candlesticks_periode = candlesticks.iloc[start_position:end_position]
    p1 = figure(x_axis_type="datetime", title="Trading Strategy")
    p1.grid.grid_line_alpha = 0.3
    p1.xaxis.axis_label = "Date"
    p1.yaxis.axis_label = "Value"

    holding = pd.DataFrame(columns=["transactTime", "money"])
    first_candlestick = candlesticks_periode.head(1)
    holding = holding.append(
        {
            "transactTime": first_candlestick["open time"].values[0],
            "money": first_candlestick["open"].values[0],
        },
        ignore_index=True,
    )

    ordered = trades[["close time", "close money"]].rename(
        columns={"close time": "transactTime", "close money": "money"}
    )

    ordered = ordered.append(
        trades[["open time", "open money"]].rename(
            columns={"open time": "transactTime", "open money": "money"}
        ),
        ignore_index=True,
    )

    ordered = ordered.sort_values("transactTime")

    holding = holding.append(ordered, ignore_index=True,)

    last_candlestick = candlesticks_periode.tail(1)
    holding = holding.append(
        {
            "transactTime": last_candlestick["open time"].values[0],
            "money": trades["close money"].tail(1).values[0],
        },
        ignore_index=True,
    )

    p1.line((holding["transactTime"]), holding["money"], color="#A6CEE3", legend_label="Trading Strategy")
    p1.line(
        (candlesticks_periode["close time"]),
        candlesticks_periode["close"],
        color="#B2DF8A",
        legend_label="Asset",
    )

    p1.circle((trades["open time"]), trades["open money"], color="#32CD32", legend_label="BUY")
    p1.circle((trades["close time"]), trades["close money"], color="#FF0000", legend_label="SELL")

    p1.legend.location = "top_left"

    output_file(output_file_path, title="Trading Strategy")

    show(gridplot([[p1]], plot_width=1000, plot_height=800))  # open a browser


def datetime(x):
    return np.array(x, dtype=np.datetime64)
