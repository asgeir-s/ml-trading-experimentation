from strategies.third.third import Third
from lib.data_util import load_candlesticks
from lib.backtest import Backtest
from lib.charting import chartTrades
import pandas as pd

strategy_tmp_path = "strategies/third/tmp"


def main():
    candlesticks = load_candlesticks("BTCUSDT", "1h")

    trade_start_position = 10000
    trade_end_position = len(candlesticks)
    features = Third.generate_features(candlesticks)
    targets = Third._generate_target(features)

    # features.to_csv(strategy_tmp_path + "/features.csv")
    pd.DataFrame({"novice": targets[0], "up_down": targets[1]}).to_csv(
        strategy_tmp_path + "/targets.csv"
    )
    signals = Backtest.run(
        TradingStrategy=Third,
        features=features,
        candlesticks=candlesticks,
        start_position=trade_start_position,
        end_position=trade_end_position,
    )
    # signals = Backtest._runWithTarget(
    #    TradingStrategy=Third,
    #    features=features,
    #    target=targets,
    #    candlesticks=candlesticks,
    #    start_position=trade_start_position,
    #    end_position=trade_end_position,
    # )
    signals.to_csv(strategy_tmp_path + "/signals.csv")

    trades = Backtest.evaluate(
        signals, candlesticks, trade_start_position, trade_end_position, 0.001
    )
    trades.to_csv(strategy_tmp_path + "/trades.csv")

    chartTrades(
        trades,
        candlesticks,
        trade_start_position,
        trade_end_position,
        strategy_tmp_path + "/chart.html",
    )


if __name__ == "__main__":
    main()
