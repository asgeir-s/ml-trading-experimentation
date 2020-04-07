from strategies.first.first import First
from lib.data_loader import load_candlesticks
from lib.backtest import Backtest
from lib.charting import chartTrades


def main():
    candlesticks = load_candlesticks("1h")

    trade_start_position = 10000
    trade_end_position = len(candlesticks)
    features = First.generate_features(candlesticks)
    targets = First._generate_target(features)

#    signals = Backtest.run(
#        TradingStrategy=First,
#        features=features,
#        candlesticks=candlesticks,
#        shorter_candlesticks=None,
#        start_position=trade_start_position,
#        end_position=trade_end_position,
#    )
    signals = Backtest._runWithTarget(
        TradingStrategy=First,
        features=features,
        target=targets,
        candlesticks=candlesticks,
        shorter_candlesticks=None,
        start_position=trade_start_position,
        end_position=trade_end_position,
    )
    signals.to_csv("strategies/first/tmp/signals.csv")

    trades = Backtest.evaluate(signals, candlesticks, trade_start_position, trade_end_position)
    trades.to_csv("strategies/first/tmp/trades.csv")

    chartTrades(
        trades,
        candlesticks,
        trade_start_position,
        trade_end_position,
        "strategies/first/tmp/chart.html",
    )


if __name__ == "__main__":
    main()
