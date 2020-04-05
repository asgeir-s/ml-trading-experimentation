from strategies.first.first import First
from lib.data_loader import load_candlesticks
from lib.backtest import Backtest


def main():
    candlesticks = load_candlesticks("1h")

    trade_start_position = 10000
    trade_end_position = 12000
    features = First.generate_features(candlesticks)

    signals = Backtest.run(First, features, candlesticks, trade_start_position, trade_end_position)
    signals.to_csv("strategies/first/tmp/signals.csv")

    start_money = 1000
    trades = Backtest.evaluate(
        signals, candlesticks, trade_start_position, trade_end_position, start_money
    )
    trades.to_csv("strategies/first/tmp/trades.csv")


if __name__ == "__main__":
    main()
