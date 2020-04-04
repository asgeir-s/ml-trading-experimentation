from strategies.first import First
from lib.data_loader import load_candlesticks

def main():
    candlesticks = load_candlesticks("1h")

    init_set = candlesticks.iloc[:10000]
    test_set = candlesticks.iloc[10000:15000]

    firstTradingSystem = First(candlesticks)

    signal = firstTradingSystem.execute(test_set)

    print(f"signal is: {signal}")


if __name__ == '__main__':
    main()