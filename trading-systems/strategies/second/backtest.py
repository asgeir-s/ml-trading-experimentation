from strategies.second.second import Second
from lib.data_util import load_candlesticks
from lib.backtest import Backtest, set_up_strategy_tmp_path
from lib.charting import chartTrades
import pathlib
import pandas as pd

strategy_tmp_path = set_up_strategy_tmp_path(
    strategy_dir=str(pathlib.Path(__file__).parent.absolute())
)


def main():
    candlesticks = load_candlesticks("BTCUSDT", "1h")

    trade_start_position = 18000
    trade_end_position = len(candlesticks)

    strategy = Second()

    features = strategy.generate_features(candlesticks)
    targets = strategy._generate_targets(candlesticks, features)

    features.to_csv(strategy_tmp_path + "features.csv")

    pd.DataFrame(targets).to_csv(strategy_tmp_path + "targets.csv")

    signals = Backtest.run(
        strategy=strategy,
        features=features,
        candlesticks=candlesticks,
        start_position=trade_start_position,
        end_position=trade_end_position,
        signals_csv_path=strategy_tmp_path + "/signals.csv"
    )
    # signals = Backtest._runWithTarget(
    #     strategy=strategy,
    #     features=features,
    #     targets=targets,
    #     candlesticks=candlesticks,
    #     start_position=trade_start_position,
    #     end_position=trade_end_position,
    #     signals_csv_path=strategy_tmp_path + "signals.csv",
    # )

    trades = Backtest.evaluate(
        signals, candlesticks, trade_start_position, trade_end_position, 0.001
    )
    trades.to_csv(strategy_tmp_path + "trades.csv")

    chartTrades(
        trades,
        candlesticks,
        trade_start_position,
        trade_end_position,
        strategy_tmp_path + "chart.html",
    )


if __name__ == "__main__":
    main()
