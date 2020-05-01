from lib.data_util import load_candlesticks
from lib.backtest import Backtest, setup_file_path
from lib.charting import chartTrades
import pandas as pd
from strategies import UpDownDoubleSpiral as Strategy


def main():
    strategy = Strategy()

    tmp_path = "./tmp/" + strategy.__class__.__name__ + "/"
    candlesticks = load_candlesticks("BTCUSDT", "1h")

    trade_start_position = 10000
    trade_end_position = len(candlesticks)

    features = strategy.generate_features(candlesticks)

    for stop_loss_treshold in range(7, 20):
        for fast_treshold in range(2, 20):  # burde begynne på 1, men vi har vært igjennom den
            for slow_treshold in range(fast_treshold, 23):
                path_builder = setup_file_path(tmp_path)
                parameters = pd.DataFrame(
                    columns=["stop_loss_treshold", "slow_treshold", "fast_treshold",]
                )
                computed_stop_loss_treshold = 1.0 - (stop_loss_treshold / 1000.0)
                computed_slow_treshold = 1.0 + (slow_treshold / 1000.0)
                computed_fast_treshold = 1.0 + (fast_treshold / 1000.0)

                parameters = parameters.append(
                    {
                        "stop_loss_treshold": computed_stop_loss_treshold,
                        "slow_treshold": computed_slow_treshold,
                        "fast_treshold": computed_fast_treshold,
                    },
                    ignore_index=True,
                )

                parameters.to_csv(path_builder("parameters"))

                strategy = Strategy(
                    stop_loss_treshold=computed_stop_loss_treshold,
                    slow_treshold=computed_slow_treshold,
                    fast_treshold=computed_fast_treshold,
                )
                signals = Backtest.run(
                    strategy=strategy,
                    features=features,
                    candlesticks=candlesticks,
                    start_position=trade_start_position,
                    end_position=trade_end_position,
                    signals_csv_path=path_builder("signals"),
                )
                if len(signals) >= 2:
                    trades = Backtest.evaluate(
                        signals, candlesticks, trade_start_position, trade_end_position, 0.000
                    )

                    trades.to_csv(path_builder("trades"))

                    chartTrades(
                        trades,
                        candlesticks,
                        trade_start_position,
                        trade_end_position,
                        path_builder("chart", extension="html"),
                    )


if __name__ == "__main__":
    main()
