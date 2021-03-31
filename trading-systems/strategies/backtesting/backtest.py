# %%
import os, sys

sys.path.insert(0, os.path.abspath("../.."))

# %% CONFIG
from strategies import PricePredictor as Strategy

SYMBOL = "LTCBTC"
CANDLESTICKS_INTERVAL = "1h"
trade_start_position = 20000
forward_look_for_target: int = 4  # number of missing targets at the end

# %%
from lib.data_util import load_candlesticks
from lib.backtest import Backtest, setup_file_path
from lib.charting import chartTrades
from binance.client import Client

# %%
# client = Client()

# %%
strategy = Strategy(backtest=True)
tmp_path = "./tmp/" + strategy.__class__.__name__ + "-" + SYMBOL + "-" + CANDLESTICKS_INTERVAL + "/"

path_builder = setup_file_path(tmp_path)

# %%
candlesticks = load_candlesticks(
    SYMBOL, CANDLESTICKS_INTERVAL, custom_data_path="../../tmp", #binance_client=client
)

features = strategy.generate_features(candlesticks)[:-forward_look_for_target]
# targets = strategy._generate_targets(candlesticks, features)
candlesticks = candlesticks[:-forward_look_for_target]
trade_end_position = len(candlesticks)

features.to_csv(path_builder("features"))

# pd.DataFrame(targets).to_csv(path_builder("targets"))
path_builder = None

# %% [markdown]
# ### Data exploration
# Here we will explore the features that are generated
# %% ploting data
plot_features = features[["high", "low", "close"]]
plot_features.index = candlesticks["close time"]
_ = plot_features.plot(subplots=True)

# %% describe data
features.describe().transpose()

# %%
path_builder = setup_file_path(tmp_path)

signals = Backtest.run(
    strategy=strategy,
    features=features,
    candlesticks=candlesticks,
    start_position=trade_start_position,
    end_position=trade_end_position,
    signals_csv_path=path_builder("signals"),
)

# %%
if path_builder is None:
    path_builder = setup_file_path(tmp_path)

# signals = pd.read_csv("./tmp/PricePredictor/2021-03-26-14:26:30-signals.csv", index_col=[0])

trades = Backtest.evaluate(signals, candlesticks, trade_start_position, trade_end_position, 0.001)
trades.to_csv(path_builder("trades"))

chartTrades(
    trades,
    candlesticks,
    trade_start_position,
    trade_end_position,
    path_builder("chart", extension="html"),
    # "./tmp/PricePredictor/2021-03-22-20-20-chart.html"
)
path_builder = None

# %%
trades

# %%
trades.describe()

# %%
