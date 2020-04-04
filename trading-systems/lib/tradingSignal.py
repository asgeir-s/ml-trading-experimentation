from enum import Enum


class TradingSignal(Enum):
    BUY = 1
    HOLD = 0
    SELL = -1
