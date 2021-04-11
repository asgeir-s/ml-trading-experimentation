from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from lib.tradingSignal import TradingSignal


@dataclass
class Stop_loss_Take_profit:
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class Position:
    signal: Optional[TradingSignal] = None
    stop_loss_take_profit: Optional[Stop_loss_Take_profit] = None
    reason: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
