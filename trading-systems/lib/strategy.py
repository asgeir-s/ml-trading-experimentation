import pandas as pd
import abc
from dataclasses import dataclass
from lib import tradingSignal


@dataclass
class Strategy(abc.ABC):
    @abc.abstractmethod
    def execute(self, features: pd.DataFrame) -> tradingSignal:
        pass
