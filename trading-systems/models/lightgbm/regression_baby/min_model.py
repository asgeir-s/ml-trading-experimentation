from .max_model import RegressionBabyMaxModel
import pandas as pd
from dataclasses import dataclass
from targets.regression import max_over_periods


@dataclass  # type: ignore
class RegressionBabyMinModel(RegressionBabyMaxModel):

    def generate_target(self, candlesticks: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        return max_over_periods.generate_target(
            candlesticks, column="low", periodes=6, percentage=True, min=True
        )

    def __hash__(self) -> int:
        return hash(self.__class__.__name__) + hash(self.model)
