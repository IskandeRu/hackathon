import pandas as pd
from abc import ABC, abstractmethod

from typing import Optional


class Strategy(ABC):

    @abstractmethod
    def required_rows(self):
        raise NotImplementedError("Specify required_rows!")

    @abstractmethod
    def compute_target_position(self, current_data: pd.DataFrame, current_position: float) -> Optional[float]:
        assert len(current_data) == self.required_rows  # This much data will be fed to model

        return None  # If None is returned, no action is executed


class YourStrategy(Strategy):
    required_rows = 2*24*60   # minutes of data to be fed to model.

    def compute_target_position(self, current_data: pd.DataFrame, current_position: float) -> Optional[float]:
        avg_price = current_data['price'].mean()
        current_price = current_data['price'][-1]

        target_position = current_position + (avg_price - current_price)/1000

        return target_position

