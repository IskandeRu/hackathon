import pandas as pd
from abc import ABC, abstractmethod
from statsmodels.tsa.arima.model import ARIMA
from typing import Optional
class Strategy(ABC):

    @abstractmethod
    def required_rows(self):
        raise NotImplementedError("Specify required_rows!")

    @abstractmethod
    def compute_target_position(self, current_data: pd.DataFrame, current_position: float) -> Optional[float]:
        assert len(current_data) == self.required_rows  # This much data will be fed to model

        return None  # If None is returned, no action is executed


class MeanReversionStrategy(Strategy):
    required_rows = 2*24*60   # minutes of data to be fed to model.

    def compute_target_position(self, current_data: pd.DataFrame, current_position: float) -> Optional[float]:
        avg_price = current_data['price'].mean()
        current_price = current_data['price'][-1]

        target_position = current_position + (avg_price - current_price)/1000

        return target_position


class YourStrategy(Strategy):
    required_rows = 2000000  # Specify how many minutes of data are required for live prediction

    open_positions = []

    def __init__(self):
        training_data = pd.read_pickle("data/train_data.pickle")
        self.ARIMAmodel = ARIMA(training_data['price'][-500000:], order=(2, 2, 1))
        self.ARIMAmodel = self.ARIMAmodel.fit()

    def compute_target_position(self, current_data: pd.DataFrame, current_position: float) -> Optional[float]:

        y_pred = self.ARIMAmodel.get_forecast(5, exog=current_data['price'])

        current_price = current_data['price'][-1]

        tmp = y_pred.predicted_mean.to_numpy()
        first, last = tmp[0], tmp[-1]
        #min, max = tmp.min, tmp.max

        pos = 0
        if current_price > first:
            pos = pos+2
        if current_price > last:
            pos = pos+1
        if current_price < last:
            pos = pos-5

        for open_pos in self.open_positions:
            pos_price = open_pos['price']
            pos_vol = open_pos['volume']

            if pos_price > current_price:
                if pos_vol > 0:
                    if first > current_price:
                        pos = 7
                else:
                    if first < current_price:
                        pos = pos_vol
                        self.open_positions.remove(open_pos)

        self.open_positions.append({'price': current_price, 'volume': pos})
        return pos