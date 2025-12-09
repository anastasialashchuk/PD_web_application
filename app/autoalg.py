# autoalg.py
import numpy as np
from scipy.signal import argrelextrema

class SignalProcessing:
    def __init__(self, frac=0.1, order_min=5, order_max=5):
        self.frac = frac
        self.order_min = order_min
        self.order_max = order_max

    def get_point(self, values, frame):
        values = np.array(values)
        frame = np.array(frame)

        # Находим локальные максимумы
        max_indices = argrelextrema(values, np.greater, order=self.order_max)[0]
        maxP = frame[max_indices]
        maxA = values[max_indices]

        # Находим локальные минимумы
        min_indices = argrelextrema(values, np.less, order=self.order_min)[0]
        minP = frame[min_indices]
        minA = values[min_indices]

        return maxP, minP, maxA, minA, self.frac, self.order_min, self.order_max