import numpy as np
import pandas as pd

class SensorStream:

    def __init__(self, file_path, sensors):
        # read data and normalize column names
        self.data = pd.read_excel(file_path)
        self.data.columns = self.data.columns.str.lower()

        # keep sensors lowercase as well so slicing works reliably
        self.sensors = [s.lower() for s in sensors]
        self.index = 0

    def next(self):
        if self.index >= len(self.data):
            self.index = 0
        row = self.data.iloc[self.index][self.sensors].values
        self.index += 1
        return row