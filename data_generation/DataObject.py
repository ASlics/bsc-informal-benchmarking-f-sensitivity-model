import pandas as pd


class DataObject:
    def __init__(self, data: pd.DataFrame):
        self.data = data
