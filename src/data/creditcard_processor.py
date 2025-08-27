# src/data/creditcard_processor.py
import pandas as pd
import numpy as np

class CreditCardProcessor:
    REQUIRED = {'Time','Amount','Class', *{f'V{i}' for i in range(1,29)}}

    def __init__(self, path: str):
        self.path = path
        self.df = None

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.path)
        missing = self.REQUIRED - set(df.columns)
        if missing:
            raise ValueError(f"creditcard schema mismatch. Missing cols: {missing}")
        self.df = df
        return self.df

    def basic_preprocess(self) -> pd.DataFrame:
        df = self.df.copy().sort_values('Time').reset_index(drop=True)
        df['log_amount'] = (df['Amount'].fillna(0) + 1).apply(np.log)
        df['hour_of_day'] = (df['Time'] // 3600) % 24
        self.df = df
        return self.df

    def save_processed(self, path:str):
        self.df.to_parquet(path, index=False)