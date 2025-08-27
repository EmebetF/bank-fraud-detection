import pandas as pd
from pathlib import Path

class DataLoader:
    """
    Load raw CSV datasets.
    """
    def __init__(self, raw_data_dir: str = "data/raw"):
        self.raw_data_dir = Path(raw_data_dir)
    def load_csv(self, filename: str) -> pd.DataFrame:
        path = self.raw_data_dir / filename
        return pd.read_csv(path)
