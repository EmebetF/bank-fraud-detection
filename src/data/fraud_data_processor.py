from src.data.data_loader import DataLoader
from src.data.data_cleaner import DataCleaner
import pandas as pd

class FraudDataProcessor:
    """
    High-level class to orchestrate ingestion, cleaning, and enrichment.
    """
    def __init__(self, raw_data_dir: str = "data/raw", processed_data_dir: str = "data/processed"):
        self.loader = DataLoader(raw_data_dir)
        self.cleaner = DataCleaner(processed_data_dir)
    
    def load_and_clean_fraud_data(self) -> pd.DataFrame:
        df = self.loader.load_csv("fraud_data.csv")
        return self.cleaner.clean_fraud_data(df)

    def load_and_clean_ip_data(self) -> pd.DataFrame:
        df = self.loader.load_csv("ip_country.csv")
        return self.cleaner.clean_ip_country_data(df)
