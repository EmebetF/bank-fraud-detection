import pandas as pd
from pathlib import Path

class DataCleaner:
    """
    Clean datasets: handle missing values, type casting, remove duplicates.
    """
    def __init__(self, processed_data_dir: str = "data/processed"):
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

    def clean_fraud_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signup_time"] = pd.to_datetime(df["signup_time"], errors="coerce")
        df["purchase_time"] = pd.to_datetime(df["purchase_time"], errors="coerce")
        df["purchase_value"] = pd.to_numeric(df["purchase_value"], errors="coerce")
        df = df.dropna(subset=["user_id", "signup_time", "purchase_time", "ip_address", "class"])
        df["user_id"] = df["user_id"].astype(int)
        df["age"] = df["age"].fillna(-1).astype(int)
        df["class"] = df["class"].astype(int)
        out_path = self.processed_data_dir / "clean_fraud_data.csv"
        df.to_csv(out_path, index=False)
        return df

    def clean_ip_country_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df["lower_bound_ip_address"] = pd.to_numeric(df["lower_bound_ip_address"], errors="coerce")
        df["upper_bound_ip_address"] = pd.to_numeric(df["upper_bound_ip_address"], errors="coerce")
        df = df.dropna(subset=["lower_bound_ip_address", "upper_bound_ip_address", "country"])
        df = df.sort_values(by=["lower_bound_ip_address"]).reset_index(drop=True)
        out_path = self.processed_data_dir / "clean_ip_country.csv"
        df.to_csv(out_path, index=False)
        return df
