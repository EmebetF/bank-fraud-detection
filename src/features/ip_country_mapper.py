import pandas as pd
import bisect
from pathlib import Path

class IPCountryMapper:
    """
    Expert-level IP-to-country mapping utility.
    Converts numeric IP addresses to country using bisect search.
    """

    def __init__(self, ip_country_df: pd.DataFrame, processed_data_dir: str = "data/processed"):
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

        # Precompute lists for fast bisect lookup
        self.lower_bounds = ip_country_df["lower_bound_ip_address"].tolist()
        self.upper_bounds = ip_country_df["upper_bound_ip_address"].tolist()
        self.countries = ip_country_df["country"].tolist()

        self.logger.info(f"Loaded {len(self.countries)} IP ranges for country mapping.")

    def ip_to_country(self, ip: float) -> str:
        """
        Map single numeric IP to country using bisect search.
        """
        idx = bisect.bisect_right(self.lower_bounds, ip) - 1
        if idx >= 0 and self.lower_bounds[idx] <= ip <= self.upper_bounds[idx]:
            return self.countries[idx]
        return "Unknown"

    def map_ip_to_country(self, fraud_df: pd.DataFrame) -> pd.DataFrame:
        """
        Map all IP addresses in the fraud dataset to country.
        Saves processed dataset to disk.
        """
        self.logger.info("Mapping IP addresses to countries...")
        fraud_df["country"] = fraud_df["ip_address"].apply(self.ip_to_country)

        out_path = self.processed_data_dir / "fraud_with_country.csv"
        fraud_df.to_csv(out_path, index=False)
        self.logger.info(f"Saved mapped dataset to {out_path}")
        return fraud_df