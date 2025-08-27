# src/features/feature_engineer.py
import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def time_features(self) -> pd.DataFrame:
        df = self.df
        df['signup_time'] = pd.to_datetime(df['signup_time'], utc=True)
        df['purchase_time'] = pd.to_datetime(df['purchase_time'], utc=True)
        df['time_since_signup_s'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds().clip(lower=0)
        df['purchase_hour'] = df['purchase_time'].dt.hour.astype('int8')
        df['purchase_dayofweek'] = df['purchase_time'].dt.dayofweek.astype('int8')
        df['is_weekend'] = df['purchase_dayofweek'].isin([5,6]).astype('int8')
        self.df = df
        return self.df

    def cumulative_counts(self, key_col: str, prefix: str) -> pd.DataFrame:
        df = self.df.sort_values('purchase_time')
        grp = df.groupby(key_col, sort=False)
        # number of previous transactions by this key
        df[f'{prefix}_tx_count_before'] = grp.cumcount().astype('int32')
        # seconds since previous transaction for key
        prev_time = grp['purchase_time'].shift(1)
        df[f'{prefix}_secs_since_prev'] = ((df['purchase_time'] - prev_time).dt.total_seconds()
                                           .fillna(9.99e8).clip(0, 60*60*24*365).astype('float32'))
        self.df = df
        return self.df

    def freq_encoding(self, cols):
        df = self.df
        for c in cols:
            vc = df[c].value_counts()
            df[f'{c}_freq'] = df[c].map(vc).fillna(0).astype('int32')
        self.df = df
        return self.df

    def add_risk_flags(self):
        df = self.df
        df['flag_instant_purchase'] = (df['time_since_signup_s'] < 60).astype('int8')
        # unusual hour flag (e.g., midnight-5am)
        df['flag_odd_hour'] = df['purchase_hour'].isin([0,1,2,3,4,5]).astype('int8')
        self.df = df
        return self.df

    def winsorize_purchase_value(self, clip_q=0.995):
        df = self.df
        upper = df['purchase_value'].quantile(clip_q)
        df['purchase_value_raw'] = df['purchase_value']
        df['purchase_value'] = df['purchase_value'].clip(0, upper)
        self.df = df
        return self.df

    def run_all(self):
        self.time_features()
        self.cumulative_counts('user_id', 'user')
        self.cumulative_counts('device_id', 'device')
        self.cumulative_counts('ip_address', 'ip')
        self.freq_encoding(['device_id','ip_address','country','browser','source'])
        self.add_risk_flags()
        self.winsorize_purchase_value()
        return self.df
