import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_curve, auc, confusion_matrix
import joblib

class BaseModel:
    def __init__(self, df: pd.DataFrame, target: str = "class", processed_dir: str = "data/processed/models"):
        self.df = df.copy()
        self.target = target
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.X = self.df.drop(columns=[self.target])
        self.y = self.df[self.target]
        self.model = None

        # Split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

    def evaluate(self):
        preds = self.model.predict(self.X_test)
        f1 = f1_score(self.y_test, preds)
        cm = confusion_matrix(self.y_test, preds)
        return {"f1_score": f1, "confusion_matrix": cm}

    def save_model(self, name: str):
        path = self.processed_dir / f"{name}.joblib"
        joblib.dump(self.model, path)
        return path
