from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from .base_model import BaseModel

class LogisticRegressionModel(BaseModel):
    def __init__(self, df, target="class", processed_dir="data/processed/models"):
        super().__init__(df, target, processed_dir)

    def train(self):
        # Standardize features
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
        ])
        self.model.fit(self.X_train, self.y_train)
        return self.evaluate()
