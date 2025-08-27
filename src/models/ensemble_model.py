from sklearn.ensemble import RandomForestClassifier
from .base_model import BaseModel

class EnsembleModel(BaseModel):
    def __init__(self, df, target="class", processed_dir="data/processed/models"):
        super().__init__(df, target, processed_dir)

    def train(self):
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        )
        self.model.fit(self.X_train, self.y_train)
        return self.evaluate()
