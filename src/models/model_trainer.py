from .logistic_regression_model import LogisticRegressionModel
from .ensemble_model import EnsembleModel
import logging

class ModelTrainer:
    def __init__(self, df, target="class", processed_dir="data/processed/models"):
        self.df = df
        self.target = target
        self.processed_dir = processed_dir
        self.logger = logging.getLogger("ModelTrainer")
        self.results = {}

    def train_models(self):
        lr_model = LogisticRegressionModel(self.df, self.target, self.processed_dir)
        self.results["logistic_regression"] = lr_model.train()
        lr_model.save_model("logistic_regression_model")

        ensemble_model = EnsembleModel(self.df, self.target, self.processed_dir)
        self.results["ensemble_model"] = ensemble_model.train()
        ensemble_model.save_model("ensemble_model")

        return self.results
