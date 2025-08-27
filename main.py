from pathlib import Path
import pandas as pd

# Task 1

# Import your previously defined classes
from src.data.fraud_data_processor import FraudDataProcessor
from src.features.ip_country_mapper import IPCountryMapper
from src.features.feature_engineer import FeatureEngineer
from src.features.eda_report import EDAReport

# Task 2
from src.models.model_trainer import ModelTrainer

# Task 3
from models.explainability.shap_explainer import SHAPExplainer
from models.explainability.evaluation_report import EvaluationReport
from models.ensemble_model import EnsembleModel


class PipelineRunner:
    """
    Master runner to execute the full Task 1 workflow:
    1. Load and preprocess fraud data
    2. Map IPs to countries
    3. Feature engineering
    4. Generate EDA report
    """

    def __init__(self,
                 fraud_path = Path("./data/raw/Fraud_Data.csv"),
                 ip_path = Path("./data/raw/IpAddress_to_Country.csv"),
                 processed_dir = Path("./data/processed"),
                 figures_dir = Path("./reports/figures")):

        self.fraud_path = Path(fraud_path)
        self.ip_path = Path(ip_path)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        # 1. Load & preprocess raw fraud data
        fraud_processor = FraudDataProcessor(
            raw_data_dir="data/raw",
            processed_data_dir=self.processed_dir
        )
        fraud_df = fraud_processor.load_and_clean_fraud_data()
        ip_df = fraud_processor.load_and_clean_ip_data()


        # 2. Map IPs to countries
        ip_mapper = IPCountryMapper(ip_df, processed_data_dir=self.processed_dir)
        fraud_df = ip_mapper.map_ip_to_country(fraud_df)

        # 3. Feature engineering
        fe = FeatureEngineer(fraud_df)
        fraud_features = fe.run_all()
        
        # 4. Generate EDA report
        eda = EDAReport(fraud_features)
        target_summary = eda.target_summary()

        # Save final dataset
        final_path = self.processed_dir / "fraud_features_final.csv"
        fraud_features.to_csv(final_path, index=False)
        
        # Task 2
        model_trainer = ModelTrainer(fraud_df)
        results = model_trainer.train_models()
        
        
    # Task 3
        explainer = SHAPExplainer(ensemble_model.model, ensemble_model.X_train)
        explainer.global_feature_importance()
        explainer.local_explanation(ensemble_model.X_test, sample_index=0)

        evaluator = EvaluationReport(ensemble_model.model, ensemble_model.X_test, ensemble_model.y_test)
        evaluator.plot_confusion_matrix()
        evaluator.pr_auc_curve()

        return fraud_features,results


if __name__ == "__main__":
    runner = PipelineRunner()
    runner.run()
