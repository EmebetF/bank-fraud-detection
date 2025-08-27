import shap
import matplotlib.pyplot as plt
from pathlib import Path

class SHAPExplainer:
    def __init__(self, model, X_train, processed_dir="reports/figures"):
        self.model = model
        self.X_train = X_train
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def global_feature_importance(self, top_n=20):
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.X_train)
        
        # For binary classification, take shap_values[1] for positive class
        shap.summary_plot(shap_values[1], self.X_train, plot_type="bar", max_display=top_n, show=False)
        out_path = self.processed_dir / "global_feature_importance.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        return out_path

    def local_explanation(self, X_sample, sample_index=0):
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_sample)
        shap.force_plot(explainer.expected_value[1], shap_values[1][sample_index,:], X_sample.iloc[sample_index,:], matplotlib=True, show=False)
        out_path = self.processed_dir / f"local_explanation_{sample_index}.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        return out_path
