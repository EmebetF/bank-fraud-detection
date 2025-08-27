from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

class EvaluationReport:
    def __init__(self, model, X_test, y_test, processed_dir="reports/figures"):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def plot_confusion_matrix(self):
        preds = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, preds)
        plt.figure(figsize=(5,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        out_path = self.processed_dir / "confusion_matrix.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        return out_path

    def pr_auc_curve(self):
        probs = self.model.predict_proba(self.X_test)[:,1]
        precision, recall, _ = precision_recall_curve(self.y_test, probs)
        pr_auc_score = auc(recall, precision)
        plt.figure(figsize=(6,4))
        plt.plot(recall, precision, label=f'PR AUC={pr_auc_score:.4f}')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        out_path = self.processed_dir / "pr_curve.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        return out_path
