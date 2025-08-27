import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class EDAReport:
    """
    Unified EDAReport for fraud dataset
    Combines numeric summaries, categorical summaries, and basic visualizations
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    # ----- Numeric / Target summaries -----
    def target_summary(self):
        rate = self.df['class'].mean()
        counts = self.df['class'].value_counts()
        return {"fraud_rate": rate, "counts": counts.to_dict()}

    def univariate_numeric(self, cols):
        return self.df[cols].describe(percentiles=[0.01,0.05,0.25,0.5,0.75,0.95,0.99]).T

    def top_categorical(self, col, n=10):
        return self.df[col].value_counts().head(n)

    def class_by_hour(self):
        return self.df.groupby('purchase_hour')['class'].agg(['mean','count'])

    # ----- Visualization methods -----
    def plot_class_distribution(self, save_path=None):
        plt.figure(figsize=(6,4))
        sns.countplot(x='class', data=self.df)
        plt.title("Class Distribution (0=Non-Fraud, 1=Fraud)")
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    def plot_numeric_distributions(self, cols, save_path=None):
        n_cols = len(cols)
        n_rows = (n_cols // 3) + int(n_cols % 3 > 0)
        plt.figure(figsize=(5*n_cols, 4*n_rows))
        for i, c in enumerate(cols):
            plt.subplot(n_rows, 3, i+1)
            sns.histplot(self.df[c], kde=True, bins=30)
            plt.title(f"{c} distribution")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    def plot_categorical_top(self, col, n=10, save_path=None):
        top_values = self.df[col].value_counts().head(n)
        plt.figure(figsize=(8,4))
        sns.barplot(x=top_values.index, y=top_values.values)
        plt.title(f"Top {n} {col} categories")
        plt.xticks(rotation=45)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()
