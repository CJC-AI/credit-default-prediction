import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

def plot_dist_target(target: str, df: pd.DataFrame):
    sns.countplot(x=target, data=df)
    plt.title('Target Class Distribution')
    plt.show()

def plot_hist(column: str, df: pd.DataFrame):
    df[column].hist(bins=15, figsize=(40,30))
    plt.show()

def boxplot(x:str, cols, df: pd.DataFrame):
    for col in cols:
        plt.figure(figsize=(10,5))
        sns.boxplot(x=x, y=col, data=df)
        plt.title(f'Boxplot of {col} by Default Status')
        plt.show()

def plot_heatmap(col, df: pd.DataFrame):
    corr = df[col].corr()
    plt.figure(figsize=(12,10))
    sns.heatmap(corr, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

def plot_kde(df: pd.DataFrame, cols, target: str):
    for col in cols:
        plt.figure(figsize=(10,5))
        sns.kdeplot(data=df, x=col, hue=target,common_norm=False)
        plt.title(f'Distribution of {col} by Default Status')
        plt.show()

def plot_prec_reca(rf, xgb, X_test, y_test):
    # Get probabilities from both models
    rf_probs = rf.predict_proba(X_test)[:, 1]
    xgb_probs = xgb.predict_proba(X_test)[:, 1]

    # Compute precision-recall curve
    rf_precision, rf_recall, rf_thresholds = precision_recall_curve(y_test, rf_probs)
    xgb_precision, xgb_recall, xgb_thresholds = precision_recall_curve(y_test, xgb_probs)

    # Plot precision-recall curves
    plt.figure(figsize=(10, 7))

    plt.plot(rf_recall, rf_precision, label='Random Forest', linewidth=3)
    plt.plot(xgb_recall, xgb_precision, label='XGBoost', linewidth=3)

    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curve (RF vs XGB)', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()
