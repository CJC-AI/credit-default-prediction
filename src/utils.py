import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

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