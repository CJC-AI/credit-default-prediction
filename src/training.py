import sys
from pathlib import Path
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# --- Path Resolution ---
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

TRAINING_DATA_PATH = BASE_DIR / "data" / "processed" / "credit_default_engineered.csv"
API_ARTIFACTS_DIR = BASE_DIR / "artifacts" / "api"
MODEL_ARTIFACTS_DIR = BASE_DIR / "artifacts" / "model"


def encode_features(df: pd.DataFrame, target: str):
    """
    Separate the target variable and apply one-hot encoding to categorical features.

    Args:
        df (pd.DataFrame): The input dataframe containing features and the target.
        target (str): The name of the target column.

    Returns:
        tuple: A tuple containing:
            - X (pd.DataFrame): The one-hot encoded feature matrix.
            - y (pd.Series): The target variable.
    """
    X = df.drop(columns=[target])
    y = df[target]

    X = pd.get_dummies(X, drop_first=True)
    return X, y


def load_and_prep(path: str = TRAINING_DATA_PATH):
    """
    Load data, encode features, scale them, and split into train/test sets.

    Args:
        path (str, optional): The file path to the engineered dataset. 
                              Defaults to TRAINING_DATA_PATH.

    Returns:
        tuple: Contains X_train, X_test, y_train, y_test, the fitted StandardScaler,
               and a list of the feature column names.
    """
    df = pd.read_csv(path)
    X, y = encode_features(df, "default")

    # feature scaling 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert scaled array back to DataFrame for SHAP analysis
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.1, random_state=42)

    # Feature columns for model deployment
    feature_columns = X_train.columns.tolist()

    return X_train, X_test, y_train, y_test, scaler, feature_columns


def train_LR(X_train, y_train):
    """
    Train a Logistic Regression classification model.

    Args:
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target labels.

    Returns:
        LogisticRegression: The trained Logistic Regression model.
    """
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_RF(X_train, y_train):
    """
    Train a Random Forest classification model with balanced class weights.

    Args:
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target labels.

    Returns:
        RandomForestClassifier: The trained Random Forest model.
    """
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_XGB(X_train, y_train):
    """
    Train an XGBoost classification model.

    Args:
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target labels.

    Returns:
        XGBClassifier: The trained XGBoost model.
    """
    model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def main():
    """
    Main execution sequence for model training pipeline.
    
    Loads and prepares the dataset, trains multiple models (Logistic Regression, 
    Random Forest, XGBoost), and saves the trained models alongside the scaler 
    and feature columns as artifacts to disk.
    """
    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test, scaler, feature_columns = load_and_prep()

    print("Training Logistic Regression...")
    lr_model = train_LR(X_train, y_train)
    joblib.dump(lr_model, MODEL_ARTIFACTS_DIR / "LR.pkl")

    print("Training Random Forest...")
    rf_model = train_RF(X_train, y_train)
    joblib.dump(rf_model, MODEL_ARTIFACTS_DIR / "RF.pkl")

    print("Training XGBoost...")
    xgb_model = train_XGB(X_train, y_train)
    joblib.dump(xgb_model, MODEL_ARTIFACTS_DIR / "XGB.pkl")

    # Save scaler for inference
    joblib.dump(scaler, API_ARTIFACTS_DIR / "scaler.pkl")

    # Save feature columns for inference consistency
    joblib.dump(feature_columns, API_ARTIFACTS_DIR / "feature_columns.pkl")

    print("✅ Training complete.")
    print(f"Models saved to: {MODEL_ARTIFACTS_DIR}")
    print(f"Scaler & feature columns saved to: {API_ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()