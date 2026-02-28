import numpy as np
import shap
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    precision_recall_curve
)


def feature_importance(X, y):
    """
    Calculate feature importances using a simple Decision Tree model.

    Args:
        X (pd.DataFrame): The input features for the model.
        y (pd.Series or np.ndarray): The target variable.

    Returns:
        pd.Series: Feature importances sorted in descending order.
    """
    model = DecisionTreeClassifier(max_depth=3)
    model.fit(X,y)
    importances = pd.Series(model.feature_importances_, index=X.columns)
    return importances.sort_values(ascending=False)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a classification model using standard classification metrics.

    Args:
        model (estimator): The trained machine learning model.
        X_test (pd.DataFrame or np.ndarray): The testing features.
        y_test (pd.Series or np.ndarray): The true labels for the test set.

    Returns:
        dict: A dictionary containing AUC, Recall, Precision, F1, Accuracy, 
              and the Confusion Matrix.
    """
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    return {
        "AUC": roc_auc_score(y_test, probs),
        "Recall": recall_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "F1": f1_score(y_test, preds),
        "Accuracy": accuracy_score(y_test, preds),
        "Confusion Matrix": confusion_matrix(y_test, preds)
    }

def find_best_threshold(model, X_test, y_test):
    """
    Find the optimal classification threshold that maximizes the F1 score.

    Args:
        model (estimator): The trained classification model.
        X_test (pd.DataFrame or np.ndarray): The testing features.
        y_test (pd.Series or np.ndarray): The true labels for the test set.

    Returns:
        dict: A dictionary containing the best threshold, its corresponding F1 score, 
              precision, and recall.
    """
    probs = model.predict_proba(X_test)[:,1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, probs)

    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = f1_scores.argmax()

    return {
        "best_threshold": thresholds[best_idx],
        "best_f1": f1_scores[best_idx],
        "precision": precisions[best_idx],
        "recall": recalls[best_idx],
    }


def evaluate_with_threshold(model, X_test, y_test, threshold):
    """
    Evaluate the model's performance using a custom probability threshold.

    Args:
        model (estimator): The trained classification model.
        X_test (pd.DataFrame or np.ndarray): The testing features.
        y_test (pd.Series or np.ndarray): The true labels for the test set.
        threshold (float): The probability threshold to classify the positive class.

    Returns:
        dict: Evaluation metrics (Recall, Precision, F1, Accuracy, Confusion Matrix) 
              based on the custom threshold.
    """
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    return {
        "Recall": recall_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "F1": f1_score(y_test, preds),
        "Accuracy": accuracy_score(y_test, preds),
        "Confusion Matrix": confusion_matrix(y_test, preds)
    }


def optimize_recall(model, X_test, y_test, prec: float):
    """
    Find the threshold that maximizes recall while maintaining a minimum precision.

    Args:
        model (estimator): The trained classification model.
        X_test (pd.DataFrame or np.ndarray): The testing features.
        y_test (pd.Series or np.ndarray): The true labels for the test set.
        prec (float): The minimum acceptable precision score.

    Returns:
        dict: A dictionary containing the highest recall achievable and the 
              probability threshold required to achieve it.
    """
    probs = model.predict_proba(X_test)[:,1]
    precision, recall, thresholds = precision_recall_curve(y_test, probs)

    mask = precision >= prec
    return{
        
        'best_recall': recall[mask].max(),
        'best_threshold': thresholds[mask.argmax()]
    }

def shap_stab_check(model, X_test):
    """
    Calculate SHAP value stability to understand feature contribution variance.

    Args:
        model (estimator): The trained tree-based machine learning model.
        X_test (pd.DataFrame): The testing features used to calculate SHAP values.

    Returns:
        pd.DataFrame: A DataFrame detailing the mean absolute SHAP value, standard 
                      deviation of SHAP values, and the stability ratio for each feature.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    std_shap = np.std(np.abs(shap_values), axis=0)

    shap_stability = pd.DataFrame({
        'feature': X_test.columns,
        'mean_abs_shap': mean_abs_shap,
        'std_abs_shap': std_shap,
        'stability_ratio': std_shap / mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)

    return shap_stability