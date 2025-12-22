import numpy as np
import shap
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    precision_recall_curve
)

def evaluate_model(model, X_test, y_test):
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
    probs = model.predict_proba(X_test)[:,1]
    precision, recall, thresholds = precision_recall_curve(y_test, probs)

    mask = precision >= prec
    return{
        
        'best_recall': recall[mask].max(),
        'best_threshold': thresholds[mask.argmax()]
    }

def shap_stab_check(model, X_test):
    '''
    Shap stability check
    '''
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