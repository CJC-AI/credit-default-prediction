# Project Overview

This project focuses on building an end-to-end **credit default risk prediction system** using classical machine learning and gradient boosting techniques. The objective is to predict whether a credit card customer will default on payment, using historical repayment behavior, billing information, and engineered risk indicators.

The project mirrors **real-world bank-style credit risk modeling**, emphasizing:

* Robust exploratory data analysis (EDA)
* Business-driven feature engineering
* Strong baseline modeling
* Advanced tree-based models (Random Forest, XGBoost)
* Threshold optimization, explainability (SHAP), and governance

---

# Dataset Description

The dataset contains **30,000 customers** and **25+ features**, including:

* **Demographics**: SEX, EDUCATION, MARRIAGE, AGE
* **Credit Profile**: LIMIT_BAL
* **Repayment Status**: PAY_0 to PAY_6
* **Billing History**: BILL_AMT1 to BILL_AMT6
* **Payment History**: PAY_AMT1 to PAY_AMT6
* **Target Variable**: `default` (1 = default, 0 = non-default)

The dataset is **moderately imbalanced (~22% default rate)**, reflecting realistic consumer credit portfolios.

---

# Project Architecture

```
credit-default-risk/
│
├── notebooks/                
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
│
├── src/                       
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── training.py
│   └── eval.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Baseline Model

**Logistic Regression** was used as the baseline due to:

* Interpretability
* Regulatory acceptance in banking
* Strong linear benchmark

Baseline performance highlighted **high precision but low recall**, confirming that linear models struggle to capture complex repayment dynamics.

---

# Feature Engineering

Bank-style feature engineering was applied:

### Behavioral Risk

* `max_delinquency`
* `num_missed_payments`

### Exposure & Utilization

* `avg_bill_amt`
* `max_utilization`
* `bill_vol` (billing volatility)

### Payment Capacity

* `avg_payment`
* Payment-to-bill ratios

### Data Cleaning & Encoding

* Invalid category handling (EDUCATION=0, MARRIAGE=0)
* One-hot encoding of categorical variables

---

# Model Development

Three models were trained:

| Model               | Characteristics                      |
| ------------------- | ------------------------------------ |
| Logistic Regression | Interpretable, linear, baseline      |
| Random Forest       | Non-linear, robust, ensemble         |
| XGBoost             | Boosted trees, high predictive power |

Advanced steps:

* Class imbalance handling
* Precision–Recall threshold tuning
* SHAP-based explainability
* Feature stability analysis

---

# Results Comparison (Side-by-Side)

| Metric    | Logistic Regression | Random Forest (Tuned) | XGBoost (Tuned) |
| --------- | ------------------- | --------------------- | --------------- |
| ROC-AUC   | 0.72                | 0.78                  | **0.79**        |
| Recall    | 0.24                | 0.55                  | **0.56**        |
| Precision | **0.70**            | 0.56                  | 0.55            |
| F1 Score  | 0.35                | **0.55**              | **0.55**        |
| Accuracy  | 0.81                | 0.81                  | 0.80            |

---

# Business Interpretation

* **Logistic Regression**: Conservative model, flags fewer defaulters but with high confidence. Suitable for strict approval policies.
* **Random Forest**: Balanced model with strong recall and interpretability through feature importance.
* **XGBoost**: Best overall risk capture, identifying more defaulters at acceptable precision.

For credit risk operations:

* High recall reduces **unexpected losses**
* Threshold tuning enables alignment with **risk appetite**

---

# Model Card (XGBoost – Final Candidate)

**Intended Use**: Consumer credit default risk prediction

**Target**: Probability of default within next billing cycle

**Key Drivers**:

* Delinquency history
* Payment consistency
* Credit utilization

**Limitations**:

* No temporal validation
* No reject inference
* Dataset-specific

**Ethical Considerations**:

* No protected attributes used for decisioning
* SHAP ensures transparency

---

# How to Run

```bash
conda create -n credit-risk python=3.10
conda activate credit-risk
pip install -r requirements.txt
python src/training.py
```

---

# Future Improvements

* Probability calibration (Platt / Isotonic)
* Time-based cross-validation
* Reject inference
* Cost-sensitive optimization
* Deployment via FastAPI
* Monitoring & model drift detection


