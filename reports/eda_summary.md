Exploratory Data Analysis (EDA) Report — Credit Default Dataset

This report summarizes the structure, distribution, relationships, and insights found within the dataset used for predicting credit card default risk.

## 1.  Dataset Overview

The dataset contains 30,000 customers, with features describing credit limits, demographics, past payments, bill amounts, repayment amounts, and a target label:

default:

0 = No default

1 = Default next month

The dataset includes:

Numerical Features

LIMIT_BAL

PAY_0 to PAY_6 (payment status history)

BILL_AMT1 to BILL_AMT6

PAY_AMT1 to PAY_AMT6

AGE

Categorical Features

SEX (M, F)

EDUCATION (Graduate, University, High School, Others, Unknown)

MARRIAGE (Single, Married, Others, Unknown)

## 2. Target Variable Balance
Default	Count	Percentage
0 (No Default)	23,360	78%
1 (Default)	6,640	22%

 Insight:

The dataset is imbalanced — only ~22% default.

Models may require class weighting, SMOTE, or threshold tuning.

## 3. Summary Statistics for Numerical Features

The dataset shows:

Key Insights

LIMIT_BAL is right-skewed with many customers around 50k–240k and very few at the max (1M).

Payment status features (PAY_0–PAY_6) cluster around -1 to 1, indicating:

0 = paid on time

-1 = fully paid

1+ = delay

BILL_AMT and PAY_AMT features are heavily right-skewed, with:

Many near 0

Some extremely large outliers (e.g., >1,000,000 NT dollars)

 Interpretation:
Spending and payment behavior vary widely; strong skew indicates the need for scaling or log transformation.

## 4. Distribution Plots (Histograms)

Across all numerical features:

General Observations

Payment delays (PAY_X): concentrated around -1 to 1, with few severe delays (5+).

Bill amounts: right-skewed with large outliers.

Repayment amounts: mostly small, some extremely large.

Default: strongly skewed (many zeros vs few ones).

Interpretation:
These features contain high variability, skewness, and outliers, all of which significantly affect models like logistic regression but are well-handled by tree-based models.

## 5. Distribution of LIMIT_BAL by Default Status

Key Findings:

Defaulters have lower credit limits.

Non-defaulters span a much wider limit range, including high limits.

The distributions overlap, meaning LIMIT_BAL alone cannot predict default.

Insight:
Lower credit limits correlate with higher default rates.

## 6. Categorical Feature Analysis
SEX vs Default
Sex	Default Rate
F	20.8%
M	24.1%

Insight:
Men default slightly more than women, but difference is small.

EDUCATION vs Default
Education Level	Default Rate
Graduate School	19%
University	23.7%
High School	25.1%
Others	5%
Unknown	7%

Insight:
Default rate increases as education level decreases.
"Others" and "Unknown" are unreliable categories, likely misclassified.

MARRIAGE vs Default
Marriage Status	Default Rate
Married	23.5%
Single	20.9%
Other	26%
Unknown	9%

Insight:
"Other" category has the highest default rate, may represent unstable financial situations.

## 7. Boxplots (Default vs Categorical Features)
SEX

Slight difference in medians; men default a bit more.

No extreme variation.

EDUCATION

High school shows the highest spread toward default.

Graduate students show the lowest likelihood of default.

MARRIAGE

“Other” category shows higher median default.

Married and single have similar, lower distributions.

Insight:
Categorical features have weak-to-moderate relation to default.

## 8. Correlation Analysis
Strong Positive Correlations

BILL_AMT1–BILL_AMT6 (0.9+)

PAY_AMT1–PAY_AMT6 (moderate)

PAY_0 strongly correlated with PAY_2–PAY_6

Features Correlated with Default

Strongest: PAY_0, PAY_2, PAY_3

Moderate: PAY_4 to PAY_6

Weak: LIMIT_BAL, bill amounts, repayment amounts

Key Finding:
Payment delay history (PAY_X) is the strongest predictor of default.

## 9. Decision Tree Feature Importance

Top predictors (max_depth=3):

Feature	Importance
PAY_0	0.76
PAY_2	0.15
PAY_AMT3	0.05
PAY_6	0.017
PAY_3	0.011
Others	~0

Interpretation:

PAY_0 (latest payment status) dominates — the single most valuable predictor.

PAY_2 adds significant additional signal.

Repayment amount adds small but helpful influence.

Categorical variables have almost zero importance in the tree.

## 10. Data Quality Issues
Missing or Invalid Categories

EDUCATION includes 0, 5, 6 → should be grouped into “Unknown/Others”

MARRIAGE includes 0 → also invalid

Highly Skewed Features

BILL_AMT & PAY_AMT require scaling or winsorization.

Class Imbalance

Needs balancing techniques.

## 11. Key Conclusions
Most Important Predictors

Payment delay history (PAY_0, PAY_2, PAY_3)

Select repayment amounts (PAY_AMT3)

Mild contribution from bill amounts

Least Useful Predictors

Categorical features after encoding (SEX, EDUCATION, MARRIAGE)

LIMIT_BAL (weak correlation)


## 12.  Final Summary 

The dataset shows that payment behavior is the most significant driver of credit default risk, far outweighing demographic attributes like sex, education, or marriage status. Customers with recent payment delays and lower credit limits exhibit higher default rates. The dataset contains skewness, outliers, and some invalid categorical codes that require preprocessing. Tree-based models are likely to perform best due to nonlinear patterns and skewed distributions.