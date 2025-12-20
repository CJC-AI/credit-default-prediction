import pandas as pd
from sklearn.preprocessing import StandardScaler

def clean_df(df: pd.DataFrame):
    """
    Strip empty space and map default columns
    """
    df = df.copy()

    df.columns = df.columns.str.strip()
    df['default'] = df['default'].map({'Y': 1, 'N': 0})

    return df

def clean_categorical_features(df: pd.DataFrame):
    """
    Clean invalid categorical values.
    """
    df = df.copy()
    
    df['EDUCATION'] = df['EDUCATION'].replace({
        '0': 'Other',
        'Unknown': 'Other',
        'Others': 'Other',
        0: 'Other'
    })

    df['MARRIAGE'] = df['MARRIAGE'].replace({
        '0': 'Other',
        0: 'Other'
    })
    
    return df


def engineer_features(df: pd.DataFrame):
    """
    Bank-style feature engineering.
    """
    df = df.copy()

    pay_cols = [f'PAY_{i}' for i in [0,2,3,4,5,6]]
    bill_cols = [f'BILL_AMT{i}' for i in range(1,7)]
    pay_amt_cols = [f'PAY_AMT{i}' for i in range(1,7)]

    df['max_delinquency'] = df[pay_cols].max(axis=1)
    df['num_missed_payments'] = (df[pay_cols] > 0).sum(axis=1)

    df['avg_bill_amt'] = df[bill_cols].mean(axis=1)
    df['bill_vol'] = df[bill_cols].std(axis=1)

    df['avg_payment'] = df[pay_amt_cols].mean(axis=1)
    df['max_utilization'] = df[bill_cols].max(axis=1) / df['LIMIT_BAL']

    return df


def encode_features(df: pd.DataFrame, target: str):
    """
    One-hot encode categorical variables.
    """
    X = df.drop(columns=[target])
    y = df[target]

    X = pd.get_dummies(X, drop_first=True)
    return X, y


def scale_features(X_train, X_test):
    """
    Scale features for linear models.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler
