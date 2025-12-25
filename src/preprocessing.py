import pandas as pd
from sklearn.impute import SimpleImputer
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


def feature_engineering(df: pd.DataFrame):
    """
    Bank-style feature engineering.
    """
    df = df.copy()

    # BEHAVIORAL FEATURES
    pay_cols = [f'PAY_{i}' for i in [0,2,3,4,5,6]]
    # Payment Behavior features
    df['max_delinquency'] = df[pay_cols].max(axis=1)
    df['avg_delinquency'] = df[pay_cols].mean(axis=1)
    df['num_missed_payments'] = (df[pay_cols] > 0).sum(axis=1)
    df['recent_delinquency'] = df['PAY_0']
    # Delinquency Trend
    df['delinq_trend'] = df['PAY_0'] - df['PAY_6']

    # UTILIZATION & EXPOSURE FEATURES
    bill_cols = [f'BILL_AMT{i}' for i in range(1,7)]
    # Credit Utilization Ratio
    df['avg_bill_amt'] = df[bill_cols].mean(axis=1)
    df['credit_utilization'] = df['avg_bill_amt'] / df['LIMIT_BAL']
    # Max Utilization
    df['max_utilization'] = df[bill_cols].max(axis=1) / df['LIMIT_BAL']

    #STABILITY & VOLATILITY FEATURES
    # Bill Volatility 
    df['bill_vol'] = df[bill_cols].std(axis=1) 
    # Utilization Volatility
    df['uti_vol'] = df[bill_cols].div(df['LIMIT_BAL'], axis=0).std(axis=1)

    # PAYMENT CAPACITY & LIQUIDITY FEATURES
    pay_amt_cols = [f'PAY_AMT{i}' for i in range(1,7)]
    # Payment status Ratio
    df['avg_payment'] = df[pay_amt_cols].mean(axis=1)
    df['payment_ratio'] = df['avg_payment'] / (df['avg_bill_amt'] + 1)
    # Payment Consistency
    df['payment_std'] = df[pay_amt_cols].std(axis=1)

    # filling missing values df[payment_ratio]
    imputer = SimpleImputer(strategy='median')
    df[['payment_ratio']] = imputer.fit_transform(df[['payment_ratio']])

    # Correlation pruning
    df = df.drop([
    'BILL_AMT1','BILL_AMT2','BILL_AMT3',
    'PAY_2','PAY_3','PAY_4','PAY_5','PAY_6'
    ], axis=1)

    return df


def encode_features(df: pd.DataFrame, target: str):
    """
    One-hot encode categorical variables.
    """
    X = df.drop(columns=[target])
    y = df[target]

    X = pd.get_dummies(X, drop_first=True)
    return X, y

 
