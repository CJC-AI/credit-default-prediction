import joblib
import pandas as pd
from .preprocessing import feature_engineering

model = joblib.load('artifacts/XGB.pkl')
scaler = joblib.load('artifacts/scaler.pkl')
FEATURE_COLUMNS = joblib.load('artifacts/feature_columns.pkl')

def predict(input_dict: dict):

    # Convert dict -> DataFrame (Single row)
    df = pd.DataFrame([input_dict])

    # Feature engineering
    df = feature_engineering(df)

    # one-hot enconde df
    #df = pd.get_dummies(df)

    # Enforce column order
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    # scale
    df_scaled = scaler.transform(df)

    # predict probability
    proba = model.predict_proba(df_scaled)[0,1]
    
    return float(proba)