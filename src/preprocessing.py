import sys
import pandas as pd
from pathlib import Path

# --- Path Resolution ---
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))


def load_data(path: str):
    """
    Load raw credit default dataset from a CSV file.

    Args:
        path (str): The file path to the raw dataset.

    Returns:
        pd.DataFrame: The loaded raw dataset.
    """
    df = pd.read_csv(path)
    return df


def clean_df(df: pd.DataFrame):
    """
    Strip empty spaces from column names, map the target variable to binary values,
    and drop unnecessary identifying columns.

    Args:
        df (pd.DataFrame): The raw dataframe to be cleaned.

    Returns:
        pd.DataFrame: The dataframe with standardized columns and mapped targets.
    """
    df = df.copy()

    df.columns = df.columns.str.strip()

    df['default'] = df['default'].map({'Y': 1, 'N': 0})

    df = df.drop(columns=['ID', 'AGE'])
    
    return df


def clean_categorical_features(df: pd.DataFrame):
    """
    Standardize invalid or ambiguous categorical values in specific features.
    
    Groups unknown or miscellaneous values in 'EDUCATION' and 'MARRIAGE' 
    under a consistent 'Other' label.

    Args:
        df (pd.DataFrame): The dataframe containing categorical features to clean.

    Returns:
        pd.DataFrame: The dataframe with cleaned categorical values.
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

if __name__ == "__main__":
    RAW_DATA_PATH = "data/raw/credit_default.csv"
    OUTPUT_PATH = "data/processed/credit_default_cleaned.csv"
    
    if Path(RAW_DATA_PATH).exists():
        print(f"Loading data from {RAW_DATA_PATH}...")
        df = load_data(RAW_DATA_PATH)
        df_cleaned = clean_df(df)
        df_cleaned = clean_categorical_features(df_cleaned)
        df_cleaned.to_csv(OUTPUT_PATH, index=False)
        print(f"Cleaned data saved to {OUTPUT_PATH}")
    else:
        print(f"Error: Raw data file not found at {RAW_DATA_PATH}. Please ensure the file exists.")