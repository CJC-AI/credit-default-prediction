import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """
    Load raw credit default dataset.
    
    Parameters
    ----------
    path : str
        Path to CSV file
    
    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(path)
    return df
