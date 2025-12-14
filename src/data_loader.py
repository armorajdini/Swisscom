# src/data_loader.py

import pandas as pd
from src.config import DATA_PATH, DATE_COL

def load_data():
    df = pd.read_csv(DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    print("Data range:")
    print(df[DATE_COL].min(), "→", df[DATE_COL].max())

    return df
