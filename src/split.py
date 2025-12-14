# src/split.py

import pandas as pd
from src.config import DATE_COL, TRAIN_MONTHS, TEST_MONTHS

def monthly_train_test_split(df):

    train_months = pd.to_datetime(TRAIN_MONTHS)
    test_months = pd.to_datetime(TEST_MONTHS)

    train_df = df[df[DATE_COL].isin(train_months)]
    test_df = df[df[DATE_COL].isin(test_months)]

    print("Train rows:", len(train_df))
    print("Test rows: ", len(test_df))

    return train_df, test_df
