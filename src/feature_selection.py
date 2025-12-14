# src/feature_selection.py

import pandas as pd


def select_features(df: pd.DataFrame):
    """
    Trennt Target und Features.
    Entfernt IDs, Target-Leakage und ALLE Datetime-Spalten automatisch.
    """

    df = df.copy()

    # -----------------------
    # Target
    # -----------------------
    y = df["target"]

    # -----------------------
    # Explizit zu droppende Spalten
    # -----------------------
    DROP_COLS = [
        "target",
        "Target_Variable_Count",
        "Target_Variable_LATEST",
        "Customer_ID",
        "Invoice_ID",
    ]

    DROP_COLS = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=DROP_COLS, errors="ignore")

    # -----------------------
    # 🔴 KRITISCHER TEIL
    # Alle datetime64-Spalten entfernen
    # -----------------------
    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns

    if len(datetime_cols) > 0:
        print("Dropping datetime columns:", list(datetime_cols))
        df = df.drop(columns=datetime_cols)

    X = df

    return X, y
