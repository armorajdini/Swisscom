# src/preprocessing.py

import pandas as pd
from src.config import TARGET_MODE, BILLING_SR_CODES


# --------------------------------------------------
# 1) TARGET BUILDING
# --------------------------------------------------

def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Erstellt das Target abhängig vom TARGET_MODE:
    - billing_clean
    - billing_broad
    """

    df = df.copy()

    # -----------------------------
    # Broad Billing
    # -----------------------------
    if TARGET_MODE == "billing_broad":
        df["target"] = (df["Target_Variable_Count"] > 0).astype(int)
        print(f"Billing positives (broad): {df['target'].sum()}")
        return df

    # -----------------------------
    # Clean Billing (OHNE Mahnung)
    # -----------------------------
    is_billing_sr = (
        (df["Target_Variable_Count"] > 0) &
        (df["Latest_SR_Detail"].isin(BILLING_SR_CODES))
    )

    has_dunning = (
        (df["Total_Dunning_Count"] > 0) |
        (df["Dunning_Count_within_90_Days_of_Bill"] > 0) |
        (df["Latest_Mahnstufe"].notna())
    )

    df["target"] = (is_billing_sr & ~has_dunning).astype(int)

    print(f"Billing positives (clean): {df['target'].sum()}")

    return df


# --------------------------------------------------
# 2) FEATURE ENCODING
# --------------------------------------------------

def encode_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    One-Hot-Encoding für alle kategorialen Features
    Train & Test konsistent
    """

    cat_cols = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    X_train_enc = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
    X_test_enc = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)

    # Sicherstellen, dass Train & Test identische Spalten haben
    X_test_enc = X_test_enc.reindex(
        columns=X_train_enc.columns,
        fill_value=0
    )

    return X_train_enc, X_test_enc
