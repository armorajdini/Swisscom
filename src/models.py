# src/models.py

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import numpy as np

def logistic_model():
    return LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        n_jobs=-1
    )

def random_forest_model():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    )

def xgboost_model(y_train):
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale = neg / pos if pos > 0 else 1

    return XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale,
        eval_metric="logloss",
        random_state=42
    )
