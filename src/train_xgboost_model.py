# src/train_xgboost_model.py

from src.models import xgboost_model

def train_xgb(X_train, y_train):
    model = xgboost_model(y_train)
    model.fit(X_train, y_train)
    return model
