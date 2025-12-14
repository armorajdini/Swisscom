# main.py

from src.data_loader import load_data
from src.preprocessing import build_target, encode_features
from src.feature_selection import select_features
from src.split import monthly_train_test_split
from src.models import logistic_model, random_forest_model
from src.train_billing_model import train_model
from src.train_xgboost_model import train_xgb
from src.evaluate import full_visual_evaluation

def main():

    df = load_data()
    df = build_target(df)

    train_df, test_df = monthly_train_test_split(df)

    X_train, y_train = select_features(train_df)
    X_test, y_test = select_features(test_df)

    # 🔥 HIER der entscheidende Schritt
    X_train, X_test = encode_features(X_train, X_test)

    # Logistic Regression
    log = train_model(logistic_model(), X_train, y_train)
    full_visual_evaluation(log, X_test, y_test, "Logistic Regression")

    # Random Forest
    rf = train_model(random_forest_model(), X_train, y_train)
    full_visual_evaluation(rf, X_test, y_test, "Random Forest")

    # XGBoost
    xgb = train_xgb(X_train, y_train)
    full_visual_evaluation(xgb, X_test, y_test, "XGBoost")

if __name__ == "__main__":
    main()
