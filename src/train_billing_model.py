# src/train_billing_model.py

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model
