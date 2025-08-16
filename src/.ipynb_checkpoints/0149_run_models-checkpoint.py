from src.0149_baseline_models import (
    create_logistic_regression_pipeline,
    create_random_forest_pipeline,
    train_and_evaluate
)
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv("your_data.csv")
X = data["text"] 
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    random_state=42
)

# TF-IDF settings
tfidf_params = {
    "ngram_range": (1, 2),
    "max_features": 5000,   
}

# Logistic Regression settings
lr_params = {"C": 0.1, "max_iter": 1000}

# Random Forest settings
rf_params = {"n_estimators": 100, "max_depth": 50}

# Logistic Regression
print("Training Logistic Regression...")
lr_pipeline = create_logistic_regression_pipeline(
    tfidf_params=tfidf_params,
    model_params=lr_params
)
lr_model, lr_report = train_and_evaluate(
    lr_pipeline, X_train, y_train, X_test, y_test
)

# Random Forest
print("\nTraining Random Forest...")
rf_pipeline = create_random_forest_pipeline(
    tfidf_params=tfidf_params,
    model_params=rf_params
)
rf_model, rf_report = train_and_evaluate(
    rf_pipeline, X_train, y_train, X_test, y_test
)

print("=== Logistic Regression Results ===")
print(lr_report)

print("\n=== Random Forest Results ===")
print(rf_report)