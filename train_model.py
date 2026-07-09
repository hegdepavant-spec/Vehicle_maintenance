"""
train_model.py
Train Random Forest classifier for Vehicle Maintenance Prediction.
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)
import json

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def train_random_forest(X_train, y_train, n_estimators=150, max_depth=12, random_state=42):
    """Train Random Forest model."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics dict."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
        "precision": round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        "recall": round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
        "roc_auc": round(roc_auc_score(y_test, y_prob) * 100, 2),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "y_prob": y_prob.tolist(),
        "y_test": y_test.tolist(),
    }

    feature_importances = model.feature_importances_.tolist()
    metrics["feature_importances"] = feature_importances

    print(f"[Model] Accuracy: {metrics['accuracy']}%")
    print(f"[Model] F1-Score: {metrics['f1_score']}%")
    print(f"[Model] ROC-AUC: {metrics['roc_auc']}%")
    return metrics


def save_model(model, path=None):
    """Save trained model."""
    if path is None:
        path = os.path.join(MODEL_DIR, "rf_model.pkl")
    joblib.dump(model, path)
    print(f"[Model] Saved to {path}")


def load_model(path=None):
    """Load saved model."""
    if path is None:
        path = os.path.join(MODEL_DIR, "rf_model.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None


def train_and_save(X_train, X_test, y_train, y_test):
    """Full training pipeline."""
    print("[Training] Starting Random Forest training...")
    model = train_random_forest(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    save_model(model)

    # Save metrics
    metrics_to_save = {k: v for k, v in metrics.items()
                       if k not in ("y_prob", "y_test")}
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics_to_save, f, indent=2)

    # Save feature names
    feature_names = list(X_train.columns)
    with open(os.path.join(MODEL_DIR, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)

    return model, metrics


if __name__ == "__main__":
    from preprocessing import load_and_preprocess
    csv_path = os.path.join(os.path.dirname(__file__), "data", "vehicle_maintenance_data.csv")
    X_train, X_test, y_train, y_test, enc, sc, cols = load_and_preprocess(csv_path)
    model, metrics = train_and_save(X_train, X_test, y_train, y_test)
    print("Training complete.")
