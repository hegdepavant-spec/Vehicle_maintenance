"""
preprocessing.py
Robust data preprocessing pipeline for Vehicle Maintenance Prediction.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from datetime import datetime

# Paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

CATEGORICAL_COLS = [
    "Vehicle_Model", "Fuel_Type", "Transmission_Type", "Owner_Type",
    "Maintenance_History", "Service_History", "Tire_Condition",
    "Brake_Condition", "Battery_Status"
]

NUMERICAL_COLS = [
    "Mileage", "Reported_Issues", "Vehicle_Age", "Engine_Size",
    "Odometer_Reading", "Insurance_Premium", "Fuel_Efficiency",
    "days_since_last_service", "warranty_remaining_days"
]

TARGET = "Need_Maintenance"


def parse_date_safe(date_series, reference_date=None):
    """Safely parse dates and compute days difference."""
    if reference_date is None:
        reference_date = datetime.now()
    parsed = pd.to_datetime(date_series, errors="coerce")
    delta = (reference_date - parsed).dt.days
    return delta.fillna(delta.median())


def preprocess_data(df: pd.DataFrame, fit: bool = True,
                    encoders: dict = None, scaler: StandardScaler = None):
    """
    Full preprocessing pipeline.
    fit=True: fit encoders/scaler (training). fit=False: transform only (inference).
    Returns: X (DataFrame), y (Series), encoders, scaler
    """
    df = df.copy()

    # --- Date Features ---
    ref_date = datetime(2024, 6, 1)
    df["days_since_last_service"] = parse_date_safe(df.get("Last_Service_Date", pd.Series()), ref_date)
    df["warranty_remaining_days"] = (
        pd.to_datetime(df.get("Warranty_Expiry_Date", pd.Series()), errors="coerce") - ref_date
    ).dt.days.fillna(0).clip(lower=0)

    # --- Drop raw date columns ---
    df.drop(columns=["Last_Service_Date", "Warranty_Expiry_Date"], errors="ignore", inplace=True)

    # --- Fill missing values ---
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")

    for col in NUMERICAL_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)

    # --- Encode categoricals ---
    if fit:
        encoders = {}
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
        joblib.dump(encoders, os.path.join(MODEL_DIR, "encoder.pkl"))
    else:
        for col in CATEGORICAL_COLS:
            if col in df.columns and encoders and col in encoders:
                le = encoders[col]
                df[col] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else 0
                )

    # --- Feature Matrix ---
    feature_cols = [c for c in CATEGORICAL_COLS + NUMERICAL_COLS if c in df.columns]
    X = df[feature_cols].copy()

    # --- Scale ---
    if fit:
        scaler = StandardScaler()
        X[NUMERICAL_COLS] = scaler.fit_transform(X[[c for c in NUMERICAL_COLS if c in X.columns]])
        joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    else:
        if scaler:
            num_cols_present = [c for c in NUMERICAL_COLS if c in X.columns]
            X[num_cols_present] = scaler.transform(X[num_cols_present])

    y = df[TARGET].astype(int) if TARGET in df.columns else None

    return X, y, encoders, scaler


def load_and_preprocess(csv_path: str):
    """Load CSV and run full preprocessing pipeline."""
    df = pd.read_csv(csv_path)
    print(f"[Preprocessing] Loaded {len(df)} rows, {df.shape[1]} columns.")
    X, y, encoders, scaler = preprocess_data(df, fit=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[Preprocessing] Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test, encoders, scaler, X.columns.tolist()


def preprocess_single_row(row_dict: dict, encoders: dict, scaler: StandardScaler):
    """Preprocess a single input row for inference."""
    df = pd.DataFrame([row_dict])
    X, _, _, _ = preprocess_data(df, fit=False, encoders=encoders, scaler=scaler)
    return X


if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), "data", "vehicle_maintenance_data.csv")
    X_train, X_test, y_train, y_test, enc, sc, cols = load_and_preprocess(csv_path)
    print(f"Feature columns: {cols}")
    print("Preprocessing complete.")
