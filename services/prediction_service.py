"""Adapter around the existing trained ML model and diagnostic engines."""

from cache_manager import get, load_all_models
from diagnostic_engine import run_diagnostics
from preprocessing import preprocess_single_row
from rul_engine import compute_rul


RISK_ORDER = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}


def run_prediction(row: dict) -> dict:
    if get("model") is None and not load_all_models():
        raise RuntimeError("The trained maintenance model is unavailable.")
    model, encoders, scaler = get("model"), get("encoders"), get("scaler")
    features = preprocess_single_row(row, encoders, scaler)
    prediction = int(model.predict(features)[0])
    probability = model.predict_proba(features)[0]
    confidence = round(float(probability[prediction]) * 100, 1)
    rul = compute_rul(row)
    diagnostics = run_diagnostics(row)
    risk_level = max((rul["rul_risk_level"], diagnostics["overall_risk_level"]), key=lambda level: RISK_ORDER[level])
    return {
        "prediction": bool(prediction), "confidence": confidence, "risk_level": risk_level,
        "rul_display": rul["rul_display"], "rul_km": rul["rul_km"], "vhi": rul["vhi"],
        "key_factors": rul["key_factors"], "diagnostics": diagnostics,
    }
