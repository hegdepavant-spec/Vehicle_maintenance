"""
cache_manager.py
Global model cache — prevents repeated loading / SHAP initialisation.
"""

import os
import joblib
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# ── Global Cache ──────────────────────────────────────────────────────────────
_cache = {
    "model":         None,
    "encoders":      None,
    "scaler":        None,
    "shap_explainer":None,
    "feature_names": None,
    "metrics":       None,
    "X_test":        None,
    "y_test":        None,
    "y_prob":        None,
    "initialized":   False,
}


def is_initialized() -> bool:
    return _cache["initialized"]


def get(key):
    return _cache.get(key)


def set_cache(key, value):
    _cache[key] = value


def load_all_models():
    """Load all saved models into cache. Returns True if successful."""
    try:
        model_path   = os.path.join(MODEL_DIR, "rf_model.pkl")
        encoder_path = os.path.join(MODEL_DIR, "encoder.pkl")
        scaler_path  = os.path.join(MODEL_DIR, "scaler.pkl")
        metrics_path = os.path.join(MODEL_DIR, "metrics.json")

        if not all(os.path.exists(p) for p in [model_path, encoder_path, scaler_path]):
            return False

        _cache["model"]    = joblib.load(model_path)
        _cache["encoders"] = joblib.load(encoder_path)
        _cache["scaler"]   = joblib.load(scaler_path)

        # Load metrics
        if os.path.exists(metrics_path):
            import json
            with open(metrics_path) as f:
                _cache["metrics"] = json.load(f)

        # Load feature names and validate count matches the current pipeline
        EXPECTED_FEATURE_COUNT = 10
        feat_path = os.path.join(MODEL_DIR, "feature_names.json")
        if os.path.exists(feat_path):
            import json
            with open(feat_path) as f:
                _cache["feature_names"] = json.load(f)
            # If the saved model was trained on a different feature set, force retrain
            if len(_cache["feature_names"]) != EXPECTED_FEATURE_COUNT:
                print(f"[Cache] Feature count mismatch: saved={len(_cache['feature_names'])}, "
                      f"expected={EXPECTED_FEATURE_COUNT}. Forcing retrain.")
                return False
        else:
            print("[Cache] feature_names.json not found. Forcing retrain.")
            return False

        # Load SHAP explainer if cached
        shap_path = os.path.join(MODEL_DIR, "shap_explainer.pkl")
        if os.path.exists(shap_path):
            try:
                _cache["shap_explainer"] = joblib.load(shap_path)
            except Exception:
                pass

        _cache["initialized"] = True
        print("[Cache] All models loaded successfully.")
        return True

    except Exception as e:
        print(f"[Cache] Error loading models: {e}")
        return False


def clear_cache():
    """Clear all cached objects (for re-training)."""
    for key in _cache:
        if key != "initialized":
            _cache[key] = None
    _cache["initialized"] = False
