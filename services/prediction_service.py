"""Adapter around the trained ML model, diagnostic engines, and SHAP explainer.

Returns prediction results including SHAP explanation text so the LLM can
explain the model's reasoning rather than inventing its own.
"""

from cache_manager import get, load_all_models, set_cache
from diagnostic_engine import run_diagnostics
from explainability import get_shap_explainer, compute_shap_values_single, build_shap_explanation_text
from preprocessing import preprocess_single_row
from rul_engine import compute_rul


RISK_ORDER = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}


def run_prediction(row: dict, user_provided_keys: set | None = None) -> dict:
    """Run ML prediction with SHAP explanation.

    Args:
        row: Complete feature dict (defaults already merged for missing fields).
        user_provided_keys: Set of ML column names actually provided by the user.
            Used to mark the prediction as preliminary when many fields are defaults.
    """
    if get("model") is None and not load_all_models():
        raise RuntimeError("The trained maintenance model is unavailable.")

    model, encoders, scaler = get("model"), get("encoders"), get("scaler")
    feat_names = get("feature_names")
    features = preprocess_single_row(row, encoders, scaler)

    prediction = int(model.predict(features)[0])
    probability = model.predict_proba(features)[0]
    confidence = round(float(probability[prediction]) * 100, 1)

    rul = compute_rul(row)
    diagnostics = run_diagnostics(row)
    risk_level = max(
        (rul["rul_risk_level"], diagnostics["overall_risk_level"]),
        key=lambda level: RISK_ORDER.get(level, 0),
    )

    # ── SHAP explanation ─────────────────────────────────────────────────────
    shap_explanation = ""
    shap_top_factors = []
    explainer = get("shap_explainer")
    if explainer is None and model is not None:
        explainer = get_shap_explainer(model)
        if explainer is not None:
            set_cache("shap_explainer", explainer)

    if explainer is not None and feat_names:
        shap_vals = compute_shap_values_single(explainer, features)
        if shap_vals is not None and len(shap_vals) == len(feat_names):
            shap_explanation = build_shap_explanation_text(shap_vals, feat_names)
            # Top 3 contributing factors for structured use
            pairs = sorted(zip(feat_names, shap_vals), key=lambda x: abs(x[1]), reverse=True)
            shap_top_factors = [
                {"feature": name.replace("_", " ").title(), "impact": round(float(val), 4)}
                for name, val in pairs[:3]
            ]

    # ── Preliminary flag ─────────────────────────────────────────────────────
    total_features = 10
    user_count = len(user_provided_keys) if user_provided_keys else 0
    is_preliminary = user_count < total_features

    return {
        "prediction": bool(prediction),
        "confidence": confidence,
        "risk_level": risk_level,
        "rul_display": rul["rul_display"],
        "rul_km": rul["rul_km"],
        "vhi": rul["vhi"],
        "key_factors": rul["key_factors"],
        "diagnostics": diagnostics,
        "shap_explanation": shap_explanation,
        "shap_top_factors": shap_top_factors,
        "preliminary": is_preliminary,
        "features_provided": user_count,
        "features_total": total_features,
    }
