"""
explainability.py
SHAP-based explainability for vehicle maintenance predictions.
"""

import os
import numpy as np
import shap
import joblib
import warnings
warnings.filterwarnings("ignore")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
_shap_explainer_cache = None


def get_shap_explainer(model, X_background=None, force_rebuild=False):
    """Get or create SHAP TreeExplainer (cached)."""
    global _shap_explainer_cache
    explainer_path = os.path.join(MODEL_DIR, "shap_explainer.pkl")

    if not force_rebuild and _shap_explainer_cache is not None:
        return _shap_explainer_cache

    if not force_rebuild and os.path.exists(explainer_path):
        try:
            _shap_explainer_cache = joblib.load(explainer_path)
            return _shap_explainer_cache
        except Exception:
            pass

    if model is None:
        return None

    # Use subset for efficiency
    background = X_background
    if background is not None and len(background) > 200:
        background = shap.sample(background, 200, random_state=42)

    explainer = shap.TreeExplainer(
        model,
        data=background,
        feature_perturbation="interventional" if background is not None else "tree_path_dependent",
    )
    try:
        joblib.dump(explainer, explainer_path)
    except Exception as e:
        print(f"[SHAP] Warning: could not cache explainer: {e}")
    _shap_explainer_cache = explainer
    return explainer


def compute_shap_values_single(explainer, X_row):
    """Compute SHAP values for a single row."""
    try:
        shap_vals = explainer.shap_values(X_row, check_additivity=False)
        # Older SHAP: list[class] -> (rows, features)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]  # class 1 = maintenance needed

        shap_vals = np.asarray(shap_vals)

        # Newer SHAP can return (rows, features, classes) for classifiers.
        if shap_vals.ndim == 3:
            class_index = 1 if shap_vals.shape[-1] > 1 else 0
            shap_vals = shap_vals[:, :, class_index]

        if shap_vals.ndim == 2:
            shap_vals = shap_vals[0]

        return np.asarray(shap_vals, dtype=float).reshape(-1)
    except Exception as e:
        print(f"[SHAP] Error computing values: {e}")
        return None


def compute_shap_values_batch(explainer, X_batch):
    """Compute SHAP values for a batch (for global summary)."""
    try:
        shap_vals = explainer.shap_values(X_batch)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        shap_vals = np.asarray(shap_vals)
        if shap_vals.ndim == 3:
            class_index = 1 if shap_vals.shape[-1] > 1 else 0
            shap_vals = shap_vals[:, :, class_index]
        return shap_vals
    except Exception as e:
        print(f"[SHAP] Batch error: {e}")
        return None


def build_shap_explanation_text(shap_values, feature_names, top_n=5):
    """Generate human-readable SHAP explanation."""
    if shap_values is None or feature_names is None:
        return "SHAP explanation unavailable."

    pairs = sorted(zip(feature_names, shap_values), key=lambda x: abs(x[1]), reverse=True)
    top = pairs[:top_n]

    positives = [(n, v) for n, v in top if v > 0]
    negatives = [(n, v) for n, v in top if v < 0]

    parts = []
    if positives:
        pos_names = ", ".join(
            n.replace("_", " ").title() for n, _ in positives[:3]
        )
        parts.append(f"Maintenance was predicted mainly due to: **{pos_names}**.")
    if negatives:
        neg_names = ", ".join(
            n.replace("_", " ").title() for n, _ in negatives[:2]
        )
        parts.append(f"Factors reducing maintenance risk: {neg_names}.")

    if not parts:
        parts.append("No dominant SHAP factors identified.")

    return " ".join(parts)


def get_feature_importance_data(model, feature_names):
    """Return sorted feature importances."""
    importances = model.feature_importances_
    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    names = [p[0] for p in pairs]
    vals = [p[1] for p in pairs]
    return names, vals


if __name__ == "__main__":
    print("SHAP explainability module loaded successfully.")
