"""
app.py
Vehicle Maintenance Prediction & Explainable Diagnostics Platform
Main Gradio Dashboard â€” Production-Grade UI
"""

import os
import sys
import json
import time
import html
import warnings
import numpy as np
import pandas as pd
import gradio as gr
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

# â”€â”€ Project Imports â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
sys.path.insert(0, os.path.dirname(__file__))

from preprocessing    import load_and_preprocess, preprocess_single_row
from train_model      import train_and_save
from explainability   import (get_shap_explainer, compute_shap_values_single,
                              build_shap_explanation_text, get_feature_importance_data)
from diagnostic_engine import run_diagnostics
from rul_engine        import compute_rul, compute_vhi
from ai_agent          import run_chat, api_available
from agents.vehicle_agent import handle_message
from services.advisor_service import gemini_health_check
from visualizations    import (plot_vhi_gauge, plot_rul_gauge, plot_component_risk,
                               plot_shap_waterfall, plot_feature_importance,
                               plot_confusion_matrix, plot_roc_curve, plot_metrics_bar)
from cache_manager     import load_all_models, is_initialized, get, set_cache
from utils             import (
    build_diagnostic_report_html,
    build_summary_html,
    format_number,
    get_failed_failing_components,
)

# â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
BASE_DIR  = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "vehicle_maintenance_data.csv")

# â”€â”€ Global state for AI chat context â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_last_vehicle_context = {}

RISK_ORDER = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}


def _max_risk(*levels):
    valid = [str(level).upper() for level in levels if level]
    return max(valid, key=lambda level: RISK_ORDER.get(level, -1)) if valid else "LOW"

# â”€â”€ Startup: Train or Load Models â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def initialise_system():
    """Train models if not cached, else load from disk."""
    if is_initialized():
        return True, "Models already loaded."

    # Try loading existing models first
    if load_all_models():
        return True, "âœ… Models loaded from disk."

    # Train fresh
    try:
        print("[Init] Training models from scratch...")
        X_train, X_test, y_train, y_test, encoders, scaler, feature_names = \
            load_and_preprocess(DATA_PATH)

        model, metrics = train_and_save(X_train, X_test, y_train, y_test)

        # Build SHAP explainer
        explainer = get_shap_explainer(model, X_train, force_rebuild=True)

        # Cache everything
        set_cache("model",          model)
        set_cache("encoders",       encoders)
        set_cache("scaler",         scaler)
        set_cache("feature_names",  feature_names)
        set_cache("shap_explainer", explainer)
        set_cache("metrics",        metrics)
        set_cache("X_test",         X_test)
        set_cache("y_test",         y_test.tolist())
        set_cache("y_prob",         metrics.get("y_prob", []))

        # Mark initialized in cache module
        import cache_manager
        cache_manager._cache["initialized"] = True

        print("[Init] System ready.")
        return True, "âœ… Models trained and ready."
    except Exception as e:
        return False, f"âŒ Initialisation error: {e}"


# â”€â”€ Prediction Core â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_prediction(
    vehicle_model, mileage, maintenance_history, reported_issues,
    vehicle_age, fuel_type, transmission_type, engine_size,
    odometer_reading, last_service_date, warranty_expiry_date,
    owner_type, insurance_premium, service_history, accident_history,
    fuel_efficiency, tire_condition, brake_condition, battery_status
):
    """Run full analysis on a single vehicle row."""
    global _last_vehicle_context

    model    = get("model")
    encoders = get("encoders")
    scaler   = get("scaler")
    explainer= get("shap_explainer")
    feat_names = get("feature_names")

    if model is None:
        err = "âš ï¸ Models not loaded. Please wait for initialisation."
        return err, None, None, None, None, None, None, None, err

    row_raw = {
        "Vehicle_Model":       vehicle_model,
        "Mileage":             mileage,
        "Maintenance_History": maintenance_history,
        "Reported_Issues":     reported_issues,
        "Vehicle_Age":         vehicle_age,
        "Fuel_Type":           fuel_type,
        "Transmission_Type":   transmission_type,
        "Engine_Size":         engine_size,
        "Odometer_Reading":    odometer_reading,
        "Last_Service_Date":   last_service_date,
        "Warranty_Expiry_Date":warranty_expiry_date,
        "Owner_Type":          owner_type,
        "Insurance_Premium":   insurance_premium,
        "Service_History":     service_history,
        "Accident_History":    accident_history,
        "Fuel_Efficiency":     fuel_efficiency,
        "Tire_Condition":      tire_condition,
        "Brake_Condition":     brake_condition,
        "Battery_Status":      battery_status,
    }

    try:
        # â”€â”€ Preprocess for ML â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        X_row = preprocess_single_row(row_raw, encoders, scaler)

        # â”€â”€ ML Prediction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        pred       = int(model.predict(X_row)[0])
        prob       = model.predict_proba(X_row)[0]
        confidence = round(float(prob[pred]) * 100, 1)

        # â”€â”€ RUL & VHI â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        rul_result = compute_rul(row_raw)
        vhi        = rul_result["vhi"]
        rul_km     = rul_result["rul_km"]
        rul_label  = rul_result["rul_label"]

        # â”€â”€ Diagnostics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        diag = run_diagnostics(row_raw)

        # â”€â”€ SHAP â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        shap_vals  = None
        shap_text  = "SHAP explanation unavailable."
        shap_fig   = None

        if explainer is None and model is not None:
            explainer = get_shap_explainer(model)
            if explainer is not None:
                set_cache("shap_explainer", explainer)

        if explainer is not None and feat_names:
            shap_vals = compute_shap_values_single(explainer, X_row)
            if shap_vals is not None and len(shap_vals) == len(feat_names):
                shap_text = build_shap_explanation_text(shap_vals, feat_names)
                shap_fig = plot_shap_waterfall(shap_vals, feat_names)
            elif shap_vals is not None:
                shap_text = (
                    "SHAP explanation unavailable because the SHAP value count "
                    f"({len(shap_vals)}) did not match the feature count ({len(feat_names)})."
                )

        # â”€â”€ Recommendation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        overall_risk = _max_risk(diag["overall_risk_level"], rul_result.get("rul_risk_level"))
        recommendation = diag["most_vulnerable_recommendation"]
        component_status = get_failed_failing_components(rul_result.get("component_health", {}))
        if overall_risk == "CRITICAL" or rul_result.get("rul_label") == "Immediate Maintenance Required":
            recommendation = (
                "Immediate Maintenance Required. Vehicle health/RUL has reached a critical "
                f"risk threshold. {recommendation}"
            )
        elif overall_risk == "HIGH" or pred == 1:
            recommendation = (
                "Maintenance should be scheduled before extended operation. "
                f"{recommendation}"
            )

        # â”€â”€ Store context for AI agent â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        _last_vehicle_context = {
            "prediction":    "YES" if pred == 1 else "NO",
            "confidence":    confidence,
            "vhi":           vhi,
            "rul_km":        rul_km,
            "rul_display":   rul_result.get("rul_display"),
            "rul_label":     rul_label,
            "most_vulnerable": diag["most_vulnerable_component"],
            "risk_level":    overall_risk,
            "shap_explanation": shap_text,
            "recommendation":   recommendation,
            "key_factors":       rul_result.get("key_factors", []),
            "component_health":  rul_result.get("component_health", {}),
            "failed_components": component_status["failed"],
            "failing_components": component_status["failing"],
            "degraded_components": component_status["degraded"],
        }

        # â”€â”€ Build Outputs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        summary_html = build_summary_html(
            pred, confidence, vhi, rul_km, rul_label,
            diag["most_vulnerable_component"], overall_risk,
            shap_text, recommendation,
            component_health=rul_result.get("component_health", {}),
            key_factors=rul_result.get("key_factors", []),
            rul_interpretation=rul_result.get("interpretation", ""),
        )
        diagnostic_report_html = build_diagnostic_report_html(diag, rul_result)

        vhi_fig  = plot_vhi_gauge(vhi)
        rul_fig  = plot_rul_gauge(rul_km)
        risk_fig = plot_component_risk(diag["systems"])

        return (
            summary_html,
            vhi_fig,
            rul_fig,
            risk_fig,
            shap_fig if shap_fig else plot_shap_waterfall(None, None),
            shap_text,
            rul_result["interpretation"],
            gr.update(value=f"Context loaded: {row_raw.get('Vehicle_Model','Vehicle')} analysis complete. Ask me anything!"),
            diagnostic_report_html,
        )

    except Exception as e:
        import traceback
        err = f"âŒ Prediction error: {e}\n{traceback.format_exc()}"
        print(err)
        return err, None, None, None, None, "Error", "Error", gr.update(value="Error during analysis."), err


# â”€â”€ System Insights â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def get_system_insights():
    """Return model performance plots for the System Insights tab."""
    metrics = get("metrics")
    y_test  = get("y_test")
    y_prob  = get("y_prob")
    model   = get("model")
    feat_names = get("feature_names")

    if metrics is None:
        empty = plot_vhi_gauge(0)
        return empty, empty, empty, empty, "No metrics available. Train the model first."

    cm_fig      = plot_confusion_matrix(metrics["confusion_matrix"])
    metrics_fig = plot_metrics_bar(metrics)

    roc_fig = None
    if y_test and y_prob:
        try:
            roc_fig = plot_roc_curve(y_test, y_prob)
        except Exception:
            roc_fig = None

    fi_fig = None
    if model and feat_names:
        fi_names, fi_vals = get_feature_importance_data(model, feat_names)
        fi_fig = plot_feature_importance(fi_names, fi_vals)

    report = (
        f"**Accuracy:** {metrics.get('accuracy',0):.1f}%   |   "
        f"**Precision:** {metrics.get('precision',0):.1f}%   |   "
        f"**Recall:** {metrics.get('recall',0):.1f}%   |   "
        f"**F1-Score:** {metrics.get('f1_score',0):.1f}%   |   "
        f"**ROC-AUC:** {metrics.get('roc_auc',0):.1f}%"
    )

    return cm_fig, metrics_fig, roc_fig or cm_fig, fi_fig or cm_fig, report


# â”€â”€ AI Chat â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def chat_with_agent(user_message, history):
    """Handle chat turn."""
    user_message = (user_message or "").strip()
    if not user_message:
        return history, ""

    ctx = _last_vehicle_context if _last_vehicle_context else None
    history = list(history or [])
    response = run_chat(user_message, history=history, vehicle_context=ctx)

    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": response})
    return history, ""


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# CSS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

CUSTOM_CSS = """
:root {
  --bg-deep:      #07111F;
  --bg-card:      #0F1B2D;
  --bg-panel:     #132238;
  --border:       #2C3D55;
  --border-glow:  rgba(56,189,248,0.45);
  --primary:      #38BDF8;
  --primary-strong:#0EA5E9;
  --secondary:    #8B5CF6;
  --accent:       #C084FC;
  --success:      #00E676;
  --warning:      #FACC15;
  --danger:       #FB7185;
  --text-main:    #F8FAFC;
  --text-muted:   #A8B3C7;
  --text-sub:     #CBD5E1;
  --button-text:  #04111F;
  --radius:       10px;
  --radius-sm:    8px;
  --shadow:       0 8px 32px rgba(0,0,0,0.5);
}

* { box-sizing: border-box; }

body, .gradio-container {
  background: var(--bg-deep) !important;
  font-family: "Segoe UI", Roboto, Arial, sans-serif !important;
  color: var(--text-main) !important;
  font-size: 16px !important;
  line-height: 1.55 !important;
}

/* â”€â”€ Header Banner â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
.header-banner {
  background: linear-gradient(135deg, #060B14 0%, #0D1A2E 50%, #060B14 100%);
  border-bottom: 1px solid var(--border);
  padding: 24px 32px 20px;
  position: relative;
  overflow: hidden;
}
.header-banner::before {
  content: '';
  position: absolute;
  top: -60px; left: -60px;
  width: 300px; height: 300px;
  background: radial-gradient(circle, rgba(0,212,255,0.06) 0%, transparent 70%);
  pointer-events: none;
}
.header-banner::after {
  content: '';
  position: absolute;
  bottom: -80px; right: -80px;
  width: 350px; height: 350px;
  background: radial-gradient(circle, rgba(123,47,190,0.08) 0%, transparent 70%);
  pointer-events: none;
}
.header-title {
  font-size: 2em;
  font-weight: 800;
  background: linear-gradient(90deg, #00D4FF, #7B2FBE, #A855F7);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin: 0 0 6px 0;
}
.header-sub {
  color: var(--text-sub);
  font-size: 0.95em;
  letter-spacing: 0;
}

/* â”€â”€ Tabs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
.tab-nav {
  background: var(--bg-card) !important;
  border-bottom: 1px solid var(--border) !important;
  padding: 0 8px !important;
}
.tab-nav button {
  background: #101C2F !important;
  color: var(--text-sub) !important;
  border: 1px solid #22324A !important;
  border-bottom: 2px solid #22324A !important;
  border-radius: 8px 8px 0 0 !important;
  padding: 12px 18px !important;
  font-weight: 700 !important;
  font-size: 0.92em !important;
  transition: all 0.2s ease !important;
  letter-spacing: 0 !important;
}
.tab-nav button:hover {
  color: #FFFFFF !important;
  border-color: var(--primary) !important;
  background: #16304E !important;
}
.tab-nav button.selected {
  color: #FFFFFF !important;
  border-bottom: 3px solid var(--primary) !important;
  background: #164266 !important;
}

/* â”€â”€ Cards â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
.stat-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 18px 20px;
  transition: border-color 0.2s, box-shadow 0.2s;
}
.stat-card:hover {
  border-color: var(--border-glow);
  box-shadow: 0 0 20px rgba(0,212,255,0.07);
}
.metric-label {
  font-size: 0.78em;
  color: var(--text-sub);
  text-transform: uppercase;
  letter-spacing: 1px;
  margin-bottom: 6px;
}
.metric-value {
  font-size: 1.8em;
  font-weight: 800;
  color: var(--primary);
}

/* â”€â”€ Section Heading â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
.section-heading {
  font-size: 1.05em;
  font-weight: 800;
  color: var(--text-main);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  padding: 6px 0 10px 0;
  border-bottom: 1px solid var(--border);
  margin-bottom: 14px;
}

/* â”€â”€ Inputs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
.gradio-container input,
.gradio-container select,
.gradio-container textarea {
  background: var(--bg-panel) !important;
  border: 1px solid #3B4F6C !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-main) !important;
  font-size: 1.28em !important;
}
.gradio-container input:focus,
.gradio-container select:focus {
  border-color: var(--primary) !important;
  box-shadow: 0 0 0 2px rgba(0,212,255,0.12) !important;
}
label { color: var(--text-sub) !important; font-size: 0.9em !important; font-weight:700 !important; }

/* â”€â”€ Buttons â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
.gradio-container button,
.gradio-container .btn-primary button,
.gradio-container .btn-secondary button,
.gradio-container button.btn-primary,
.gradio-container button.btn-secondary {
  min-height: 44px !important;
  border-radius: 8px !important;
  border: 1px solid #4B6588 !important;
  background: #1F334D !important;
  color: #FFFFFF !important;
  font-weight: 800 !important;
  font-size: 0.95em !important;
  letter-spacing: 0 !important;
  padding: 10px 18px !important;
  cursor: pointer !important;
  box-shadow: 0 2px 0 rgba(0,0,0,.25), 0 0 0 1px rgba(255,255,255,.04) inset !important;
  transition: transform .12s ease, background .16s ease, border-color .16s ease, box-shadow .16s ease !important;
}
.gradio-container button:hover {
  background: #294466 !important;
  border-color: var(--primary) !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 8px 20px rgba(14,165,233,.18) !important;
}
.gradio-container button:focus-visible {
  outline: 3px solid rgba(56,189,248,.45) !important;
  outline-offset: 2px !important;
}
.gradio-container .btn-primary button,
.gradio-container button.btn-primary {
  background: linear-gradient(135deg, #7DD3FC, #38BDF8 48%, #0EA5E9) !important;
  color: var(--button-text) !important;
  border: 1px solid #BAE6FD !important;
}
.gradio-container .btn-primary button:hover,
.gradio-container button.btn-primary:hover {
  background: linear-gradient(135deg, #BAE6FD, #38BDF8 48%, #0284C7) !important;
}
.gradio-container .btn-secondary button,
.gradio-container button.btn-secondary {
  background: linear-gradient(135deg, #DDD6FE, #A78BFA 48%, #7C3AED) !important;
  color: #100724 !important;
  border: 1px solid #EDE9FE !important;
}

/* â”€â”€ Plot Containers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
.plot-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 4px;
  overflow: hidden;
}

/* â”€â”€ Chat â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
.chatbot-container {
  background: transparent !important;
  border: 0 !important;
  border-radius: 0 !important;
}
.chatbot-container .message,
.chatbot-container [data-testid="bot"],
.chatbot-container [data-testid="user"] {
  font-family: "Segoe UI", Roboto, Arial, sans-serif !important;
  font-size: 18px !important;
  line-height: 1.65 !important;
}
.message.user-message {
  background: rgba(56,189,248,0.16) !important;
  border-radius: var(--radius-sm) !important;
  color: #F8FAFC !important;
}
.message.bot-message {
  background: rgba(139,92,246,0.14) !important;
  border-radius: var(--radius-sm) !important;
  color: #F8FAFC !important;
}

/* Focused service-advisor workspace */
.advisor-shell { max-width: 1180px; margin: 0 auto; padding: 28px 18px 42px; }
.advisor-hero { padding: 20px 4px 24px; }
.advisor-kicker { color:#38BDF8; font-size:13px; font-weight:800; letter-spacing:.12em; text-transform:uppercase; }
.advisor-hero h1 { font-size:34px; line-height:1.15; margin:7px 0 10px; color:#F8FAFC; }
.advisor-hero p { max-width:650px; margin:0; font-size:18px; color:#A8B3C7; line-height:1.55; }
.advisor-chat-panel { background:#0F1B2D; border:1px solid #2C3D55; border-radius:16px; overflow:hidden; box-shadow:0 18px 42px rgba(0,0,0,.22); }
.advisor-chat-head { display:flex; align-items:center; gap:10px; padding:16px 20px; border-bottom:1px solid #2C3D55; color:#E2E8F0; font-weight:800; font-size:18px; }
.advisor-dot { width:10px; height:10px; border-radius:99px; background:#34D399; box-shadow:0 0 0 4px rgba(52,211,153,.12); }
.advisor-side { display:grid; gap:14px; }
.advisor-side-card { background:#0F1B2D; border:1px solid #2C3D55; border-radius:14px; padding:20px; }
.advisor-side-card h3 { color:#F8FAFC; font-size:17px; margin:0 0 10px; }
.advisor-side-card p, .advisor-side-card li { color:#A8B3C7; font-size:16px; line-height:1.6; }
.advisor-side-card ul { margin:0; padding-left:19px; }
.advisor-composer { padding:14px; background:#0F1B2D; border:1px solid #2C3D55; border-radius:14px; margin-top:14px; }
.advisor-composer textarea, .advisor-composer input { font-size:18px !important; }
.advisor-reset button { min-height:auto !important; background:transparent !important; border:0 !important; color:#94A3B8 !important; padding:5px 0 !important; font-size:14px !important; box-shadow:none !important; }
@media (max-width: 800px) { .advisor-shell { padding:18px 12px 28px; } .advisor-hero h1 { font-size:28px; } .advisor-hero p { font-size:16px; } }

/* â”€â”€ Status Bar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
.status-bar {
  background: var(--bg-card);
  border: 1px solid #334761;
  border-radius: var(--radius-sm);
  padding: 12px 16px;
  font-size: 0.92em;
  color: var(--text-main);
  font-weight: 700;
}

/* â”€â”€ Info Box â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
.info-box {
  background: rgba(56,189,248,0.08);
  border: 1px solid rgba(125,211,252,0.28);
  border-radius: var(--radius-sm);
  padding: 15px 17px;
  color: var(--text-main);
  font-size: 0.92em;
  line-height: 1.7;
}

/* Gradio overrides */
.gr-box,
.gr-panel,
.block,
.form {
  background: var(--bg-card) !important;
  border-color: #2C3D55 !important;
}
.gradio-container .wrap,
.gradio-container .contain {
  gap: 14px !important;
}
.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
  color: #94A3B8 !important;
  opacity: 1 !important;
}
footer { display: none !important; }
.progress-bar { background: var(--primary) !important; }

@media (max-width: 768px) {
  .header-banner { padding: 18px 16px 16px; }
  .header-title { font-size: 1.45em; }
  .header-sub { font-size: 0.88em; }
  .tab-nav button { padding: 10px 12px !important; font-size: 0.85em !important; }
  .gradio-container button { width: 100% !important; }
}
"""


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# GRADIO UI
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def build_legacy_ui():
    with gr.Blocks(
        css=CUSTOM_CSS,
        title="Vehicle Maintenance AI Platform",
        theme=gr.themes.Base(
            primary_hue="cyan",
            secondary_hue="purple",
            neutral_hue="slate",
        )
    ) as demo:

        # â”€â”€ Header â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        gr.HTML("""
        <div class="header-banner">
          <div class="header-title">Predictive Vehicle Maintenance Platform</div>
          <div class="header-sub">
            AI-Powered Diagnostics &nbsp;Â·&nbsp; SHAP Explainability &nbsp;Â·&nbsp;
            Health-Based RUL Estimation &nbsp;Â·&nbsp; Conversational AI Assistant
          </div>
        </div>
        """)

        # â”€â”€ System Status â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        with gr.Row():
            status_box = gr.HTML(
                '<div class="status-bar">â³ Initialising models... please wait.</div>'
            )

        # â”€â”€ Tabs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        with gr.Tabs(elem_classes="tab-nav") as tabs:

            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            # TAB 1: Dashboard Overview
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            with gr.TabItem("Dashboard"):
                gr.HTML('<div class="section-heading">Platform Overview</div>')
                with gr.Row():
                    gr.HTML("""
                    <div class="stat-card">
                      <div class="metric-label">Model</div>
                      <div class="metric-value" style="font-size:1.2em;">Random Forest</div>
                      <div style="color:#64748B;font-size:0.75em;margin-top:4px;">150 Trees Â· Max Depth 12</div>
                    </div>""")
                    gr.HTML("""
                    <div class="stat-card">
                      <div class="metric-label">Dataset Size</div>
                      <div class="metric-value">50K</div>
                      <div style="color:#64748B;font-size:0.75em;margin-top:4px;">Vehicle Records</div>
                    </div>""")
                    gr.HTML("""
                    <div class="stat-card">
                      <div class="metric-label">Features</div>
                      <div class="metric-value">17</div>
                      <div style="color:#64748B;font-size:0.75em;margin-top:4px;">Engineered Inputs</div>
                    </div>""")
                    gr.HTML("""
                    <div class="stat-card">
                      <div class="metric-label">XAI Method</div>
                      <div class="metric-value" style="color:#A855F7;font-size:1.2em;">SHAP</div>
                      <div style="color:#64748B;font-size:0.75em;margin-top:4px;">TreeExplainer Â· Local + Global</div>
                    </div>""")

                gr.HTML("""
                <div class="info-box" style="margin-top:16px;">
                  <strong style="color:#00D4FF;">How to use this platform:</strong><br>
                  1. Go to <strong>Vehicle Prediction</strong> â†’ Fill in vehicle details â†’ Click Analyse<br>
                  2. View <strong>Explainable AI</strong> tab for SHAP feature impact<br>
                  3. View <strong>RUL Analysis</strong> tab for Remaining Useful Life estimate<br>
                  4. View <strong>Diagnostics</strong> tab for subsystem risk breakdown<br>
                  5. Use <strong>AI Assistant</strong> tab to ask questions about the analysis<br>
                  6. View <strong>System Insights</strong> tab for model performance metrics
                </div>""")

                gr.HTML("""
                <div class="info-box" style="margin-top:12px;border-color:rgba(123,47,190,0.3);">
                  <strong style="color:#A855F7;">Technical Notes:</strong><br>
                  Â· <strong>RUL Estimation</strong> is health-score-based (not IoT/sensor-based) and estimates
                    distance before maintenance risk becomes critical<br>
                  Â· <strong>SHAP</strong> explains ML predictions only (maintenance YES/NO)<br>
                  Â· <strong>VHI</strong> is computed from weighted subsystem conditions
                </div>""")

            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            # TAB 2: Vehicle Prediction
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            with gr.TabItem("Vehicle Prediction"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML('<div class="section-heading">Vehicle Details</div>')
                        vehicle_model   = gr.Dropdown(
                            choices=["Car","Truck","Van","Bus","Motorcycle","SUV"],
                            value="Car", label="Vehicle Model")
                        fuel_type       = gr.Dropdown(
                            choices=["Petrol","Diesel","Electric","Hybrid"],
                            value="Petrol", label="Fuel Type")
                        transmission    = gr.Dropdown(
                            choices=["Automatic","Manual"],
                            value="Automatic", label="Transmission Type")
                        owner_type      = gr.Dropdown(
                            choices=["First","Second","Third"],
                            value="First", label="Owner Type")
                        vehicle_age     = gr.Slider(0, 20, value=5, step=1, label="Vehicle Age (years)")
                        engine_size     = gr.Slider(500, 5000, value=2000, step=100, label="Engine Size (cc)")
                        mileage         = gr.Slider(0, 200000, value=50000, step=1000, label="Annual Mileage (km)")

                    with gr.Column(scale=1):
                        gr.HTML('<div class="section-heading">Condition & History</div>')
                        brake_cond      = gr.Dropdown(
                            choices=["New","Good","Moderate","Worn","Worn Out","Critical"],
                            value="Good", label="Brake Condition")
                        tire_cond       = gr.Dropdown(
                            choices=["New","Good","Moderate","Worn","Worn Out","Critical"],
                            value="Good", label="Tire Condition")
                        battery_status  = gr.Dropdown(
                            choices=["New","Good","Moderate","Weak","Dead","Critical"],
                            value="Good", label="Battery Status")
                        maint_history   = gr.Dropdown(
                            choices=["Excellent","Good","Average","Poor"],
                            value="Average", label="Maintenance History")
                        service_history = gr.Dropdown(
                            choices=["Excellent","Good","Average","Poor"],
                            value="Good", label="Service History")
                        accident_hist   = gr.Slider(0, 5, value=0, step=1, label="Accident History (count)")
                        reported_issues = gr.Slider(0, 10, value=0, step=1, label="Reported Issues")

                    with gr.Column(scale=1):
                        gr.HTML('<div class="section-heading">Readings & Dates</div>')
                        odometer        = gr.Number(value=80000, label="Odometer Reading (km)", precision=0)
                        insurance       = gr.Number(value=15000, label="Insurance Premium (â‚¹/year)", precision=0)
                        fuel_eff        = gr.Number(value=15.0,  label="Fuel Efficiency (km/L)", precision=1)
                        last_service    = gr.Textbox(value="2023-06-15", label="Last Service Date (YYYY-MM-DD)")
                        warranty_expiry = gr.Textbox(value="2025-12-31", label="Warranty Expiry Date (YYYY-MM-DD)")

                        gr.HTML("<br>")
                        analyse_btn = gr.Button("Analyze Vehicle", elem_classes="btn-primary")

                # â”€â”€ Output Row â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                gr.HTML('<div class="section-heading" style="margin-top:20px;">Analysis Results</div>')
                summary_html = gr.HTML('<div class="info-box">Run analysis to see results.</div>')

                with gr.Row():
                    vhi_plot = gr.Plot(label="Vehicle Health Index", elem_classes="plot-card")
                    rul_plot = gr.Plot(label="Remaining Useful Life", elem_classes="plot-card")
                    risk_plot= gr.Plot(label="Subsystem Risk",        elem_classes="plot-card")

                # Hidden outputs fed to other tabs
                shap_output_hidden    = gr.State(value=None)
                shap_text_hidden      = gr.State(value="")
                rul_text_hidden       = gr.State(value="")
                chat_status_hidden    = gr.State(value="")

            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            # TAB 3: Explainable AI (SHAP)
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            with gr.TabItem("Explainable AI"):
                gr.HTML('<div class="section-heading">SHAP Feature Explainability</div>')
                gr.HTML("""
                <div class="info-box" style="margin-bottom:14px;">
                  <strong style="color:#00D4FF;">About SHAP:</strong> SHAP (SHapley Additive exPlanations) 
                  shows which features most influenced the maintenance prediction. 
                  <span style="color:#FF3D71;">Red bars</span> push the prediction toward maintenance needed. 
                  <span style="color:#00E676;">Green bars</span> push against maintenance.
                  Run a Vehicle Prediction first to see results here.
                </div>""")
                shap_plot_display = gr.Plot(label="SHAP Waterfall â€” Local Explanation",
                                            elem_classes="plot-card")
                shap_text_display = gr.Markdown("*Run a vehicle prediction first.*")

                gr.HTML('<div class="section-heading" style="margin-top:16px;">Global Feature Importance</div>')
                fi_plot = gr.Plot(label="Random Forest Feature Importance", elem_classes="plot-card")

                load_fi_btn = gr.Button("Load Global Feature Importance", elem_classes="btn-secondary")

                def load_fi():
                    model = get("model")
                    feat_names = get("feature_names")
                    if model and feat_names:
                        fi_n, fi_v = get_feature_importance_data(model, feat_names)
                        return plot_feature_importance(fi_n, fi_v)
                    return None

                load_fi_btn.click(load_fi, outputs=fi_plot)

            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            # TAB 4: RUL Analysis
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            with gr.TabItem("RUL Analysis"):
                gr.HTML('<div class="section-heading">Remaining Useful Life Analysis</div>')
                gr.HTML("""
                <div class="info-box" style="margin-bottom:14px;">
                  <strong style="color:#A855F7;">RUL Method:</strong> Health-based estimate of remaining distance before maintenance risk becomes critical.
                  This is not total vehicle lifespan; it is a maintenance-risk horizon based on VHI, subsystem health, odometer load, reported issues, accident history, and service history.
                  The estimate explains when maintenance risk may become critical, not how long the vehicle can exist mechanically.
                </div>""")
                with gr.Row():
                    rul_gauge_display = gr.Plot(label="RUL Before Critical Risk", elem_classes="plot-card")
                    vhi_gauge_display = gr.Plot(label="VHI Gauge", elem_classes="plot-card")
                rul_interpretation = gr.Markdown("*Run a vehicle prediction to see RUL analysis.*")

            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            # TAB 5: Diagnostics
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            with gr.TabItem("Diagnostics"):
                gr.HTML('<div class="section-heading">Subsystem Diagnostic Report</div>')
                gr.HTML("""
                <div class="info-box" style="margin-bottom:14px;">
                  Rule-based diagnostic engine analyzes each vehicle subsystem independently
                  based on reported conditions, odometer, and maintenance history.
                </div>""")
                diag_risk_plot = gr.Plot(label="Component Risk Scores", elem_classes="plot-card")
                diag_report    = gr.HTML('<div class="info-box">Run a vehicle prediction to see diagnostics.</div>')

            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            # TAB 6: AI Assistant
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            with gr.TabItem("AI Assistant"):
                gr.HTML('<div class="section-heading">Conversational AI Diagnostics Assistant</div>')

                api_status_html = (
                    '<div class="info-box" style="border-color:rgba(0,230,118,0.3);">'
                    'âœ… AI Assistant: API key configured â€” ready to chat.</div>'
                    if api_available() else
                    '<div class="info-box" style="border-color:rgba(255,61,113,0.3);">'
                    'âš ï¸ AI Assistant: No API key found. Add <code>GROQ_API_KEY</code> to <code>.env</code>. '
                    'Get a free key at <a href="https://console.groq.com" target="_blank" '
                    'style="color:#00D4FF;">console.groq.com</a></div>'
                )
                gr.HTML(api_status_html)

                chatbot = gr.Chatbot(
                    height=420,
                    elem_classes="chatbot-container",
                    label="VehicleAI Assistant",
                    layout="bubble",
                )
                with gr.Row():
                    chat_input = gr.Textbox(
                        placeholder="Ask about your vehicle analysis, maintenance tips, RUL, SHAP results...",
                        label="Your Message",
                        scale=5,
                        lines=1,
                    )
                    send_btn = gr.Button("Send", elem_classes="btn-primary", scale=1)

                chat_status = gr.Textbox(
                    label="Context Status",
                    value="No vehicle analysed yet. Run a prediction for context-aware responses.",
                    interactive=False,
                )

                # Quick prompts
                gr.HTML('<div style="color:#64748B;font-size:0.78em;margin-top:8px;">Quick prompts:</div>')
                with gr.Row():
                    qp1 = gr.Button("What does my VHI score mean?", size="sm")
                    qp2 = gr.Button("Explain the SHAP results",       size="sm")
                    qp3 = gr.Button("What should I do about my RUL?", size="sm")
                    qp4 = gr.Button("Tips to improve battery life",    size="sm")

                def send_message(msg, history):
                    return chat_with_agent(msg, history)

                send_btn.click(send_message,   inputs=[chat_input, chatbot], outputs=[chatbot, chat_input])
                chat_input.submit(send_message, inputs=[chat_input, chatbot], outputs=[chatbot, chat_input])
                qp1.click(lambda h: chat_with_agent("What does my VHI score mean?", h), inputs=chatbot, outputs=[chatbot, chat_input])
                qp2.click(lambda h: chat_with_agent("Explain the SHAP results in simple terms", h), inputs=chatbot, outputs=[chatbot, chat_input])
                qp3.click(lambda h: chat_with_agent("What should I do about my RUL estimate?", h), inputs=chatbot, outputs=[chatbot, chat_input])
                qp4.click(lambda h: chat_with_agent("Give me tips to improve battery life", h), inputs=chatbot, outputs=[chatbot, chat_input])

            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            # TAB 7: System Insights
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            with gr.TabItem("System Insights"):
                gr.HTML('<div class="section-heading">Model Performance & Evaluation</div>')
                load_insights_btn = gr.Button("Load Model Insights", elem_classes="btn-primary")
                metrics_report    = gr.Markdown("*Click above to load model performance metrics.*")

                with gr.Row():
                    cm_plot      = gr.Plot(label="Confusion Matrix",       elem_classes="plot-card")
                    metrics_plot = gr.Plot(label="Performance Metrics",    elem_classes="plot-card")

                with gr.Row():
                    roc_plot     = gr.Plot(label="ROC Curve",              elem_classes="plot-card")
                    fi_sys_plot  = gr.Plot(label="Feature Importance",     elem_classes="plot-card")

                def load_insights():
                    cm, mp, rp, fp, report = get_system_insights()
                    return cm, mp, rp, fp, report

                load_insights_btn.click(
                    load_insights,
                    outputs=[cm_plot, metrics_plot, roc_plot, fi_sys_plot, metrics_report]
                )

        # â”€â”€ Wire up the main Analyse button â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        def full_analysis(*args):
            (summary, vhi_fig, rul_fig, risk_fig,
             shap_fig, shap_txt, rul_txt, chat_upd, diag_txt) = run_prediction(*args)

            # Derive row for diagnostic display
            return (
                summary,
                vhi_fig,
                rul_fig,
                risk_fig,
                # XAI tab
                shap_fig,
                f"**SHAP Explanation:** {shap_txt}" if shap_txt else "*No SHAP data.*",
                # RUL tab
                rul_fig,
                vhi_fig,
                f"**RUL Interpretation:** {rul_txt}" if rul_txt else "*No RUL data.*",
                # Diagnostics tab
                risk_fig,
                diag_txt,
                # Chat status
                chat_upd,
            )

        analyse_btn.click(
            full_analysis,
            inputs=[
                vehicle_model, mileage, maint_history, reported_issues,
                vehicle_age, fuel_type, transmission, engine_size,
                odometer, last_service, warranty_expiry,
                owner_type, insurance, service_history, accident_hist,
                fuel_eff, tire_cond, brake_cond, battery_status,
            ],
            outputs=[
                summary_html,
                vhi_plot, rul_plot, risk_plot,
                shap_plot_display, shap_text_display,
                rul_gauge_display, vhi_gauge_display, rul_interpretation,
                diag_risk_plot, diag_report,
                chat_status,
            ]
        )

        # â”€â”€ On Load: Initialise â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        def on_load():
            ok, msg = initialise_system()
            if ok:
                return f'<div class="status-bar">âœ… System Ready â€” {msg}</div>'
            else:
                return f'<div class="status-bar" style="border-color:rgba(255,61,113,0.4);color:#FF3D71;">âŒ {msg}</div>'

        demo.load(on_load, outputs=status_box)

    return demo


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def build_ui():
    """Chat-first UI. The agent collects only the details it needs."""
    with gr.Blocks(
        css=CUSTOM_CSS,
        title="AI Vehicle Service Advisor",
        theme=gr.themes.Base(primary_hue="cyan", secondary_hue="purple", neutral_hue="slate"),
    ) as demo:
        gr.HTML("""
        <div class="advisor-shell advisor-hero">
          <div class="advisor-kicker">Vehicle care, explained clearly</div>
          <h1>Your personal service advisor</h1>
          <p>Tell me what your vehicle is doing. I’ll help you understand what may be happening and what to do next.</p>
        </div>
        """)
        with gr.Column(elem_classes="advisor-shell"):
            status_box = gr.HTML('<div class="status-bar">Preparing your advisor…</div>')
            memory = gr.State(value=None)
            with gr.Row(equal_height=False):
                with gr.Column(scale=8, min_width=480):
                    with gr.Column(elem_classes="advisor-chat-panel"):
                        gr.HTML('<div class="advisor-chat-head"><span class="advisor-dot"></span>Service Advisor</div>')
                        chatbot = gr.Chatbot(
                            value=[{"role": "assistant", "content": "Hi — what have you noticed with your vehicle today? I’ll help you make sense of it."}],
                            height=570, show_label=False, layout="bubble", elem_classes="chatbot-container",
                        )
                    with gr.Column(elem_classes="advisor-composer"):
                        with gr.Row():
                            message = gr.Textbox(show_label=False, lines=1, scale=6,
                                placeholder="Describe the sound, warning, performance issue, or anything else you notice…")
                            send = gr.Button("Send", elem_classes="btn-primary", scale=1)
                        reset = gr.Button("Start a new conversation", elem_classes="advisor-reset")
                with gr.Column(scale=4, min_width=280, elem_classes="advisor-side"):
                    assessment_card = gr.HTML('<div class="advisor-side-card"><h3>Current assessment</h3><p>Your latest service guidance will appear here.</p></div>')
                    gr.HTML("""
                    <div class="advisor-side-card">
                      <h3>How I can help</h3>
                      <ul>
                        <li>Explain unusual sounds and warning signs</li>
                        <li>Suggest sensible next checks</li>
                        <li>Help you decide how urgently to visit a service centre</li>
                      </ul>
                    </div>
                    """)

        def render_assessment_card(session_memory):
            result = (session_memory or {}).get("last_result")
            if not result:
                return '<div class="advisor-side-card"><h3>Current assessment</h3><p>Your latest service guidance will appear here.</p></div>'
            risk = html.escape(str(result.get("risk_level", "Unknown")).title())
            confidence = float(result.get("confidence", 0))
            colour = "#FF3D71" if risk in {"High", "Critical"} else "#FACC15" if risk == "Moderate" else "#00E676"
            return f'<div class="advisor-side-card" style="border-left:4px solid {colour};"><h3 style="color:{colour};">{risk} attention level</h3><p style="margin:0;">The model has {confidence:.0f}% confidence in the current assessment. Keep sharing what you notice and I’ll refine the advice.</p></div>'

        def respond(user_message, history, session_memory):
            text = (user_message or "").strip()
            if not text:
                return history, "", session_memory, render_assessment_card(session_memory)
            history = list(history or [])
            reply, session_memory = handle_message(text, session_memory)
            history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": reply}])
            return history, "", session_memory, render_assessment_card(session_memory)

        def reset_session():
            return ([{"role": "assistant", "content": "New assessment started. What is happening with your vehicle?"}], None, "", render_assessment_card(None))

        send.click(respond, [message, chatbot, memory], [chatbot, message, memory, assessment_card], show_progress="full")
        message.submit(respond, [message, chatbot, memory], [chatbot, message, memory, assessment_card], show_progress="full")
        reset.click(reset_session, outputs=[chatbot, memory, message, assessment_card])

        def on_load():
            ok, msg = initialise_system()
            if ok:
                return '<div class="status-bar">● Advisor ready to help</div>'
            return f'<div class="status-bar" style="border-color:rgba(255,61,113,.4);color:#FF3D71;">❌ {msg}</div>'

        demo.load(on_load, outputs=status_box)
    from fastapi.responses import JSONResponse

    def health_gemini():
        report = gemini_health_check()
        return JSONResponse(content=report, status_code=200 if report.get("ok") else 503)

    demo.app.add_api_route("/health/gemini", health_gemini, methods=["GET"], include_in_schema=False)
    return demo


if __name__ == "__main__":
    print("=" * 60)
    print("  Vehicle Maintenance AI Platform")
    print("=" * 60)

    # Pre-initialise before launching UI
    ok, msg = initialise_system()
    print(f"[Startup] {msg}")

    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False,
    )
