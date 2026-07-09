"""
utils.py
Shared utility functions for the Vehicle Maintenance Platform.
"""

import json
import os
from datetime import datetime


RISK_COLORS = {
    "LOW": "#00E676",
    "MODERATE": "#FFD600",
    "HIGH": "#FF9800",
    "CRITICAL": "#FF3D71",
}


def format_number(n, suffix=""):
    """Format a number with commas."""
    try:
        return f"{int(n):,}{suffix}"
    except (TypeError, ValueError):
        return str(n)


def health_color(score: float) -> str:
    if score >= 80:
        return "#00E676"
    if score >= 55:
        return "#FFD600"
    if score >= 35:
        return "#FF9800"
    return "#FF3D71"


def health_label(score: float) -> str:
    if score >= 80:
        return "Healthy"
    if score >= 55:
        return "Monitor"
    if score >= 35:
        return "Degraded"
    return "Critical"


def risk_badge(level: str) -> str:
    """Return HTML badge for risk level."""
    level = (level or "UNKNOWN").upper()
    colour = RISK_COLORS.get(level, "#9E9E9E")
    return (
        f'<span style="display:inline-flex;align-items:center;gap:6px;color:{colour};'
        f'font-weight:800;">{level}</span>'
    )


def condition_badge(label: str) -> str:
    """Return HTML badge for condition label."""
    colours = {
        "Healthy": "#00E676",
        "Excellent": "#00D4FF",
        "Good": "#00E676",
        "Moderate": "#FFD600",
        "High Risk": "#FF9800",
        "Critical": "#FF3D71",
        "Immediate Maintenance Required": "#FF3D71",
    }
    colour = colours.get(label, "#9E9E9E")
    return f'<span style="color:{colour};font-weight:bold;">{label}</span>'


def maintenance_badge(prediction: int, confidence: float, force_required: bool = False) -> str:
    """Return HTML badge for maintenance prediction."""
    if prediction == 1 or force_required:
        return (
            f'<div style="color:#FF3D71;font-size:1.4em;font-weight:bold;">'
            f'MAINTENANCE REQUIRED ({confidence:.1f}% confidence)</div>'
        )
    return (
        f'<div style="color:#00E676;font-size:1.4em;font-weight:bold;">'
        f'NO IMMEDIATE MAINTENANCE FLAG ({confidence:.1f}% confidence)</div>'
    )


def format_rul(rul_km, rul_label="") -> str:
    if rul_label == "Immediate Maintenance Required" or int(rul_km or 0) <= 500:
        return "Immediate Maintenance Required"
    return format_number(rul_km, " km")


def _card(label, value, accent, subtext=""):
    return f"""
    <div class="metric-card" style="background:#0F172A;border:1px solid #1E293B;
        border-radius:10px;padding:16px;border-left:4px solid {accent};min-height:118px;">
      <div style="font-size:0.72em;color:#94A3B8;font-weight:700;letter-spacing:.08em;
          text-transform:uppercase;margin-bottom:8px;">{label}</div>
      <div style="color:{accent};font-weight:800;font-size:1.35em;line-height:1.25;">{value}</div>
      <div style="color:#64748B;font-size:0.78em;line-height:1.5;margin-top:8px;">{subtext}</div>
    </div>
    """


def _health_bar(name, data):
    score = float(data.get("score", 0))
    condition = data.get("condition", "N/A")
    colour = health_color(score)
    label = health_label(score)
    warning = (
        '<div style="margin-top:8px;color:#FF3D71;font-size:0.76em;font-weight:700;">'
        'Critical component attention required</div>'
        if score < 35
        else ""
    )
    return f"""
    <div style="background:#0B1220;border:1px solid #1E293B;border-radius:10px;padding:14px;">
      <div style="display:flex;justify-content:space-between;gap:12px;margin-bottom:8px;">
        <div style="color:#E2E8F0;font-weight:700;">{name}</div>
        <div style="color:{colour};font-weight:800;">{score:.0f}%</div>
      </div>
      <div style="height:9px;background:#1E293B;border-radius:99px;overflow:hidden;">
        <div style="height:9px;width:{max(0, min(100, score)):.0f}%;background:{colour};border-radius:99px;"></div>
      </div>
      <div style="display:flex;justify-content:space-between;gap:12px;margin-top:8px;
          color:#94A3B8;font-size:0.78em;">
        <span>{label}</span><span>Condition: {condition}</span>
      </div>
      {warning}
    </div>
    """


def build_component_health_html(component_health: dict) -> str:
    if not component_health:
        return '<div class="info-box">Component health data unavailable.</div>'
    return "".join(_health_bar(name, data) for name, data in component_health.items())


def get_failed_failing_components(component_health: dict) -> dict:
    """Classify core vehicle components into failed and failing groups."""
    failed = []
    failing = []
    degraded = []

    for name, data in (component_health or {}).items():
        score = float(data.get("score", 0))
        condition = data.get("condition", "N/A")
        item = {
            "name": name.replace(" Health", ""),
            "score": score,
            "condition": condition,
        }
        if score <= 20:
            failed.append(item)
        elif score <= 40:
            failing.append(item)
        elif score < 55:
            degraded.append(item)

    return {"failed": failed, "failing": failing, "degraded": degraded}


def _component_status_chip(item, status, colour):
    return f"""
    <div style="background:rgba(255,255,255,.03);border:1px solid {colour};
        border-radius:10px;padding:12px;">
      <div style="display:flex;justify-content:space-between;gap:10px;align-items:center;">
        <div style="color:#E2E8F0;font-weight:800;">{item['name']}</div>
        <div style="color:{colour};font-weight:900;font-size:.82em;">{status}</div>
      </div>
      <div style="color:#94A3B8;font-size:.8em;margin-top:6px;">
        Health: <span style="color:{colour};font-weight:800;">{item['score']:.0f}%</span>
        &nbsp;|&nbsp; Condition: {item['condition']}
      </div>
    </div>
    """


def build_failed_components_html(component_health: dict) -> str:
    """Render explicit failed/failing component callouts."""
    status = get_failed_failing_components(component_health)
    failed = status["failed"]
    failing = status["failing"]
    degraded = status["degraded"]

    if not failed and not failing and not degraded:
        return """
        <div style="background:#0F172A;border:1px solid #1E293B;border-left:4px solid #00E676;
            border-radius:10px;padding:14px;color:#CBD5E1;">
          <div style="color:#94A3B8;font-size:.75em;font-weight:800;text-transform:uppercase;margin-bottom:6px;">
            Failed / Failing Components
          </div>
          No failed or failing core components detected. Continue scheduled inspection intervals.
        </div>
        """

    failed_html = "".join(_component_status_chip(item, "FAILED / CRITICAL", "#FF3D71") for item in failed)
    failing_html = "".join(_component_status_chip(item, "FAILING", "#FF9800") for item in failing)
    degraded_html = "".join(_component_status_chip(item, "DEGRADED", "#FFD600") for item in degraded)

    headline = []
    if failed:
        headline.append(f"{len(failed)} failed/critical")
    if failing:
        headline.append(f"{len(failing)} failing")
    if degraded:
        headline.append(f"{len(degraded)} degraded")
    headline_text = ", ".join(headline)

    return f"""
    <div style="background:#0F172A;border:1px solid #1E293B;border-left:4px solid #FF3D71;
        border-radius:10px;padding:16px;">
      <div style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:12px;">
        <div style="color:#94A3B8;font-size:.75em;font-weight:800;text-transform:uppercase;">
          Failed / Failing Components
        </div>
        <div style="color:#FF3D71;font-weight:900;font-size:.86em;">{headline_text}</div>
      </div>
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px;">
        {failed_html}{failing_html}{degraded_html}
      </div>
    </div>
    """


def _key_factors_html(factors):
    if not factors:
        return "<li>No major RUL drivers available.</li>"
    return "".join(f"<li>{factor}</li>" for factor in factors)


def _recommendation_text(prediction, risk_level, rul_label, recommendation):
    if risk_level == "CRITICAL" or rul_label == "Immediate Maintenance Required":
        return (
            "Immediate Maintenance Required. "
            f"{recommendation}"
        )
    if risk_level == "HIGH" or prediction == 1:
        return f"Maintenance recommended before extended operation. {recommendation}"
    return f"Continue scheduled maintenance and monitor weak subsystems. {recommendation}"


def build_diagnostic_report_html(diag: dict, rul_result: dict) -> str:
    """Build the Diagnostics tab report with subsystem health and RUL reasoning."""
    component_health = rul_result.get("component_health", {})
    factors = rul_result.get("key_factors", [])
    risk_level = rul_result.get("rul_risk_level", diag.get("overall_risk_level", "UNKNOWN"))
    risk_color = RISK_COLORS.get(risk_level, "#9E9E9E")

    return f"""
<div style="display:grid;gap:14px;">
  <div style="background:#0F172A;border:1px solid #1E293B;border-left:4px solid {risk_color};
      border-radius:10px;padding:16px;">
    <div style="color:#94A3B8;font-size:.75em;font-weight:700;text-transform:uppercase;">
      Diagnostic Interpretation
    </div>
    <div style="color:#E2E8F0;font-size:1em;line-height:1.65;margin-top:8px;">
      {rul_result.get("interpretation", "Run a vehicle prediction to see diagnostics.")}
    </div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;">
    {build_component_health_html(component_health)}
  </div>
  {build_failed_components_html(component_health)}
  <div style="background:#0B1220;border:1px solid #1E293B;border-radius:10px;padding:16px;">
    <div style="color:#94A3B8;font-size:.75em;font-weight:700;text-transform:uppercase;margin-bottom:8px;">
      Key Factors Affecting RUL
    </div>
    <ul style="margin:0;padding-left:18px;color:#CBD5E1;line-height:1.7;">
      {_key_factors_html(factors)}
    </ul>
  </div>
</div>
"""


def build_summary_html(
    prediction,
    confidence,
    vhi,
    rul_km,
    rul_label,
    most_vulnerable,
    risk_level,
    shap_text,
    recommendation,
    component_health=None,
    key_factors=None,
    rul_interpretation="",
):
    """Build a production-style HTML diagnostic summary."""
    component_health = component_health or {}
    key_factors = key_factors or []

    rul_text = format_rul(rul_km, rul_label)
    forced_maintenance = (
        risk_level == "CRITICAL"
        or rul_label == "Immediate Maintenance Required"
        or int(rul_km or 0) <= 500
        or float(vhi or 0) < 40
    )
    pred_color = "#FF3D71" if prediction == 1 or forced_maintenance else "#00E676"
    pred_label = (
        "MAINTENANCE REQUIRED"
        if prediction == 1 or forced_maintenance
        else "NO IMMEDIATE MAINTENANCE FLAG"
    )
    risk_color = RISK_COLORS.get(risk_level, "#9E9E9E")
    recommendation_text = _recommendation_text(prediction, risk_level, rul_label, recommendation)

    return f"""
<div style="background:#0B1220;border:1px solid #1E293B;border-radius:14px;padding:22px;
  font-family:'Inter',sans-serif;color:#E2E8F0;box-shadow:0 10px 30px rgba(0,0,0,.32);">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:16px;
      flex-wrap:wrap;margin-bottom:18px;">
    <div>
      <div style="color:#00D4FF;font-weight:800;font-size:1.15em;letter-spacing:.04em;">
        Vehicle Diagnostic Dashboard
      </div>
      <div style="color:#94A3B8;font-size:.86em;margin-top:6px;">
        RUL estimates distance remaining before maintenance risk becomes critical.
      </div>
    </div>
    <div style="color:{pred_color};font-weight:900;border:1px solid {pred_color};
        border-radius:999px;padding:8px 12px;background:rgba(255,255,255,.03);">
      {pred_label}
    </div>
  </div>

  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin-bottom:16px;">
    {_card("Vehicle Health Index", f"{vhi:.1f}%", health_color(vhi), "Weighted health score from odometer, subsystem condition, and service history.")}
    {_card("RUL Before Critical Risk", rul_text, "#A855F7", f"Risk state: {condition_badge(rul_label)}")}
    {_card("Risk Level", risk_badge(risk_level), risk_color, f"Model confidence: {confidence:.1f}%")}
    {_card("Weakest Component", most_vulnerable, risk_color, "This subsystem has the lowest health score and drives the recommendation.")}
  </div>

  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;margin-bottom:16px;">
    {build_component_health_html(component_health)}
  </div>

  <div style="margin-bottom:16px;">
    {build_failed_components_html(component_health)}
  </div>

  <div style="display:grid;grid-template-columns:1.1fr .9fr;gap:12px;">
    <div style="background:#0F172A;border:1px solid #1E293B;border-radius:10px;padding:16px;
        border-left:4px solid #00D4FF;">
      <div style="font-size:.75em;color:#94A3B8;font-weight:800;text-transform:uppercase;margin-bottom:8px;">
        Professional Interpretation
      </div>
      <div style="color:#CBD5E1;font-size:.92em;line-height:1.7;">
        {rul_interpretation or shap_text}
      </div>
      <div style="color:#94A3B8;font-size:.78em;line-height:1.6;margin-top:10px;">
        SHAP summary: {shap_text}
      </div>
    </div>

    <div style="background:#0F172A;border:1px solid #1E293B;border-radius:10px;padding:16px;
        border-left:4px solid {pred_color};">
      <div style="font-size:.75em;color:#94A3B8;font-weight:800;text-transform:uppercase;margin-bottom:8px;">
        Maintenance Recommendation
      </div>
      <div style="color:#CBD5E1;font-size:.92em;line-height:1.7;margin-bottom:12px;">
        {recommendation_text}
      </div>
      <div style="font-size:.75em;color:#94A3B8;font-weight:800;text-transform:uppercase;margin-bottom:8px;">
        Key Factors Affecting RUL
      </div>
      <ul style="margin:0;padding-left:18px;color:#CBD5E1;line-height:1.65;font-size:.88em;">
        {_key_factors_html(key_factors)}
      </ul>
    </div>
  </div>
</div>
"""


def save_analysis_log(data: dict, log_dir: str = None):
    """Save analysis result to a JSON log file."""
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(__file__), "assets", "logs")
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(log_dir, f"analysis_{ts}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return path
