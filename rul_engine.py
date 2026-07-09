"""
rul_engine.py
Maintenance-window Remaining Useful Life (RUL) estimation.

RUL in this project means estimated remaining distance before maintenance risk
becomes critical. It is bounded to 0-10,000 km and must not be interpreted as
total vehicle lifespan, end-of-life prediction, or survivability.
"""

MAX_MAINTENANCE_RUL_KM = 10_000

BRAKE_SCORES = {"new": 100, "good": 75, "moderate": 55, "worn": 35, "worn out": 35, "critical": 10}
TIRE_SCORES = {"new": 100, "good": 75, "moderate": 55, "worn": 35, "worn out": 30, "critical": 10}
BATTERY_SCORES = {"new": 100, "good": 75, "moderate": 55, "weak": 40, "dead": 5, "critical": 5}
MAINT_SCORES = {"excellent": 100, "good": 75, "average": 65, "poor": 30, "unknown": 50}
SERVICE_SCORES = {"excellent": 100, "good": 75, "average": 65, "poor": 30, "unknown": 50}
ACCIDENT_MAP = {0: 100, 1: 75, 2: 55, 3: 35}

RUL_WINDOWS = {
    "Excellent": (8_000, 10_000),
    "Good": (5_000, 7_000),
    "Moderate": (3_000, 5_000),
    "Poor": (1_000, 3_000),
    "Critical": (0, 1_000),
}


def _s(mapping, key, default=50):
    return mapping.get(str(key).lower().strip(), default)


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def get_component_health(row: dict) -> dict:
    """Return the core subsystem health scores used by the RUL model."""
    return {
        "Brake Health": {
            "score": _s(BRAKE_SCORES, row.get("Brake_Condition", ""), 50),
            "condition": row.get("Brake_Condition", "N/A"),
        },
        "Tire Health": {
            "score": _s(TIRE_SCORES, row.get("Tire_Condition", ""), 50),
            "condition": row.get("Tire_Condition", "N/A"),
        },
        "Battery Health": {
            "score": _s(BATTERY_SCORES, row.get("Battery_Status", ""), 50),
            "condition": row.get("Battery_Status", "N/A"),
        },
    }


def _mileage_health(odometer: float) -> float:
    """Strong odometer penalty used for health and maintenance-window sizing."""
    if odometer >= 240_000:
        return 10.0
    if odometer >= 200_000:
        return 22.0
    if odometer >= 160_000:
        return 38.0
    if odometer >= 120_000:
        return 55.0
    if odometer >= 80_000:
        return 70.0
    if odometer >= 40_000:
        return 84.0
    return 95.0


def compute_vhi(row: dict) -> float:
    """
    Compute Vehicle Health Index (VHI) from 0-100.
    Odometer and combined subsystem wear are intentionally weighted strongly.
    """
    odometer = _safe_float(row.get("Odometer_Reading", 0))
    mileage_health = _mileage_health(odometer)

    component_health = get_component_health(row)
    brake_score = component_health["Brake Health"]["score"]
    tire_score = component_health["Tire Health"]["score"]
    battery_score = component_health["Battery Health"]["score"]
    maint_score = _s(MAINT_SCORES, row.get("Maintenance_History", ""), 50)
    service_score = _s(SERVICE_SCORES, row.get("Service_History", ""), 50)
    accidents = _safe_int(row.get("Accident_History", 0))
    accident_score = ACCIDENT_MAP.get(min(accidents, 3), 35)

    vhi = (
        0.28 * mileage_health
        + 0.17 * brake_score
        + 0.17 * tire_score
        + 0.14 * battery_score
        + 0.10 * maint_score
        + 0.09 * service_score
        + 0.05 * accident_score
    )

    return round(_clamp(vhi, 0.0, 100.0), 2)


def _combined_degradation_score(row: dict, component_health: dict, vhi: float, odometer: float) -> float:
    """
    Higher score means healthier maintenance window.
    Uses combined wear, not a single best component, so one new subsystem cannot dominate RUL.
    """
    brake = component_health["Brake Health"]["score"]
    tire = component_health["Tire Health"]["score"]
    battery = component_health["Battery Health"]["score"]
    weakest = min(brake, tire, battery)
    average_subsystem = (brake + tire + battery) / 3.0
    maint = _s(MAINT_SCORES, row.get("Maintenance_History", ""), 50)
    service = _s(SERVICE_SCORES, row.get("Service_History", ""), 50)
    accidents = min(_safe_int(row.get("Accident_History", 0)), 3)
    issues = min(_safe_int(row.get("Reported_Issues", 0)), 10)

    score = (
        0.26 * vhi
        + 0.20 * _mileage_health(odometer)
        + 0.19 * average_subsystem
        + 0.14 * weakest
        + 0.08 * maint
        + 0.08 * service
        + 0.05 * ACCIDENT_MAP.get(accidents, 35)
    )

    # Strong maintenance scheduling penalties. These bound the output as a
    # service interval rather than a lifespan forecast.
    if odometer >= 240_000:
        score -= 22
    elif odometer >= 200_000:
        score -= 16
    elif odometer >= 160_000:
        score -= 10
    elif odometer >= 120_000:
        score -= 5

    score -= issues * 2.8
    score -= accidents * 3.5

    if weakest <= 20:
        score = min(score, 24)
    elif weakest <= 35:
        score = min(score, 42)

    return round(_clamp(score, 0.0, 100.0), 2)


def _tier_from_score(score: float, weakest_score: float, odometer: float, issues: int) -> str:
    if score < 18 or weakest_score <= 20 or odometer >= 260_000 or issues >= 8:
        return "Critical"
    if score < 45 or weakest_score <= 35 or odometer >= 220_000:
        return "Poor"
    if score < 62 or weakest_score <= 55 or odometer >= 160_000:
        return "Moderate"
    if score < 78 or odometer >= 100_000:
        return "Good"
    return "Excellent"


def _window_value(tier: str, score: float) -> int:
    low, high = RUL_WINDOWS[tier]
    tier_floor = {
        "Critical": 0,
        "Poor": 30,
        "Moderate": 45,
        "Good": 62,
        "Excellent": 78,
    }[tier]
    tier_ceiling = {
        "Critical": 18,
        "Poor": 45,
        "Moderate": 62,
        "Good": 78,
        "Excellent": 100,
    }[tier]
    ratio = (score - tier_floor) / max(1, tier_ceiling - tier_floor)
    return int(round(low + _clamp(ratio, 0.0, 1.0) * (high - low), -2))


def _risk_level_from_tier(tier: str) -> str:
    return {
        "Excellent": "LOW",
        "Good": "MODERATE",
        "Moderate": "HIGH",
        "Poor": "CRITICAL",
        "Critical": "CRITICAL",
    }[tier]


def _label_from_tier(tier: str, rul_km: int) -> str:
    if tier == "Critical" and rul_km <= 500:
        return "Immediate Maintenance Required"
    return {
        "Excellent": "Healthy",
        "Good": "Good",
        "Moderate": "Moderate",
        "Poor": "High Risk",
        "Critical": "Critical",
    }[tier]


def _build_key_factors(row: dict, component_health: dict, vhi: float, odometer: float, score: float) -> list:
    factors = []
    weakest_name, weakest = min(component_health.items(), key=lambda item: item[1]["score"])

    factors.append(
        f"{weakest_name} is the weakest subsystem at {weakest['score']:.0f}% "
        f"({weakest['condition']})."
    )
    factors.append(f"Combined degradation score is {score:.1f}/100, which sets the maintenance window.")

    if odometer >= 160_000:
        factors.append(f"High odometer reading ({int(odometer):,} km) strongly reduces RUL.")
    elif odometer >= 100_000:
        factors.append(f"Odometer reading ({int(odometer):,} km) reduces the next service window.")

    issues = _safe_int(row.get("Reported_Issues", 0))
    if issues > 0:
        factors.append(f"{issues} reported issue(s) increase near-term maintenance risk.")

    maintenance = str(row.get("Maintenance_History", "")).strip().lower()
    service = str(row.get("Service_History", "")).strip().lower()
    if maintenance in {"poor", "unknown"} or service in {"poor", "unknown"}:
        factors.append("Maintenance/service history reduces confidence in continued operation.")
    elif vhi >= 75:
        factors.append("Service and subsystem condition support a longer, but still capped, service window.")

    return factors[:5]


def compute_rul(row: dict) -> dict:
    """
    Compute bounded maintenance-oriented RUL before risk becomes critical.
    The output is always 0-10,000 km.
    """
    vhi = compute_vhi(row)
    odometer = _safe_float(row.get("Odometer_Reading", 0))
    component_health = get_component_health(row)
    weakest_name, weakest = min(component_health.items(), key=lambda item: item[1]["score"])
    issues = _safe_int(row.get("Reported_Issues", 0))

    degradation_score = _combined_degradation_score(row, component_health, vhi, odometer)
    tier = _tier_from_score(degradation_score, weakest["score"], odometer, issues)
    degraded_core_count = sum(1 for item in component_health.values() if item["score"] <= 40)
    if odometer >= 180_000 and issues >= 4 and degraded_core_count >= 2:
        tier = "Critical"
    rul_km = _clamp(_window_value(tier, degradation_score), 0, MAX_MAINTENANCE_RUL_KM)

    if tier == "Critical" and (weakest["score"] <= 15 or degradation_score < 18 or issues >= 8):
        rul_km = min(rul_km, 500)
    elif tier == "Critical" and degraded_core_count >= 2:
        rul_km = min(rul_km, 900)

    rul_km = int(rul_km)
    risk_level = _risk_level_from_tier(tier)
    label = _label_from_tier(tier, rul_km)
    rul_display = "Immediate Maintenance Required" if label == "Immediate Maintenance Required" else f"{rul_km:,} km"
    key_factors = _build_key_factors(row, component_health, vhi, odometer, degradation_score)
    weakest_label = weakest_name.replace(" Health", "")

    if label == "Immediate Maintenance Required":
        interpretation = (
            "Immediate maintenance is required because the combined degradation score is in the "
            f"critical range. The weakest subsystem is {weakest_label} at "
            f"{weakest['score']:.0f}% ({weakest['condition']}), and the maintenance window is "
            "effectively exhausted."
        )
    elif risk_level == "CRITICAL":
        interpretation = (
            f"Estimated RUL before critical maintenance risk is {rul_display}. "
            f"The vehicle is in the {tier.lower()} maintenance tier due to combined wear, "
            f"VHI {vhi:.1f}%, odometer load, and weakest subsystem {weakest_label} "
            f"at {weakest['score']:.0f}% ({weakest['condition']})."
        )
    else:
        interpretation = (
            f"Estimated RUL before critical maintenance risk is {rul_display}. "
            f"The vehicle is in the {tier.lower()} maintenance tier. VHI is {vhi:.1f}%, "
            f"with {weakest_label} as the weakest subsystem at {weakest['score']:.0f}% "
            f"({weakest['condition']}). Continue scheduled maintenance within this window."
        )

    return {
        "vhi": vhi,
        "rul_km": rul_km,
        "rul_display": rul_display,
        "rul_label": label,
        "rul_risk_level": risk_level,
        "maintenance_tier": tier,
        "interpretation": interpretation,
        "odometer": int(odometer),
        "max_maintenance_window_km": MAX_MAINTENANCE_RUL_KM,
        "component_health": component_health,
        "weakest_subsystem": weakest_name,
        "weakest_score": weakest["score"],
        "key_factors": key_factors,
        "degradation_score": degradation_score,
    }


if __name__ == "__main__":
    sample = {
        "Brake_Condition": "Worn",
        "Tire_Condition": "Good",
        "Battery_Status": "Weak",
        "Maintenance_History": "Average",
        "Service_History": "Good",
        "Odometer_Reading": 180_000,
        "Accident_History": 1,
        "Reported_Issues": 2,
    }
    result = compute_rul(sample)
    print(f"VHI: {result['vhi']}%")
    print(f"RUL: {result['rul_display']}")
    print(f"Tier: {result['maintenance_tier']}")
    print(f"Label: {result['rul_label']}")
    print(f"Interpretation: {result['interpretation']}")
