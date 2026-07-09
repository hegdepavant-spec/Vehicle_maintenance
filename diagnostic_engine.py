"""
diagnostic_engine.py
Rule-based diagnostic engine for vehicle subsystem risk analysis.
"""

# ── Condition Score Maps ────────────────────────────────────────────────────
BRAKE_SCORES = {
    "new": 100, "good": 75, "moderate": 55, "worn": 35,
    "worn out": 35, "critical": 10
}
TIRE_SCORES = {
    "new": 100, "good": 75, "moderate": 55, "worn": 35,
    "worn out": 30, "critical": 10
}
BATTERY_SCORES = {
    "new": 100, "good": 75, "moderate": 55, "weak": 40,
    "dead": 5, "critical": 5
}
MAINTENANCE_SCORES = {
    "excellent": 100, "good": 75, "average": 65,
    "poor": 30, "unknown": 50
}
SERVICE_SCORES = {
    "excellent": 100, "good": 75, "average": 65,
    "poor": 30, "unknown": 50
}
ACCIDENT_PENALTY = {0: 100, 1: 75, 2: 55, 3: 35}

MAX_EXPECTED_KM = 300_000


def _score(mapping: dict, key: str, default: int = 50) -> int:
    return mapping.get(str(key).lower().strip(), default)


def run_diagnostics(row: dict) -> dict:
    """
    Analyze a vehicle row and return subsystem risk report.
    row: dict with vehicle attributes (raw / un-encoded values).
    """
    brake_score = _score(BRAKE_SCORES, row.get("Brake_Condition", ""), 50)
    tire_score = _score(TIRE_SCORES, row.get("Tire_Condition", ""), 50)
    battery_score = _score(BATTERY_SCORES, row.get("Battery_Status", ""), 50)
    maint_score = _score(MAINTENANCE_SCORES, row.get("Maintenance_History", ""), 50)

    # Accident penalty
    try:
        accidents = int(row.get("Accident_History", 0))
    except (TypeError, ValueError):
        accidents = 0
    accident_score = ACCIDENT_PENALTY.get(min(accidents, 3), 35)

    # Odometer health
    try:
        odometer = float(row.get("Odometer_Reading", 0))
    except (TypeError, ValueError):
        odometer = 0
    mileage_health = max(0, 100 * (1 - odometer / MAX_EXPECTED_KM))

    # ── Risk Levels ─────────────────────────────────────────────────────────
    def risk_level(score: float) -> str:
        if score >= 80:
            return "LOW"
        elif score >= 55:
            return "MODERATE"
        elif score >= 35:
            return "HIGH"
        else:
            return "CRITICAL"

    def recommendation(component: str, score: float, cond_str: str = "") -> str:
        lvl = risk_level(score)
        cond_str = cond_str.strip()
        if lvl == "LOW":
            return f"{component}: No immediate action required. Continue regular service intervals."
        elif lvl == "MODERATE":
            return f"{component}: Schedule inspection within next 5,000 km. Current condition: {cond_str}."
        elif lvl == "HIGH":
            return f"{component}: Urgent inspection recommended. Condition reported as '{cond_str}' — schedule within 1,000 km."
        else:
            return f"{component}: IMMEDIATE SERVICE REQUIRED. Critical condition detected. Do not delay service."

    systems = {
        "Brake System": {
            "score": brake_score,
            "condition": row.get("Brake_Condition", "N/A"),
            "recommendation": recommendation("Brake System", brake_score, row.get("Brake_Condition", ""))
        },
        "Tire System": {
            "score": tire_score,
            "condition": row.get("Tire_Condition", "N/A"),
            "recommendation": recommendation("Tire System", tire_score, row.get("Tire_Condition", ""))
        },
        "Battery System": {
            "score": battery_score,
            "condition": row.get("Battery_Status", "N/A"),
            "recommendation": recommendation("Battery System", battery_score, row.get("Battery_Status", ""))
        },
        "Maintenance Record": {
            "score": maint_score,
            "condition": row.get("Maintenance_History", "N/A"),
            "recommendation": recommendation("Maintenance Record", maint_score, row.get("Maintenance_History", ""))
        },
        "Mileage / Odometer": {
            "score": round(mileage_health, 1),
            "condition": f"{int(odometer):,} km",
            "recommendation": recommendation("Mileage / Odometer", mileage_health, f"{int(odometer):,} km")
        },
    }

    # Most vulnerable component
    most_vulnerable = min(systems.items(), key=lambda x: x[1]["score"])
    comp_name = most_vulnerable[0]
    comp_data = most_vulnerable[1]

    overall_risk = risk_level(comp_data["score"])
    reported_issues = int(row.get("Reported_Issues", 0))

    return {
        "systems": systems,
        "most_vulnerable_component": comp_name,
        "most_vulnerable_score": comp_data["score"],
        "most_vulnerable_condition": comp_data["condition"],
        "most_vulnerable_recommendation": comp_data["recommendation"],
        "overall_risk_level": overall_risk,
        "reported_issues": reported_issues,
        "accident_history": accidents,
    }


if __name__ == "__main__":
    sample = {
        "Brake_Condition": "Worn",
        "Tire_Condition": "Good",
        "Battery_Status": "Weak",
        "Maintenance_History": "Poor",
        "Odometer_Reading": 180000,
        "Accident_History": 2,
        "Reported_Issues": 3,
    }
    result = run_diagnostics(sample)
    print(f"Most Vulnerable: {result['most_vulnerable_component']}")
    print(f"Risk Level: {result['overall_risk_level']}")
    print(f"Recommendation: {result['most_vulnerable_recommendation']}")
