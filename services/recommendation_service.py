"""Actionable, condition-specific recommendations aligned with the 10-feature model."""


def build_recommendations(row: dict, result: dict) -> list[str]:
    recommendations = []

    # Brake condition
    if str(row.get("Brake_Condition", "")).lower() in {"worn", "worn out", "critical"}:
        recommendations.append("Have the brakes inspected promptly; replace worn pads or components before regular driving.")

    # Tire condition
    if str(row.get("Tire_Condition", "")).lower() in {"worn", "worn out", "critical"}:
        recommendations.append("Replace worn tires and check alignment and tire pressure.")

    # High odometer
    if int(row.get("Odometer_Reading", 0)) >= 100_000:
        recommendations.append("Book a comprehensive service because the odometer reading warrants a full inspection.")

    # High daily usage
    try:
        avg_km = float(row.get("Average_KM_Per_Day", 0))
    except (TypeError, ValueError):
        avg_km = 0
    if avg_km >= 80:
        recommendations.append("High daily mileage increases wear — consider more frequent service intervals.")

    # Long time since last service
    try:
        days_since = float(row.get("days_since_last_service", 0))
    except (TypeError, ValueError):
        days_since = 0
    if days_since >= 365:
        recommendations.append("It has been over a year since the last service — schedule a check-up soon.")

    # Accident history
    try:
        accidents = int(row.get("Accident_History", 0))
    except (TypeError, ValueError):
        accidents = 0
    if accidents >= 2:
        recommendations.append("Multiple past accidents may have caused hidden wear — a thorough inspection is advisable.")

    if not recommendations:
        recommendations.append("Continue the regular service schedule and monitor for any new warning signs.")

    return recommendations[:4]
