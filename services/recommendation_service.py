"""Actionable, condition-specific recommendations."""


def build_recommendations(row: dict, result: dict) -> list[str]:
    recommendations = []
    if str(row.get("Brake_Condition", "")).lower() in {"worn", "worn out", "critical"}:
        recommendations.append("Have the brakes inspected promptly; replace worn pads or components before regular driving.")
    if str(row.get("Tire_Condition", "")).lower() in {"worn", "worn out", "critical"}:
        recommendations.append("Replace worn tires and check alignment and tire pressure.")
    if str(row.get("Battery_Status", "")).lower() in {"weak", "dead", "critical"}:
        recommendations.append("Test the battery and charging system; arrange replacement if it does not hold charge.")
    if int(row.get("Odometer_Reading", 0)) >= 100_000:
        recommendations.append("Book a comprehensive service because the odometer reading warrants a full inspection.")
    if int(row.get("Reported_Issues", 0)) > 0:
        recommendations.append("Ask the technician to investigate the reported symptoms during the service visit.")
    if not recommendations:
        recommendations.append("Continue the regular service schedule and monitor for any new warning signs.")
    return recommendations[:4]
