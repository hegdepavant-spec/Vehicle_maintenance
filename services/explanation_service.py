"""Plain-language assessment output. No raw model features are exposed."""

from services.recommendation_service import build_recommendations


def build_assessment_response(row: dict, result: dict, known_fields: set[str], follow_up: str | None = None) -> str:
    risk = result["risk_level"].title()
    maintenance = "I recommend arranging a service visit" if result["prediction"] else "There is no immediate service flag from the information available"
    factors = _customer_factors(row, result)[:2]
    lines = [
        "Based on what you've described, here is my initial assessment.",
        "",
        f"**Maintenance risk: {risk}**",
        f"{maintenance}. I have **{result['advisory_confidence']:.0f}% confidence** in this initial view.",
        "",
        "**What stands out**",
    ]
    lines.extend(f"- {factor}" for factor in factors)
    lines.extend(["", "**Recommended actions**"])
    lines.extend(f"- {item}" for item in build_recommendations(row, result))
    if follow_up:
        lines.extend(["", follow_up])
    return "\n".join(lines)


def _customer_factors(row: dict, result: dict) -> list[str]:
    factors = []
    if int(row.get("Odometer_Reading", 0)) >= 100_000:
        factors.append("The vehicle's mileage suggests it is due for closer routine inspection.")
    for field, label in (("Battery_Status", "battery"), ("Brake_Condition", "brakes"), ("Tire_Condition", "tires")):
        value = str(row.get(field, "")).lower()
        if value in {"weak", "dead", "critical", "worn", "worn out", "moderate"}:
            factors.append(f"The reported {label} condition needs attention.")
    if int(row.get("Reported_Issues", 0)):
        factors.append("The symptoms you mentioned also increase the chance that service is needed soon.")
    return factors or ["The current information does not point to a specific urgent component concern."]
