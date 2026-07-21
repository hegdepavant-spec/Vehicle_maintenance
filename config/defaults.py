"""Safe defaults and feature metadata for the 10-feature ML model.

Change these values to tune the agent without changing its conversation logic.

The 10 model features:
  1. Vehicle_Model        (categorical)
  2. Vehicle_Age          (numeric)
  3. Odometer_Reading      (numeric)
  4. Number_of_Services    (numeric, derived from Service_History)
  5. days_since_last_service (numeric, derived from Last_Service_Date)
  6. Accident_History      (numeric)
  7. Mileage              (numeric, fuel-efficiency proxy)
  8. Average_KM_Per_Day    (numeric, derived)
  9. Tire_Condition        (categorical)
 10. Brake_Condition       (categorical)
"""

# ── Neutral defaults used when the user hasn't supplied a feature ────────────
# These let the ML model run a best-effort prediction on partial information.
# They are never written into conversational memory or stated as customer facts.

DEFAULT_VEHICLE_FEATURES = {
    "Vehicle_Model":          "Car",
    "Vehicle_Age":            5,
    "Odometer_Reading":       60_000,
    "Number_of_Services":     6,
    "Last_Service_Date":      "2024-01-01",
    "Accident_History":       0,
    "Mileage":                50_000,
    "Average_KM_Per_Day":     33,
    "Tire_Condition":         "Good",
    "Brake_Condition":        "Good",
}

# ── All model feature keys (agent-side names used in VehicleFacts) ───────────
ALL_MODEL_FEATURES = (
    "vehicle_type",
    "vehicle_age",
    "odometer_reading",
    "number_of_services",
    "last_service_date",
    "accident_history",
    "mileage",
    "avg_km_per_day",
    "tyre_condition",
    "brake_condition",
)

# ── Feature priority for follow-up question strategy ─────────────────────────
# Ordered from most to least impactful for prediction quality.
# When multiple features are missing, the LLM should ask about the FIRST one
# in this list that is still unknown.
FEATURE_PRIORITY = (
    "brake_condition",
    "tyre_condition",
    "odometer_reading",
    "vehicle_age",
    "accident_history",
    "mileage",
    "number_of_services",
    "last_service_date",
    "avg_km_per_day",
    "vehicle_type",
)

# ── Human-friendly labels for each feature ───────────────────────────────────
# Used by the LLM to phrase natural follow-up questions.
FIELD_LABELS = {
    "vehicle_type":        "type of vehicle (car, truck, SUV, etc.)",
    "vehicle_age":         "age of the vehicle in years",
    "odometer_reading":    "current odometer reading in kilometres",
    "number_of_services":  "how many services the vehicle has had",
    "last_service_date":   "when the last service was done",
    "accident_history":    "number of past accidents",
    "mileage":             "annual mileage or fuel efficiency",
    "avg_km_per_day":      "average kilometres driven per day",
    "tyre_condition":      "current condition of the tyres",
    "brake_condition":     "current condition of the brakes",
}

# ── LLM Models ───────────────────────────────────────────────────────────────
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

