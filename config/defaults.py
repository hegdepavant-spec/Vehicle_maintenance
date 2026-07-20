"""Safe defaults for non-critical model inputs.

Change these values to tune the agent without changing its conversation logic.
"""

from datetime import date


DEFAULT_VEHICLE_FEATURES = {
    "Vehicle_Model": "Car",
    "Mileage": 12_000,
    "Maintenance_History": "Average",
    "Fuel_Type": "Petrol",
    "Transmission_Type": "Manual",
    "Engine_Size": 1500,
    "Last_Service_Date": "2024-01-01",
    "Warranty_Expiry_Date": "2025-12-31",
    "Owner_Type": "First",
    "Insurance_Premium": 20_000,
    # The saved training encoder represents service history as a 1-10 score.
    "Service_History": "6",
    "Accident_History": 0,
    "Fuel_Efficiency": 15.0,
}

# Neutral placeholders used only to make a best-effort preliminary model call.
# They are never written into conversational memory or presented as customer facts.
PRELIMINARY_PREDICTION_FALLBACKS = {
    "Vehicle_Age": 5,
    "Odometer_Reading": 60_000,
    "Battery_Status": "Good",
    "Brake_Condition": "Good",
    "Tire_Condition": "Good",
    "Reported_Issues": 0,
}

CRITICAL_FIELDS = (
    "Vehicle_Age",
    "Odometer_Reading",
    "Battery_Status",
    "Brake_Condition",
    "Tire_Condition",
    "Reported_Issues",
)

FIELD_LABELS = {
    "Vehicle_Age": "vehicle age",
    "Odometer_Reading": "current odometer reading",
    "Battery_Status": "battery condition",
    "Brake_Condition": "brake condition",
    "Tire_Condition": "tire condition",
    "Reported_Issues": "how many active issues or warning symptoms you have noticed",
}
