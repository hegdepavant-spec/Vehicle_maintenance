"""Pydantic schemas shared by Gemini structured extraction and session memory.

Aligned with the 10-feature ML model:
  vehicle_type, vehicle_age, odometer_reading, number_of_services,
  last_service_date, accident_history, mileage, avg_km_per_day,
  tyre_condition, brake_condition
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from config.defaults import ALL_MODEL_FEATURES


class VehicleFacts(BaseModel):
    vehicle_type: str | None = Field(default=None, description="Vehicle type: Car, Truck, SUV, Van, Bus, Motorcycle, or 'UNKNOWN'")
    vehicle_age: int | str | None = Field(default=None, description="Age in years, or 'UNKNOWN'")
    odometer_reading: int | str | None = Field(default=None, description="Odometer in km, or 'UNKNOWN'")
    number_of_services: int | str | None = Field(default=None, description="Total service count, or 'UNKNOWN'")
    last_service_date: str | None = Field(default=None, description="Last service date YYYY-MM-DD or natural language, or 'UNKNOWN'")
    accident_history: int | str | None = Field(default=None, description="Number of past accidents, or 'UNKNOWN'")
    mileage: int | str | None = Field(default=None, description="Annual mileage in km, or 'UNKNOWN'")
    avg_km_per_day: float | str | None = Field(default=None, description="Average km driven daily, or 'UNKNOWN'")
    tyre_condition: str | None = Field(default=None, description="Tyre condition: New, Good, Moderate, Worn, Worn Out, Critical, or 'UNKNOWN'")
    brake_condition: str | None = Field(default=None, description="Brake condition: New, Good, Moderate, Worn, Worn Out, Critical, or 'UNKNOWN'")


    def supplied(self) -> dict:
        """Return only fields that were explicitly set (non-None)."""
        values = self.model_dump() if hasattr(self, "model_dump") else self.dict()
        return {key: value for key, value in values.items() if value is not None}

    def missing_features(self) -> list[str]:
        """Return feature keys that are still None."""
        values = self.model_dump() if hasattr(self, "model_dump") else self.dict()
        return [key for key in ALL_MODEL_FEATURES if values.get(key) is None]

    def known_count(self) -> int:
        return len(ALL_MODEL_FEATURES) - len(self.missing_features())

    def total_count(self) -> int:
        return len(ALL_MODEL_FEATURES)


class AdvisorPlan(BaseModel):
    extracted_facts: VehicleFacts
    symptom_summary: str
    diagnostic_reasoning: str
    should_run_prediction: bool
    confidence_assessment: str = Field(
        default="",
        description="Brief reasoning about whether enough info is available for a useful ML prediction",
    )
    priority_missing_feature: str | None = Field(
        default=None,
        description="The single most valuable missing feature to ask about next, from the priority list",
    )
    follow_up_question: str | None = None


class AdvisorReply(BaseModel):
    reply: str
    recommendation_summary: str
