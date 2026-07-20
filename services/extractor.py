"""Pydantic schemas shared by Gemini structured extraction and session memory."""

from __future__ import annotations

from pydantic import BaseModel, Field


class VehicleFacts(BaseModel):
    vehicle_age: int | None = Field(default=None, ge=0, le=60)
    odometer_reading: int | None = Field(default=None, ge=0, le=2_000_000)
    battery_status: str | None = None
    brake_condition: str | None = None
    tire_condition: str | None = None
    reported_issues: int | None = Field(default=None, ge=0, le=10)
    vehicle_model: str | None = None
    fuel_type: str | None = None
    transmission_type: str | None = None
    maintenance_history: str | None = None
    service_history: str | None = None
    accident_history: int | None = Field(default=None, ge=0, le=20)
    mileage: int | None = Field(default=None, ge=0, le=500_000)
    engine_size: int | None = Field(default=None, ge=500, le=10_000)
    fuel_efficiency: float | None = Field(default=None, ge=1, le=150)

    def supplied(self) -> dict:
        values = self.model_dump() if hasattr(self, "model_dump") else self.dict()
        return {key: value for key, value in values.items() if value is not None}


class AdvisorPlan(BaseModel):
    extracted_facts: VehicleFacts
    symptom_summary: str
    diagnostic_reasoning: str
    should_run_prediction: bool
    follow_up_question: str | None = None


class AdvisorReply(BaseModel):
    reply: str
    recommendation_summary: str
