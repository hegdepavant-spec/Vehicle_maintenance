"""Named tool boundary used by the Vehicle Service Advisor."""

from services.prediction_service import run_prediction
from services.recommendation_service import build_recommendations


def predict_maintenance(vehicle_features: dict) -> dict:
    return run_prediction(vehicle_features)


def generate_service_recommendations(vehicle_features: dict, prediction: dict) -> list[str]:
    return build_recommendations(vehicle_features, prediction)
