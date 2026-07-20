"""The Gemini-led Vehicle Service Advisor orchestration boundary."""

from config.defaults import DEFAULT_VEHICLE_FEATURES, PRELIMINARY_PREDICTION_FALLBACKS
from services.advisor_service import compose_reply, format_gemini_error, gemini_available, logger, plan_turn
from services.memory_manager import new_memory, record_advisor_reply, update_memory
from services.tools import generate_service_recommendations, predict_maintenance


MODEL_FIELD_MAP = {
    "vehicle_age": "Vehicle_Age", "odometer_reading": "Odometer_Reading",
    "battery_status": "Battery_Status", "brake_condition": "Brake_Condition",
    "tire_condition": "Tire_Condition", "reported_issues": "Reported_Issues",
    "vehicle_model": "Vehicle_Model", "fuel_type": "Fuel_Type",
    "transmission_type": "Transmission_Type", "maintenance_history": "Maintenance_History",
    "service_history": "Service_History", "accident_history": "Accident_History",
    "mileage": "Mileage", "engine_size": "Engine_Size", "fuel_efficiency": "Fuel_Efficiency",
}


def _model_row(facts: dict) -> dict:
    mapped = {MODEL_FIELD_MAP[key]: value for key, value in facts.items() if key in MODEL_FIELD_MAP}
    return {**DEFAULT_VEHICLE_FEATURES, **PRELIMINARY_PREDICTION_FALLBACKS, **mapped}


def handle_message(message: str, memory: dict | None = None) -> tuple[str, dict]:
    """Run one natural advisor turn; GPT decides whether to call the ML tool."""
    memory = memory or new_memory()
    if not gemini_available():
        return "I’m ready to help, but the Gemini service needs an API key. Add `GEMINI_API_KEY` to `.env`, restart the app, and I’ll act as your service advisor.", memory
    try:
        plan = plan_turn(message, memory)
        memory = update_memory(memory, plan.extracted_facts.supplied(), message)
        prediction = None
        recommendations: list[str] = []
        if plan.should_run_prediction:
            row = _model_row(memory["vehicle"])
            prediction = predict_maintenance(row)
            recommendations = generate_service_recommendations(row, prediction)
            memory["last_result"] = prediction
        reply = compose_reply(message, memory, plan, prediction, recommendations)
        memory["last_assessment"] = plan.diagnostic_reasoning
        memory = record_advisor_reply(memory, reply.reply, plan.follow_up_question)
        return reply.reply, memory
    except Exception as exc:
        logger.exception("GEMINI ERROR in advisor request flow: %s", format_gemini_error(exc))
        return f"I’m sorry — {format_gemini_error(exc)}", memory
