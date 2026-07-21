"""The Gemini-led Vehicle Service Advisor orchestration boundary.

Maps extracted facts to ML model columns, manages partial-prediction strategy,
and integrates SHAP explanations into the reply flow.
"""

from datetime import datetime

from config.defaults import DEFAULT_VEHICLE_FEATURES
from services.advisor_service import compose_reply, format_gemini_error, gemini_available, llm_available, logger, plan_turn
from services.memory_manager import get_coverage_info, new_memory, record_advisor_reply, update_memory
from services.tools import generate_service_recommendations, predict_maintenance



# ── Map VehicleFacts field names → ML model column names ─────────────────────
MODEL_FIELD_MAP = {
    "vehicle_type":        "Vehicle_Model",
    "vehicle_age":         "Vehicle_Age",
    "odometer_reading":    "Odometer_Reading",
    "number_of_services":  "Number_of_Services",
    "last_service_date":   "Last_Service_Date",
    "accident_history":    "Accident_History",
    "mileage":             "Mileage",
    "avg_km_per_day":      "Average_KM_Per_Day",
    "tyre_condition":      "Tire_Condition",
    "brake_condition":     "Brake_Condition",
}


def _model_row(facts: dict) -> tuple[dict, set[str]]:
    """Build a complete ML input row from known facts + defaults.

    Returns:
        (row_dict, user_provided_keys): The complete row and the set of ML column
        names that were explicitly provided by the user (not defaults).
    """
    # Start with safe defaults for all features
    row = dict(DEFAULT_VEHICLE_FEATURES)

    # Overlay user-provided facts
    user_provided_keys = set()
    for fact_key, value in facts.items():
        ml_col = MODEL_FIELD_MAP.get(fact_key)
        if ml_col and value is not None and value != "UNKNOWN":
            try:
                if ml_col in ["Vehicle_Age", "Odometer_Reading", "Number_of_Services", "Accident_History", "Mileage", "Average_KM_Per_Day"]:
                    row[ml_col] = float(value) if "." in str(value) else int(value)
                else:
                    row[ml_col] = value
                user_provided_keys.add(ml_col)
            except (ValueError, TypeError):
                pass

    # Derive Average_KM_Per_Day if odometer and age are known but avg_km isn't
    if "Average_KM_Per_Day" not in user_provided_keys:
        if "Vehicle_Age" in user_provided_keys and "Odometer_Reading" in user_provided_keys:
            try:
                age_days = max(int(row["Vehicle_Age"]) * 365, 1)
                row["Average_KM_Per_Day"] = round(int(row["Odometer_Reading"]) / age_days, 1)
                user_provided_keys.add("Average_KM_Per_Day")
            except (ValueError, TypeError):
                pass

    return row, user_provided_keys



def _should_predict(facts: dict) -> bool:
    """Decide whether we have enough information for a useful ML prediction.

    We attempt prediction when at least 2 meaningful features are known.
    The model uses defaults for missing features, so even partial predictions
    are useful — they'll just be marked as preliminary.
    """
    meaningful_keys = {"vehicle_age", "odometer_reading", "brake_condition",
                       "tyre_condition", "accident_history", "mileage",
                       "number_of_services", "avg_km_per_day"}
    known_count = sum(1 for k in meaningful_keys if k in facts and facts[k] is not None)
    return known_count >= 2


def handle_message(message: str, memory: dict | None = None) -> tuple[str, dict]:
    """Run one natural advisor turn; Gemini or OpenAI fallback handles extraction & response."""
    memory = memory or new_memory()

    if not llm_available():
        return (
            "I'm ready to help, but the Advisor AI service needs an API key. "
            "Add `GEMINI_API_KEY` or `OPENAI_API_KEY` to `.env`, restart the app, and I'll act as your service advisor."
        ), memory


    try:
        # ── Step 1: Gemini reasons about the message and extracts facts ──────
        plan = plan_turn(message, memory)
        memory = update_memory(memory, plan.extracted_facts.supplied(), message)

        # ── Step 2: Decide whether to run ML prediction ──────────────────────
        prediction = None
        recommendations: list[str] = []
        should_predict = plan.should_run_prediction or _should_predict(memory["vehicle"])
        previous_prediction = memory.get("last_result")

        if should_predict:
            row, user_keys = _model_row(memory["vehicle"])
            prediction = predict_maintenance(row, user_keys)
            recommendations = generate_service_recommendations(row, prediction)
            memory["last_result"] = prediction

        # ── Step 3: Build coverage info for the reply prompt ─────────────────
        coverage = get_coverage_info(memory)

        # ── Step 4: Compose natural reply grounded in ML + SHAP ──────────────
        reply = compose_reply(message, memory, plan, prediction, recommendations, coverage, previous_prediction)


        memory["last_assessment"] = plan.diagnostic_reasoning
        memory = record_advisor_reply(memory, reply.reply, plan.follow_up_question)

        return reply.reply, memory

    except Exception as exc:
        logger.exception("GEMINI ERROR in advisor request flow: %s", format_gemini_error(exc))
        return f"I'm sorry — {format_gemini_error(exc)}", memory
