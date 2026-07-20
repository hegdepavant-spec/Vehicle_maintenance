"""Compatibility helpers for the retired dashboard chat surface.

The active application uses ``agents.vehicle_agent`` and Gemini directly.
"""

from services.advisor_service import gemini_available


def api_available() -> bool:
    return gemini_available()


def run_chat(user_message: str, history: list = None, vehicle_context: dict = None) -> str:
    if not gemini_available():
        return "Gemini is not configured. Add `GEMINI_API_KEY` to `.env` and restart the application."
    return "Please use the main Service Advisor conversation. It keeps the vehicle context for the full session."
