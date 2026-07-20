"""Gemini-led reasoning, structured extraction, and diagnostics for the advisor."""

from __future__ import annotations

import json
import logging
import os
import atexit
from threading import Lock
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from services.extractor import AdvisorPlan, AdvisorReply


DOTENV_LOADED = load_dotenv()
ADVISOR_MODEL = os.getenv("GEMINI_ADVISOR_MODEL", "gemini-2.5-flash")
LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger = logging.getLogger("vehicle_advisor.gemini")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(LOG_DIR / "gemini_advisor.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
logger.info("Gemini config loaded: dotenv_loaded=%s sdk_version=%s model=%s", DOTENV_LOADED, genai.__version__, ADVISOR_MODEL)
_client_lock = Lock()
_gemini_client: genai.Client | None = None


def gemini_available() -> bool:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    logger.info("Gemini API key check: present=%s length=%d prefix=%s", bool(key), len(key), key[:8] if key else "none")
    return bool(key)


def get_gemini_client() -> genai.Client:
    """Return one live client for the application process; never create a temporary client."""
    global _gemini_client
    key = os.environ["GEMINI_API_KEY"].strip()
    with _client_lock:
        if _gemini_client is None:
            logger.info("Gemini client initialization: model=%s key_prefix=%s", ADVISOR_MODEL, key[:8])
            _gemini_client = genai.Client(api_key=key)
        return _gemini_client


def close_gemini_client() -> None:
    """Close only during process shutdown, never at the end of an advisor request."""
    global _gemini_client
    with _client_lock:
        if _gemini_client is not None:
            logger.info("Gemini client close event: application shutdown")
            try:
                _gemini_client.close()
            except Exception:
                logger.exception("GEMINI ERROR while closing client")
            finally:
                _gemini_client = None


atexit.register(close_gemini_client)


def format_gemini_error(exc: Exception) -> str:
    status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    detail = str(exc)
    if status:
        return f"Gemini error {status}: {detail}"
    return f"Gemini connection error ({type(exc).__name__}): {detail}"


def _generate_structured(prompt: str, schema: type, operation: str):
    logger.info("Gemini request payload: %s", json.dumps({"operation": operation, "model": ADVISOR_MODEL, "prompt": prompt}, default=str))
    try:
        client = get_gemini_client()
        logger.info("Gemini request start: operation=%s", operation)
        response = client.models.generate_content(
            model=ADVISOR_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                responseMimeType="application/json",
                responseSchema=schema,
            ),
        )
        parsed = response.parsed
        logger.info("Gemini request completion: operation=%s", operation)
        logger.info("Gemini response payload (%s): %s", operation, parsed.model_dump_json() if parsed else "null")
        if parsed is None:
            raise RuntimeError("Gemini returned no structured response.")
        return parsed
    except Exception as exc:
        logger.exception("GEMINI ERROR during %s: %s", operation, format_gemini_error(exc))
        raise


def gemini_health_check() -> dict:
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dotenv_loaded": DOTENV_LOADED,
        "sdk_version": genai.__version__,
        "model": ADVISOR_MODEL,
        "api_key_configured": gemini_available(),
    }
    if not report["api_key_configured"]:
        report.update({"ok": False, "stage": "configuration", "error": "GEMINI_API_KEY is missing."})
        return report
    try:
        logger.info("Gemini health check request: model=%s", ADVISOR_MODEL)
        client = get_gemini_client()
        logger.info("Gemini request start: operation=health_check")
        response = client.models.generate_content(model=ADVISOR_MODEL, contents="Reply with exactly OK.")
        output = response.text or ""
        logger.info("Gemini request completion: operation=health_check")
        logger.info("Gemini health check response: %s", output)
        report.update({"ok": True, "stage": "request", "response": output})
    except Exception as exc:
        logger.exception("GEMINI ERROR during health check: %s", format_gemini_error(exc))
        report.update({"ok": False, "stage": "request", "error": format_gemini_error(exc), "exception_type": type(exc).__name__})
    return report


def plan_turn(message: str, memory: dict) -> AdvisorPlan:
    prompt = f"""You are the reasoning engine for an experienced vehicle service advisor.
Understand the customer naturally, like an expert mechanic. Extract only facts stated or
strongly implied. Use retained facts and prior conversation so you never request known
information or repeat a question.

Known facts: {json.dumps(memory.get('vehicle', {}))}
Conversation: {json.dumps(memory.get('turns', [])[-16:])}
Earlier questions: {json.dumps(memory.get('asked_questions', []))}
Customer message: {message!r}

Decide should_run_prediction only when the available information makes the existing
maintenance model useful. It is acceptable to provide symptom-based advice before using
it. For acknowledgements such as 'okay', continue naturally without repeating yourself.
Choose at most one conversational follow-up question only when it genuinely helps."""
    return _generate_structured(prompt, AdvisorPlan, "plan_turn")


def compose_reply(message: str, memory: dict, plan: AdvisorPlan, prediction: dict | None, recommendations: list[str]) -> AdvisorReply:
    tool_result = None
    if prediction:
        tool_result = {
            "maintenance_risk": prediction["risk_level"],
            "model_confidence_percent": prediction["confidence"],
            "maintenance_recommended": prediction["prediction"],
        }
    prompt = f"""Act as a calm, experienced vehicle service advisor. Write a fresh response to
the latest customer message. Explain likely mechanical reasoning first and then practical
service advice. Never sound like a form, dashboard, chatbot, or questionnaire.

Customer message: {message!r}
Retained facts: {json.dumps(memory.get('vehicle', {}))}
Prior conversation: {json.dumps(memory.get('turns', [])[-12:])}
Current reasoning: {plan.diagnostic_reasoning}
Optional follow-up selected by you: {plan.follow_up_question!r}
Prediction tool result: {json.dumps(tool_result)}
Recommendation tool output: {json.dumps(recommendations)}

If the prediction result is present, its risk and confidence are authoritative. If it is
absent, give a careful preliminary symptom-based assessment without inventing a risk score.
Do not repeat the prior response. Keep it under 170 words, include a clear reason and a
helpful action, and ask no more than one natural question. Never mention fields, internal
models, feature vectors, missing inputs, or data collection."""
    return _generate_structured(prompt, AdvisorReply, "compose_reply")
