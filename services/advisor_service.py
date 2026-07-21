"""Gemini-led reasoning, structured extraction, and diagnostics for the advisor.

Prompts are designed to:
  - Extract features naturally from free-form user input
  - Never behave like a questionnaire
  - Ask at most ONE high-value follow-up question
  - Ground explanations in ML prediction and SHAP, never hallucinate
"""

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

import openai
from config.defaults import DEFAULT_GEMINI_MODEL, DEFAULT_OPENAI_MODEL, FEATURE_PRIORITY, FIELD_LABELS
from services.extractor import AdvisorPlan, AdvisorReply


DOTENV_LOADED = load_dotenv()
ADVISOR_MODEL = os.getenv("GEMINI_ADVISOR_MODEL", DEFAULT_GEMINI_MODEL)
OPENAI_ADVISOR_MODEL = os.getenv("OPENAI_MODEL", os.getenv("OPENAI_ADVISOR_MODEL", DEFAULT_OPENAI_MODEL))

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
logger.info(
    "LLM config loaded: dotenv_loaded=%s gemini_sdk=%s gemini_model=%s openai_model=%s",
    DOTENV_LOADED,
    genai.__version__,
    ADVISOR_MODEL,
    OPENAI_ADVISOR_MODEL,
)
_client_lock = Lock()
_gemini_client: genai.Client | None = None
_openai_client: openai.OpenAI | None = None


def gemini_available() -> bool:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    return bool(key)


def openai_available() -> bool:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    return bool(key)


GROQ_ADVISOR_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
_groq_client: openai.OpenAI | None = None


def groq_available() -> bool:
    key = os.getenv("GROQ_API_KEY", "").strip()
    return bool(key)


def llm_available() -> bool:
    return gemini_available() or openai_available() or groq_available()


def get_gemini_client() -> genai.Client:
    """Return one live client for the application process; never create a temporary client."""
    global _gemini_client
    key = os.environ["GEMINI_API_KEY"].strip()
    with _client_lock:
        if _gemini_client is None:
            logger.info("Gemini client initialization: model=%s key_prefix=%s", ADVISOR_MODEL, key[:8])
            _gemini_client = genai.Client(api_key=key)
        return _gemini_client


def get_openai_client() -> openai.OpenAI:
    """Return live OpenAI client for fallback handling."""
    global _openai_client
    key = os.environ["OPENAI_API_KEY"].strip()
    with _client_lock:
        if _openai_client is None:
            logger.info("OpenAI client initialization: model=%s key_prefix=%s", OPENAI_ADVISOR_MODEL, key[:8])
            _openai_client = openai.OpenAI(api_key=key)
        return _openai_client


def get_groq_client() -> openai.OpenAI:
    """Return live Groq client using OpenAI SDK."""
    global _groq_client
    key = os.environ["GROQ_API_KEY"].strip()
    with _client_lock:
        if _groq_client is None:
            logger.info("Groq client initialization: model=%s key_prefix=%s", GROQ_ADVISOR_MODEL, key[:8])
            _groq_client = openai.OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=key,
            )
        return _groq_client



def close_llm_clients() -> None:
    """Close active clients during process shutdown."""
    global _gemini_client, _openai_client, _groq_client
    with _client_lock:
        if _gemini_client is not None:
            logger.info("Gemini client close event: application shutdown")
            try:
                _gemini_client.close()
            except Exception:
                logger.exception("GEMINI ERROR while closing client")
            finally:
                _gemini_client = None
        if _openai_client is not None:
            logger.info("OpenAI client close event: application shutdown")
            try:
                _openai_client.close()
            except Exception:
                logger.exception("OPENAI ERROR while closing client")
            finally:
                _openai_client = None
        if _groq_client is not None:
            logger.info("Groq client close event: application shutdown")
            try:
                _groq_client.close()
            except Exception:
                logger.exception("GROQ ERROR while closing client")
            finally:
                _groq_client = None


def close_gemini_client() -> None:
    close_llm_clients()


def format_gemini_error(exc: Exception) -> str:
    status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    detail = str(exc)
    if status:
        return f"Gemini error {status}: {detail}"
    return f"Gemini connection error ({type(exc).__name__}): {detail}"


def _generate_structured_gemini(prompt: str, schema: type, operation: str):
    logger.info("Gemini request payload: %s", json.dumps({"operation": operation, "model": ADVISOR_MODEL, "prompt": prompt}, default=str))
    client = get_gemini_client()
    logger.info("Gemini request start: operation=%s", operation)
    response = client.models.generate_content(
        model=ADVISOR_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction="You are an experienced vehicle service advisor AI assistant.",
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )
    parsed = response.parsed
    logger.info("Gemini request completion: operation=%s", operation)
    logger.info("Gemini response payload (%s): %s", operation, parsed.model_dump_json() if parsed else "null")
    if parsed is None:
        raise RuntimeError("Gemini returned no structured response.")
    return parsed


def _generate_structured_openai(prompt: str, schema: type, operation: str):
    logger.info("OpenAI request payload: %s", json.dumps({"operation": operation, "model": OPENAI_ADVISOR_MODEL, "prompt": prompt}, default=str))
    client = get_openai_client()
    logger.info("OpenAI request start: operation=%s", operation)
    completion = client.beta.chat.completions.parse(
        model=OPENAI_ADVISOR_MODEL,
        messages=[
            {"role": "system", "content": "You are an experienced vehicle service advisor AI assistant."},
            {"role": "user", "content": prompt},
        ],
        response_format=schema,
    )
    parsed = completion.choices[0].message.parsed
    logger.info("OpenAI request completion: operation=%s", operation)
    logger.info("OpenAI response payload (%s): %s", operation, parsed.model_dump_json() if parsed else "null")
    if parsed is None:
        raise RuntimeError("OpenAI returned no structured response.")
    return parsed


def _generate_structured_groq(prompt: str, schema: type, operation: str):

    logger.info("Groq request payload: %s", json.dumps({"operation": operation, "model": GROQ_ADVISOR_MODEL, "prompt": prompt}, default=str))
    client = get_groq_client()
    logger.info("Groq request start: operation=%s", operation)
    
    schema_json = json.dumps(schema.model_json_schema())
    formatted_prompt = f"Return JSON strictly conforming to this schema: {schema_json}.\n\nContext and Instructions:\n{prompt}"
    
    completion = client.chat.completions.create(
        model=GROQ_ADVISOR_MODEL,
        messages=[
            {"role": "system", "content": "You are an experienced vehicle service advisor AI assistant. Output ONLY valid JSON."},
            {"role": "user", "content": formatted_prompt},
        ],
        response_format={"type": "json_object"},
    )
    raw_json = completion.choices[0].message.content or "{}"
    logger.info("Groq request completion: operation=%s", operation)
    parsed = schema.model_validate_json(raw_json)
    logger.info("Groq response payload (%s): %s", operation, parsed.model_dump_json() if parsed else "null")
    return parsed


def _generate_structured(prompt: str, schema: type, operation: str):
    errors = []

    # Strategy 1: Primary - Gemini
    if gemini_available():
        try:
            return _generate_structured_gemini(prompt, schema, operation)
        except Exception as exc:
            err_msg = str(exc)
            logger.warning("Gemini failed for %s: %s", operation, format_gemini_error(exc))
            errors.append(f"Gemini: {err_msg}")

    # Strategy 2: Secondary Fallback - OpenAI
    if openai_available():
        try:
            logger.info("FALLBACK: Attempting OpenAI (%s) for %s...", OPENAI_ADVISOR_MODEL, operation)
            return _generate_structured_openai(prompt, schema, operation)
        except Exception as exc:
            logger.warning("OpenAI fallback failed for %s: %s", operation, exc)
            errors.append(f"OpenAI: {str(exc)}")

    # Strategy 3: Tertiary Fallback - Groq
    if groq_available():
        try:
            logger.info("FALLBACK: Attempting Groq (%s) for %s...", GROQ_ADVISOR_MODEL, operation)
            return _generate_structured_groq(prompt, schema, operation)
        except Exception as exc:
            logger.warning("Groq fallback failed for %s: %s", operation, exc)
            errors.append(f"Groq: {str(exc)}")

    raise RuntimeError(f"All available LLM providers failed for {operation}. Errors: {'; '.join(errors) if errors else 'No provider configured.'}")


def llm_health_check() -> dict:
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dotenv_loaded": DOTENV_LOADED,
        "gemini": {
            "sdk_version": genai.__version__,
            "model": ADVISOR_MODEL,
            "api_key_configured": gemini_available(),
        },
        "openai": {
            "sdk_version": openai.__version__,
            "model": OPENAI_ADVISOR_MODEL,
            "api_key_configured": openai_available(),
        },
        "groq": {
            "model": GROQ_ADVISOR_MODEL,
            "api_key_configured": groq_available(),
        },
    }

    if gemini_available():
        try:
            client = get_gemini_client()
            response = client.models.generate_content(model=ADVISOR_MODEL, contents="Reply with exactly OK.")
            report["gemini"]["ok"] = True
            report["gemini"]["response"] = response.text or ""
        except Exception as exc:
            report["gemini"].update({"ok": False, "error": format_gemini_error(exc)})

    if openai_available():
        try:
            client = get_openai_client()
            response = client.chat.completions.create(model=OPENAI_ADVISOR_MODEL, messages=[{"role": "user", "content": "Reply with exactly OK."}])
            report["openai"]["ok"] = True
            report["openai"]["response"] = response.choices[0].message.content or ""
        except Exception as exc:
            report["openai"].update({"ok": False, "error": str(exc)})

    if groq_available():
        try:
            client = get_groq_client()
            response = client.chat.completions.create(model=GROQ_ADVISOR_MODEL, messages=[{"role": "user", "content": "Reply with exactly OK."}])
            report["groq"]["ok"] = True
            report["groq"]["response"] = response.choices[0].message.content or ""
        except Exception as exc:
            report["groq"].update({"ok": False, "error": str(exc)})

    report["ok"] = report.get("gemini", {}).get("ok", False) or report.get("openai", {}).get("ok", False) or report.get("groq", {}).get("ok", False)
    return report


def gemini_health_check() -> dict:
    """Backwards compatible alias for llm_health_check."""
    return llm_health_check()




def plan_turn(message: str, memory: dict) -> AdvisorPlan:
    """Reason about the customer message, extract facts, decide next action."""

    # Build the feature priority context for intelligent follow-up selection
    known_features = list((memory.get("vehicle") or {}).keys())
    missing_priority = [f for f in FEATURE_PRIORITY if f not in known_features]
    missing_labels = {f: FIELD_LABELS.get(f, f) for f in missing_priority[:3]}

    prompt = f"""You are the conversational reasoning engine for a Vehicle Maintenance AI Agent.
You are NOT the maintenance prediction model. A trained Machine Learning model is the ONLY authority for predicting maintenance risk.
Your job is to behave exactly like an experienced vehicle service advisor at a professional service center.

CURRENT KNOWN FACTS: {json.dumps(memory.get('vehicle', {}))}
CONVERSATION HISTORY: {json.dumps(memory.get('turns', [])[-16:])}
QUESTIONS ALREADY ASKED: {json.dumps(memory.get('asked_questions', []))}
CUSTOMER MESSAGE: {message!r}

FEATURE EXTRACTION & GROUNDING RULES:
- Extract structured facts for: vehicle_type, vehicle_age, odometer_reading, number_of_services, last_service_date, accident_history, mileage, avg_km_per_day, tyre_condition, brake_condition.
- Extract only what is stated or strongly implied by the customer. Never invent or assume data.
- If the customer says "I don't know", "I'm not sure", "No idea", or expresses uncertainty about a specific feature:
  - Do NOT guess or invent a value.
  - Set that feature explicitly to the string "UNKNOWN" in extracted_facts.
  - Do NOT try to ask about that feature again in your follow-up questions.

PREDICTION DECISION:
Set should_run_prediction to true when at least a few features are available. But if the user is just greeting, set it to false.

DYNAMIC FOLLOW-UP STRATEGY:
Your objective is NOT to collect every input in a fixed order. Your goal is to maximize prediction quality using the minimum number of questions.
- Decide the next question dynamically based on the current symptoms and conversation:
  * If the user mentions mileage drop/pickup issues, prioritize `last_service_date` or `number_of_services`.
  * If the user mentions spongy or soft brakes, prioritize `brake_condition`.
  * If the user mentions high odometer/mileage, prioritize regular maintenance service details.
  * Otherwise, select the feature expected to improve diagnostic quality most.
- Choose ONLY ONE follow-up question.
- Set priority_missing_feature to the key name of that feature.
- NEVER ask about features already known or marked as "UNKNOWN" in CURRENT KNOWN FACTS.
- NEVER ask multiple questions at once. Never behave like a questionnaire or a form.
- Phrase the follow-up naturally, as an experienced mechanic would in conversation.

CONFIDENCE ASSESSMENT:
Briefly note whether the available information seems sufficient for a reliable ML prediction."""


    plan = _generate_structured(prompt, AdvisorPlan, "plan_turn")

    # Code-level guarantee: if features are missing, ensure a follow-up question is set
    if missing_priority and (not plan.follow_up_question or not plan.follow_up_question.strip()):
        top_missing = missing_priority[0]
        label = FIELD_LABELS.get(top_missing, top_missing.replace("_", " "))
        plan.priority_missing_feature = top_missing
        plan.follow_up_question = f"To help refine the diagnostic accuracy, could you share the {label}?"

    return plan


def compose_reply(
    message: str,
    memory: dict,
    plan: AdvisorPlan,
    prediction: dict | None,
    recommendations: list[str],
    coverage_info: dict | None = None,
    previous_prediction: dict | None = None,
) -> AdvisorReply:
    """Compose the natural-language reply using ML results and SHAP explanation."""

    tool_result = None
    shap_text = ""
    if prediction:
        tool_result = {
            "maintenance_risk": prediction["risk_level"],
            "model_confidence_percent": prediction["confidence"],
            "maintenance_recommended": prediction["prediction"],
            "is_preliminary": prediction.get("preliminary", True),
            "features_provided": prediction.get("features_provided", 0),
            "features_total": prediction.get("features_total", 10),
        }
        shap_text = prediction.get("shap_explanation", "")

    prompt = f"""Act as a friendly, confident, and professional vehicle service advisor. Write a fresh response to the latest customer message.

ROLE BOUNDARY: You work alongside a trained Machine Learning maintenance model. You do NOT make predictions yourself; the ML model result is AUTHORITATIVE.

CUSTOMER MESSAGE: {message!r}
RETAINED FACTS: {json.dumps(memory.get('vehicle', {}))}
PRIOR CONVERSATION: {json.dumps(memory.get('turns', [])[-12:])}
CURRENT REASONING: {plan.diagnostic_reasoning}
CONFIDENCE ASSESSMENT: {plan.confidence_assessment}

PREVIOUS ML RESULT: {json.dumps(previous_prediction)}
CURRENT ML PREDICTION RESULT: {json.dumps(tool_result)}
SHAP EXPLANATION: {shap_text!r}
RECOMMENDATION TOOL OUTPUT: {json.dumps(recommendations)}
FEATURE COVERAGE: {json.dumps(coverage_info)}
REQUIRED FOLLOW-UP QUESTION TO INCLUDE: {plan.follow_up_question!r}

═══════════════════════════════════════════════════════════════
GROUNDING & PHRASING RULES — FOLLOW EXACTLY:
═══════════════════════════════════════════════════════════════

1. SILENT ML PREDICTIONS (CRITICAL RULE):
   - The ML prediction should happen silently in the background.
   - Do NOT display or repeat the Risk Level, Confidence score, or SHAP numbers after every user message.
   - Mention the prediction ONLY if:
     a) The risk level has changed significantly compared to the PREVIOUS ML RESULT (e.g., Low to High, None to Medium).
     b) All 10 features have been completed.
     c) The customer explicitly asks for the risk assessment or prediction.
   - Otherwise, do NOT report the prediction, risk, or confidence. Speak naturally about the symptoms and recommendations.

2. GROUNDING & NO HALLUCINATIONS:
   - Use phrases like "Based on the information you've provided..." instead of "Since your vehicle has..." unless explicitly stated.
   - Never assume service history, accident history, mileage, tyre condition, or brake condition if not provided.

3. WHEN USER DOESN'T KNOW:
   - If the user says "I don't know", "I'm not sure", or "No idea", accept the answer gracefully. Say something natural like "No worries, that's completely fine" and transition to the next topic. Do NOT repeat or press on that question.

4. MANDATORY FOLLOW-UP QUESTION:
   - If a follow-up question is provided in REQUIRED FOLLOW-UP QUESTION ({plan.follow_up_question!r}), you MUST include it at the end of your response so the customer can provide the details needed.
   - Ask ONLY this single question. Do NOT present a list of missing fields like a questionnaire.

5. TONE & LENGTH:
   - Speak naturally like an experienced service advisor. Be friendly, confident, and conversational. Never sound robotic or like a report.
   - Keep under 170 words. Do not repeat prior responses."""

    res = _generate_structured(prompt, AdvisorReply, "compose_reply")
    
    # Guarantee follow-up question inclusion if generated by planning phase
    if plan.follow_up_question and plan.follow_up_question.strip():
        q_text = plan.follow_up_question.strip()
        # If the question text isn't in the reply, append it cleanly at the end
        if q_text.rstrip("?").lower() not in res.reply.lower():
            res.reply = f"{res.reply.strip()}\n\n{q_text}"
            
    return res



