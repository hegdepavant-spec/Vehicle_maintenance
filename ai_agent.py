"""
ai_agent.py
AI Conversational Assistant using Groq API (primary) with graceful fallbacks.
Fully async with timeout and retry handling.
"""

import asyncio
import concurrent.futures
import os

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo").strip()

CHAT_REQUEST_TIMEOUT = float(os.getenv("CHAT_REQUEST_TIMEOUT", "20"))
CHAT_SYNC_TIMEOUT = float(os.getenv("CHAT_SYNC_TIMEOUT", "60"))


def _build_groq_client():
    """Create a Groq async client for the current event loop."""
    if not GROQ_API_KEY:
        return None
    try:
        from groq import AsyncGroq

        return AsyncGroq(api_key=GROQ_API_KEY)
    except Exception as e:
        print(f"[AI Agent] Groq init error: {e}")
        return None


def _build_openai_client():
    """Create an OpenAI async client for the current event loop."""
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import AsyncOpenAI

        return AsyncOpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        print(f"[AI Agent] OpenAI init error: {e}")
        return None


def api_available() -> bool:
    """Check if any API key is configured."""
    return bool(GROQ_API_KEY or OPENAI_API_KEY)


SYSTEM_PROMPT = """You are VehicleAI, an expert automotive diagnostics assistant embedded in a
Predictive Vehicle Maintenance Platform. Your role is to:

1. Explain maintenance predictions in clear, non-technical language
2. Interpret Vehicle Health Index (VHI) scores and what they mean
3. Explain Remaining Useful Life (RUL) estimates as bounded maintenance windows before risk becomes critical
4. Summarize SHAP feature explanations in plain English
5. Provide actionable maintenance recommendations
6. Answer general vehicle maintenance questions

Important guidelines:
- Be concise but thorough
- Use bullet points for recommendations
- Always emphasize safety when risk is HIGH or CRITICAL
- Acknowledge that RUL is an estimate based on health scoring, not sensor data
- Never describe RUL as total vehicle lifespan; it is capped at 10,000 km as a maintenance scheduling window
- If asked about something outside vehicle maintenance, politely redirect

Keep responses under 250 words unless a detailed explanation is specifically requested."""


async def _call_groq(messages: list, timeout: float = CHAT_REQUEST_TIMEOUT) -> str:
    client = _build_groq_client()
    if client is None:
        raise RuntimeError("Groq client not initialized.")

    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                max_tokens=512,
                temperature=0.6,
            ),
            timeout=timeout,
        )
        return response.choices[0].message.content.strip()
    finally:
        close = getattr(client, "close", None) or getattr(client, "aclose", None)
        if close:
            result = close()
            if asyncio.iscoroutine(result):
                await result


async def _call_openai(messages: list, timeout: float = CHAT_REQUEST_TIMEOUT) -> str:
    client = _build_openai_client()
    if client is None:
        raise RuntimeError("OpenAI client not initialized.")

    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                max_tokens=512,
                temperature=0.6,
            ),
            timeout=timeout,
        )
        return response.choices[0].message.content.strip()
    finally:
        close = getattr(client, "close", None) or getattr(client, "aclose", None)
        if close:
            result = close()
            if asyncio.iscoroutine(result):
                await result


async def chat_async(
    user_message: str,
    history: list = None,
    vehicle_context: dict = None,
) -> str:
    """
    Main async chat entry point.
    history: list of {"role": ..., "content": ...} dicts.
    vehicle_context: optional dict with current vehicle prediction info.
    """
    if not api_available():
        return (
            "Warning: **AI Assistant Unavailable**\n\n"
            "No API key is configured. To enable the AI assistant:\n"
            "1. Open the `.env` file in the project root\n"
            "2. Add your Groq API key: `GROQ_API_KEY=your_key_here`\n"
            "3. Get a free key at: https://console.groq.com/\n"
            "4. Restart the application"
        )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if vehicle_context:
        ctx_text = _build_context_text(vehicle_context)
        messages.append(
            {
                "role": "system",
                "content": f"Current vehicle analysis context:\n{ctx_text}",
            }
        )

    if history:
        messages.extend(_normalize_history(history)[-8:])

    messages.append({"role": "user", "content": user_message})

    providers = []
    if GROQ_API_KEY:
        providers.append(("Groq", _call_groq))
    if OPENAI_API_KEY:
        providers.append(("OpenAI", _call_openai))

    last_error = None
    for provider_name, provider_call in providers:
        for _ in range(2):
            try:
                return await provider_call(messages)
            except asyncio.TimeoutError:
                last_error = f"{provider_name} request timed out."
                await asyncio.sleep(1)
            except Exception as e:
                last_error = f"{provider_name}: {e}"
                await asyncio.sleep(1)

    return (
        "Warning: **AI Assistant Error**\n\n"
        "Could not get a response after 2 attempts.\n"
        f"Error: {last_error}\n\n"
        "Please check your API key and internet connection."
    )


def _build_context_text(ctx: dict) -> str:
    """Convert vehicle context dict into readable text."""
    lines = []
    if ctx.get("prediction"):
        lines.append(f"Maintenance Needed: {ctx['prediction']}")
    if ctx.get("confidence"):
        lines.append(f"Confidence: {ctx['confidence']}%")
    if ctx.get("vhi"):
        lines.append(f"Vehicle Health Index: {ctx['vhi']}%")
    if ctx.get("rul_display"):
        lines.append(f"Estimated RUL Before Critical Risk: {ctx['rul_display']}")
    elif ctx.get("rul_km"):
        lines.append(f"Estimated RUL Before Critical Risk: {ctx['rul_km']:,} km")
    if ctx.get("rul_label"):
        lines.append(f"Condition Label: {ctx['rul_label']}")
    if ctx.get("most_vulnerable"):
        lines.append(f"Most Vulnerable Component: {ctx['most_vulnerable']}")
    if ctx.get("failed_components"):
        failed = ", ".join(
            f"{item.get('name')} ({item.get('score'):.0f}%, {item.get('condition')})"
            for item in ctx["failed_components"]
        )
        lines.append(f"Failed/Critical Components: {failed}")
    if ctx.get("failing_components"):
        failing = ", ".join(
            f"{item.get('name')} ({item.get('score'):.0f}%, {item.get('condition')})"
            for item in ctx["failing_components"]
        )
        lines.append(f"Failing Components: {failing}")
    if ctx.get("degraded_components"):
        degraded = ", ".join(
            f"{item.get('name')} ({item.get('score'):.0f}%, {item.get('condition')})"
            for item in ctx["degraded_components"]
        )
        lines.append(f"Degraded Components: {degraded}")
    if ctx.get("risk_level"):
        lines.append(f"Risk Level: {ctx['risk_level']}")
    if ctx.get("shap_explanation"):
        lines.append(f"SHAP Explanation: {ctx['shap_explanation']}")
    if ctx.get("recommendation"):
        lines.append(f"Recommendation: {ctx['recommendation']}")
    return "\n".join(lines) if lines else "No vehicle analysis performed yet."


def _normalize_history(history: list) -> list:
    """Accept current Gradio message dicts and legacy tuple history."""
    normalized = []
    for item in history or []:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant"} and isinstance(content, str):
                normalized.append({"role": role, "content": content})
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            user_msg, assistant_msg = item[0], item[1]
            if user_msg:
                normalized.append({"role": "user", "content": str(user_msg)})
            if assistant_msg:
                normalized.append({"role": "assistant", "content": str(assistant_msg)})
    return normalized


def _run_coro_sync(coro):
    """
    Run an async coroutine from normal Python code, including Gradio/AnyIO
    worker threads where no event loop exists under Python 3.11.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result(timeout=CHAT_SYNC_TIMEOUT)


def run_chat(
    user_message: str,
    history: list = None,
    vehicle_context: dict = None,
) -> str:
    """Sync wrapper for Gradio worker-thread compatibility."""
    try:
        return _run_coro_sync(chat_async(user_message, history, vehicle_context))
    except concurrent.futures.TimeoutError:
        return "Warning: Assistant request timed out. Please try again."
    except Exception as e:
        return f"Warning: Assistant error: {str(e)}"


if __name__ == "__main__":
    print(f"API Available: {api_available()}")
    result = run_chat("What does a high Vehicle Health Index mean?")
    print(result)
