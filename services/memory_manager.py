"""Persistent per-session context for the Vehicle Service Advisor.

Tracks extracted vehicle features, conversation turns, and asked questions.
Provides coverage helpers so the agent knows which features are known/missing.
"""

from __future__ import annotations

from copy import deepcopy

from config.defaults import ALL_MODEL_FEATURES, FIELD_LABELS


def new_memory() -> dict:
    return {"vehicle": {}, "turns": [], "asked_questions": [], "last_assessment": None, "last_result": None}


def update_memory(memory: dict | None, facts: dict, user_message: str) -> dict:
    memory = deepcopy(memory or new_memory())
    memory.setdefault("vehicle", {}).update(facts)
    memory.setdefault("turns", []).append({"role": "user", "content": user_message})
    memory["turns"] = memory["turns"][-20:]
    return memory


def record_advisor_reply(memory: dict, reply: str, question: str | None) -> dict:
    memory = deepcopy(memory)
    memory.setdefault("turns", []).append({"role": "assistant", "content": reply})
    memory["turns"] = memory["turns"][-20:]
    if question:
        memory.setdefault("asked_questions", []).append(question)
        memory["asked_questions"] = memory["asked_questions"][-8:]
    return memory


# ── Coverage helpers ─────────────────────────────────────────────────────────

def get_known_features(memory: dict) -> set[str]:
    """Return the set of feature keys that have been explicitly provided."""
    vehicle = (memory or {}).get("vehicle", {})
    return {key for key in ALL_MODEL_FEATURES if key in vehicle and vehicle[key] is not None}


def get_missing_features(memory: dict) -> set[str]:
    """Return the set of feature keys still missing."""
    return set(ALL_MODEL_FEATURES) - get_known_features(memory)


def get_coverage_info(memory: dict) -> dict:
    """Return a summary of feature coverage for prompt injection."""
    known = get_known_features(memory)
    missing = set(ALL_MODEL_FEATURES) - known
    return {
        "known": sorted(known),
        "missing": sorted(missing),
        "known_count": len(known),
        "total": len(ALL_MODEL_FEATURES),
        "missing_labels": {f: FIELD_LABELS.get(f, f) for f in sorted(missing)},
    }
