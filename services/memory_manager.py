"""Persistent per-session context for the Vehicle Service Advisor."""

from __future__ import annotations

from copy import deepcopy


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
