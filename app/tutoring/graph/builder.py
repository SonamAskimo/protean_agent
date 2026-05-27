"""Assemble the LangGraph StateGraph for the tutoring state machine."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from .nodes import (
    advance_segment,
    build_prompt,
    classify_intent,
    go_back_segment,
    mark_done,
)
from .state import TutorState


def _route_by_intent(state: TutorState) -> str:
    intent = state.get("intent", "question")
    if intent == "continue":
        return "advance"
    if intent == "go_back":
        return "go_back"
    if intent == "done":
        return "mark_done"
    # greeting / question / repeat / simpler → stay on current segment, just build prompt
    return "build_prompt"


def build_tutor_graph() -> StateGraph:
    g = StateGraph(TutorState)

    g.add_node("classify", classify_intent)
    g.add_node("advance", advance_segment)
    g.add_node("go_back", go_back_segment)
    g.add_node("mark_done", mark_done)
    g.add_node("build_prompt", build_prompt)

    g.set_entry_point("classify")

    g.add_conditional_edges("classify", _route_by_intent, {
        "advance": "advance",
        "go_back": "go_back",
        "mark_done": "mark_done",
        "build_prompt": "build_prompt",
    })

    g.add_edge("advance", "build_prompt")
    g.add_edge("go_back", "build_prompt")
    g.add_edge("mark_done", "build_prompt")
    g.add_edge("build_prompt", END)

    return g.compile()
