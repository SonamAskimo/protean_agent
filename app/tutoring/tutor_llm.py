"""LangGraph runner for dynamic tutor instructions."""

from __future__ import annotations

import logging

from .graph.builder import build_tutor_graph
from .graph.state import TutorState
from .prompts import build_system_prompt

logger = logging.getLogger("tutor-llm")


class GraphRunner:
    """Thin wrapper around the compiled LangGraph."""

    def __init__(self, initial_state: TutorState) -> None:
        self._graph = build_tutor_graph()
        self._state: TutorState = {**initial_state}
        self._turn = 0

    @property
    def state(self) -> TutorState:
        return self._state

    def process_turn(self, user_input: str) -> str:
        """Run one graph cycle and return the new system prompt."""
        if self._turn == 0 and not user_input:
            self._state["phase"] = "greeting"
            self._state["user_input"] = ""
            self._state = self._graph.invoke(self._state)
            self._turn += 1
            return self._state.get("system_prompt") or build_system_prompt(self._state)

        self._state["user_input"] = user_input
        if self._state.get("phase") == "greeting":
            self._state["phase"] = "teaching"
        self._state = self._graph.invoke(self._state)
        self._turn += 1
        return self._state.get("system_prompt") or build_system_prompt(self._state)
