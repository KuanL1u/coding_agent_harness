"""Control tools that influence the agent loop itself.

``task_done`` is the model's signal that the task is finished. It records the
summary on a shared :class:`TaskState` object that the agent loop inspects after
each step, then returns a normal tool observation so the conversation stays
well-formed.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import Tool, ToolParam, ToolResult


@dataclass
class TaskState:
    """Mutable flag set when the model declares the task complete."""

    done: bool = False
    summary: str = ""


def build_control_tools(state: TaskState) -> list[Tool]:
    """Construct control tools that mutate ``state``."""

    def task_done(summary: str) -> ToolResult:
        state.done = True
        state.summary = summary
        return ToolResult(content=f"Task marked complete: {summary}")

    return [
        Tool(
            name="task_done",
            description=(
                "Call this when the task is fully complete. Provide a concise "
                "summary of what was accomplished. This ends the agent run."
            ),
            params=[
                ToolParam("summary", "string", "Summary of the completed work."),
            ],
            func=task_done,
        ),
    ]
