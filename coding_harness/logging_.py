"""Structured event logging for an agent run.

Every notable event (LLM request, assistant message, tool call, tool result,
loop end) is appended as one JSON object per line to a JSONL trace file, and
mirrored to the console in a compact human-readable form. Cumulative
tokens / steps / wall-clock are tracked so budgets can be enforced and reported.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, TextIO


def _preview(text: str | None, limit: int = 500) -> str:
    """Shorten text for console mirroring."""
    if not text:
        return ""
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "..."


class EventLogger:
    """Append-only JSONL trace with a console mirror and run counters."""

    def __init__(self, trace_file: str | Path, console: bool = True) -> None:
        self.trace_path = Path(trace_file)
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.console = console
        self._fh: TextIO = self.trace_path.open("a", encoding="utf-8")

        self.start_time = time.time()
        self.steps = 0
        self.total_tokens = 0

    # -- core ---------------------------------------------------------------

    def log(self, event_type: str, **fields: Any) -> None:
        """Write one event to the trace and (optionally) mirror to console."""
        record = {
            "ts": round(time.time() - self.start_time, 3),
            "event": event_type,
            **fields,
        }
        self._fh.write(json.dumps(record, default=str) + "\n")
        self._fh.flush()
        if self.console:
            self._mirror(event_type, fields)

    def _mirror(self, event_type: str, fields: dict[str, Any]) -> None:
        if event_type == "llm_request":
            print(f"[step {fields.get('step')}] -> LLM ({fields.get('num_messages')} msgs)")
        elif event_type == "assistant_message":
            thought = _preview(fields.get("content"))
            if thought:
                print(f"  thought: {thought}")
            calls = fields.get("tool_calls") or []
            for c in calls:
                print(f"  action: {c.get('name')}({_preview(c.get('arguments'), 160)})")
        elif event_type == "tool_result":
            tag = "ERROR" if fields.get("is_error") else "ok"
            print(f"  observation [{tag}]: {_preview(fields.get('content'), 300)}")
        elif event_type == "loop_end":
            print(
                f"[done] status={fields.get('status')} steps={fields.get('steps')} "
                f"tokens={fields.get('total_tokens')} "
                f"elapsed={fields.get('elapsed_s')}s"
            )

    # -- counters -----------------------------------------------------------

    def record_step(self) -> int:
        self.steps += 1
        return self.steps

    def set_total_tokens(self, total: int) -> None:
        self.total_tokens = total

    def elapsed(self) -> float:
        return round(time.time() - self.start_time, 3)

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "EventLogger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
