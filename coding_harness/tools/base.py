"""Tool abstraction, registry, and dispatch.

A :class:`Tool` pairs a JSON-schema description of its parameters with a Python
callable. The :class:`ToolRegistry` turns the registered tools into OpenAI tool
definitions (``schemas()``) and validates + dispatches an incoming ``tool_call``
back to the right callable, producing an OpenAI ``tool``-role message keyed by
the originating ``tool_call_id``.

Validation here is intentionally lightweight (presence of required params, no
unexpected params) since the model already receives a strict schema with
``additionalProperties: false``. Tool implementations are responsible for any
deeper semantic validation and signal failure by returning an error
:class:`ToolResult` rather than raising — so a bad call becomes an observation
the model can recover from instead of crashing the loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolResult:
    """The outcome of running a tool.

    ``content`` is the text the model will observe. ``is_error`` marks the result
    as a failure so the loop and logs can distinguish recoverable tool errors
    from successful observations.
    """

    content: str
    is_error: bool = False

    @classmethod
    def error(cls, message: str) -> "ToolResult":
        return cls(content=message, is_error=True)


@dataclass
class ToolParam:
    """A single tool parameter for JSON-schema generation."""

    name: str
    type: str  # JSON schema type, e.g. "string", "integer", "boolean"
    description: str
    required: bool = True
    # Optional extra JSON-schema keywords (e.g. {"enum": [...]}).
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Tool:
    """A callable tool exposed to the model.

    ``func`` receives validated keyword arguments and returns a
    :class:`ToolResult` (or a plain string, which is wrapped as a success).
    """

    name: str
    description: str
    params: list[ToolParam]
    func: Callable[..., ToolResult | str]

    def schema(self) -> dict[str, Any]:
        """Return this tool as an OpenAI tool definition.

        The parameter schema sets ``additionalProperties: false`` and lists only
        the truly required params in ``required`` so the model gets a tight
        contract.
        """
        properties: dict[str, Any] = {}
        required: list[str] = []
        for p in self.params:
            prop: dict[str, Any] = {"type": p.type, "description": p.description}
            prop.update(p.extra)
            properties[p.name] = prop
            if p.required:
                required.append(p.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }


class ToolRegistry:
    """Holds tools and routes ``tool_call`` objects to them."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI tool definitions for every registered tool."""
        return [t.schema() for t in self._tools.values()]

    def _validate_args(self, tool: Tool, args: dict[str, Any]) -> str | None:
        """Return an error string if ``args`` violate the tool schema, else None."""
        allowed = {p.name for p in tool.params}
        required = {p.name for p in tool.params if p.required}
        unexpected = set(args) - allowed
        if unexpected:
            return f"unexpected argument(s): {sorted(unexpected)}"
        missing = required - set(args)
        if missing:
            return f"missing required argument(s): {sorted(missing)}"
        return None

    def dispatch(self, tool_call: Any) -> dict[str, Any]:
        """Validate and run a single ``tool_call``; return a ``tool``-role message.

        ``tool_call`` follows the OpenAI shape (``.id`` and
        ``.function.name`` / ``.function.arguments``); a dict with the same keys
        is also accepted to keep tests independent of the SDK objects.

        Any failure — unknown tool, bad JSON, schema violation, or an exception
        raised inside the tool — is captured and returned as an error message so
        the agent loop never crashes on a bad call.
        """
        call_id, name, raw_args = _unpack_tool_call(tool_call)

        result = self._run(name, raw_args)
        prefix = "ERROR: " if result.is_error else ""
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": prefix + result.content,
        }

    def _run(self, name: str, raw_args: str) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult.error(f"unknown tool: {name!r}")

        try:
            args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError as exc:
            return ToolResult.error(f"could not parse arguments as JSON: {exc}")
        if not isinstance(args, dict):
            return ToolResult.error("tool arguments must be a JSON object")

        validation_error = self._validate_args(tool, args)
        if validation_error:
            return ToolResult.error(validation_error)

        try:
            out = tool.func(**args)
        except Exception as exc:  # noqa: BLE001 - surface as observation, don't crash
            return ToolResult.error(f"{type(exc).__name__}: {exc}")

        if isinstance(out, ToolResult):
            return out
        return ToolResult(content=str(out))


def _unpack_tool_call(tool_call: Any) -> tuple[str, str, str]:
    """Extract (id, function name, raw argument string) from a tool call.

    Supports both OpenAI SDK objects and plain dicts.
    """
    if isinstance(tool_call, dict):
        call_id = tool_call.get("id", "")
        fn = tool_call.get("function", {})
        name = fn.get("name", "")
        raw_args = fn.get("arguments", "") or ""
    else:
        call_id = getattr(tool_call, "id", "")
        fn = getattr(tool_call, "function", None)
        name = getattr(fn, "name", "")
        raw_args = getattr(fn, "arguments", "") or ""
    return call_id, name, raw_args
