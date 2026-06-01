"""The ONLY shape the evolver may emit — the structural safety boundary.

A :class:`PromptPolicyPatch` is a strict, schema-validated object. The evolver
(an LLM) can propose nothing else: :meth:`PromptPolicyPatch.parse` rejects any
field outside the whitelist, so there is no representable way for a proposal to
touch the sandbox, deny-list, timeouts, workspace root, tool execution code, or
the benchmark. Out-of-range values are rejected too, bounding even the
whitelisted knobs.

Applying a validated patch to a base :class:`PromptPolicyVersion` (via
:func:`materialize`) yields a new candidate version; nothing is mutated in place.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .registry import PromptPolicyVersion

# The complete set of fields a patch may carry. Anything else is a hard reject.
_ALLOWED_FIELDS = {
    "base_version",
    "rationale",
    "prompt_text",
    "tool_description_overrides",
    "max_steps",
    "max_total_tokens",
    "temperature",
    "parallel_tool_calls",
}

# Bounds on the tunable knobs (defence in depth on top of the field whitelist).
_MAX_STEPS_RANGE = (1, 200)
_MAX_TOKENS_RANGE = (1_000, 10_000_000)
_TEMPERATURE_RANGE = (0.0, 2.0)
_PROMPT_MAX_CHARS = 20_000
_TOOL_DESC_MAX_CHARS = 2_000


class PatchValidationError(ValueError):
    """Raised when a proposed patch violates the whitelist or value bounds."""


@dataclass
class PromptPolicyPatch:
    """A validated, whitelisted edit against ``base_version``."""

    base_version: str
    rationale: str
    prompt_text: str | None = None
    tool_description_overrides: dict[str, str] | None = None
    max_steps: int | None = None
    max_total_tokens: int | None = None
    temperature: float | None = None
    parallel_tool_calls: bool | None = None

    @classmethod
    def parse(
        cls, data: dict[str, Any], known_tools: set[str] | None = None
    ) -> "PromptPolicyPatch":
        """Validate ``data`` into a patch, or raise :class:`PatchValidationError`.

        ``known_tools`` (when supplied) restricts ``tool_description_overrides``
        to real tool names so a patch cannot invent tools.
        """
        if not isinstance(data, dict):
            raise PatchValidationError("patch must be a JSON object")

        unknown = set(data) - _ALLOWED_FIELDS
        if unknown:
            raise PatchValidationError(
                f"patch contains non-whitelisted field(s): {sorted(unknown)}"
            )

        base_version = data.get("base_version")
        if not isinstance(base_version, str) or not base_version:
            raise PatchValidationError("'base_version' (non-empty string) is required")
        rationale = data.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            raise PatchValidationError("'rationale' (non-empty string) is required")

        patch = cls(base_version=base_version, rationale=rationale.strip())

        if "prompt_text" in data:
            pt = data["prompt_text"]
            if not isinstance(pt, str) or not pt.strip():
                raise PatchValidationError("'prompt_text' must be a non-empty string")
            if len(pt) > _PROMPT_MAX_CHARS:
                raise PatchValidationError(
                    f"'prompt_text' exceeds {_PROMPT_MAX_CHARS} chars"
                )
            patch.prompt_text = pt

        if "tool_description_overrides" in data:
            patch.tool_description_overrides = _validate_tool_overrides(
                data["tool_description_overrides"], known_tools
            )

        if "max_steps" in data:
            patch.max_steps = _validate_int("max_steps", data["max_steps"], _MAX_STEPS_RANGE)
        if "max_total_tokens" in data:
            patch.max_total_tokens = _validate_int(
                "max_total_tokens", data["max_total_tokens"], _MAX_TOKENS_RANGE
            )
        if "temperature" in data:
            patch.temperature = _validate_float(
                "temperature", data["temperature"], _TEMPERATURE_RANGE
            )
        if "parallel_tool_calls" in data:
            ptc = data["parallel_tool_calls"]
            if not isinstance(ptc, bool):
                raise PatchValidationError("'parallel_tool_calls' must be a boolean")
            patch.parallel_tool_calls = ptc

        if not patch.changes():
            raise PatchValidationError(
                "patch makes no change (no tunable field set besides metadata)"
            )
        return patch

    def changes(self) -> dict[str, Any]:
        """The set fields that actually change something (excludes metadata)."""
        out: dict[str, Any] = {}
        for name in (
            "prompt_text",
            "tool_description_overrides",
            "max_steps",
            "max_total_tokens",
            "temperature",
            "parallel_tool_calls",
        ):
            value = getattr(self, name)
            if value is not None:
                out[name] = value
        return out


def materialize(
    patch: PromptPolicyPatch,
    base: PromptPolicyVersion,
    new_version_id: str,
) -> PromptPolicyVersion:
    """Apply ``patch`` to ``base`` -> a new candidate :class:`PromptPolicyVersion`.

    ``tool_description_overrides`` are merged onto the base's overrides (a patch
    can add or replace individual tool descriptions without dropping the rest).
    """
    tool_descriptions = dict(base.tool_descriptions)
    if patch.tool_description_overrides:
        tool_descriptions.update(patch.tool_description_overrides)

    return PromptPolicyVersion(
        version=new_version_id,
        prompt_text=patch.prompt_text if patch.prompt_text is not None else base.prompt_text,
        max_steps=patch.max_steps if patch.max_steps is not None else base.max_steps,
        max_total_tokens=(
            patch.max_total_tokens
            if patch.max_total_tokens is not None
            else base.max_total_tokens
        ),
        temperature=(
            patch.temperature if patch.temperature is not None else base.temperature
        ),
        parallel_tool_calls=(
            patch.parallel_tool_calls
            if patch.parallel_tool_calls is not None
            else base.parallel_tool_calls
        ),
        tool_descriptions=tool_descriptions,
        parent=base.version,
        rationale=patch.rationale,
        created=time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
    )


def _validate_tool_overrides(
    value: Any, known_tools: set[str] | None
) -> dict[str, str]:
    if not isinstance(value, dict):
        raise PatchValidationError("'tool_description_overrides' must be an object")
    out: dict[str, str] = {}
    for name, desc in value.items():
        if not isinstance(name, str) or not isinstance(desc, str):
            raise PatchValidationError(
                "'tool_description_overrides' keys and values must be strings"
            )
        if not desc.strip():
            raise PatchValidationError(f"empty description for tool {name!r}")
        if len(desc) > _TOOL_DESC_MAX_CHARS:
            raise PatchValidationError(
                f"description for {name!r} exceeds {_TOOL_DESC_MAX_CHARS} chars"
            )
        if known_tools is not None and name not in known_tools:
            raise PatchValidationError(
                f"unknown tool {name!r}; cannot override its description"
            )
        out[name] = desc
    return out


def _validate_int(name: str, value: Any, bounds: tuple[int, int]) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PatchValidationError(f"'{name}' must be an integer")
    lo, hi = bounds
    if not (lo <= value <= hi):
        raise PatchValidationError(f"'{name}'={value} out of range [{lo}, {hi}]")
    return value


def _validate_float(name: str, value: Any, bounds: tuple[float, float]) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PatchValidationError(f"'{name}' must be a number")
    lo, hi = bounds
    if not (lo <= float(value) <= hi):
        raise PatchValidationError(f"'{name}'={value} out of range [{lo}, {hi}]")
    return float(value)
