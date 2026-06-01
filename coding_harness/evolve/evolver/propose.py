"""PROPOSE — turn a weakness report into validated candidate patches.

The meta-agent takes the active version plus the diagnosis and emits one or more
:class:`PromptPolicyPatch` objects. Every proposal is run through
``PromptPolicyPatch.parse`` (the whitelist + bounds), so anything malformed or
out-of-bounds — or any attempt to touch a non-whitelisted surface — is rejected
before it can become a candidate.

Two proposers are provided:

* :func:`llm_propose` — calls an LLM with a strict JSON contract (the real
  meta-agent).
* :func:`heuristic_propose` — a deterministic, network-free fallback that maps
  known weakness ids to targeted edits. It makes the whole cycle runnable and
  testable offline, and serves as a backstop when the LLM returns nothing valid.

:func:`propose` ties them together: use the LLM when a ``complete`` callable is
given, falling back to the heuristic when it yields no valid patch.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from ..policy import PromptPolicyPatch, PromptPolicyVersion
from ..policy.patch import PatchValidationError
from .diagnose import WeaknessReport

# A callable matching LLMClient.complete(messages, ...) -> message-with-.content.
Complete = Callable[..., Any]

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


@dataclass
class ProposeResult:
    patches: list[PromptPolicyPatch] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)  # human-readable reasons


# -- heuristic proposer -------------------------------------------------------


def heuristic_propose(
    active: PromptPolicyVersion,
    report: WeaknessReport,
    *,
    max_candidates: int = 4,
    known_tools: set[str] | None = None,
) -> ProposeResult:
    """Deterministic weakness-id -> patch mapping (no network)."""
    result = ProposeResult()
    seen_ids = {w.id for w in report.weaknesses}

    raw: list[dict[str, Any]] = []

    if "premature_task_done" in seen_ids:
        raw.append(
            {
                "base_version": active.version,
                "rationale": (
                    "Episodes show task_done called while tests were failing; add a "
                    "hard pre-done test-check rule and tighten the task_done contract."
                ),
                "prompt_text": active.prompt_text.rstrip()
                + "\n\nCRITICAL: Never call task_done while any test is failing. "
                "Run run_tests as the final step and confirm a clean pass "
                "(exit_code 0, zero failures) before declaring the task complete.",
                "tool_description_overrides": {
                    "task_done": (
                        "Call this ONLY when the task is fully complete AND the most "
                        "recent run_tests passed with no failures. Provide a concise "
                        "summary of what was accomplished. This ends the agent run."
                    )
                },
            }
        )

    if "budget_exhaustion" in seen_ids:
        bumped = min(int(active.max_steps * 1.25) + 1, 200)
        if bumped > active.max_steps:
            raw.append(
                {
                    "base_version": active.version,
                    "rationale": (
                        "Runs are exhausting the step budget; give a modest step "
                        "headroom and steer the agent away from repeating calls."
                    ),
                    "max_steps": bumped,
                    "prompt_text": active.prompt_text.rstrip()
                    + "\n\nBe efficient: briefly plan before acting, and never repeat "
                    "an identical tool call — if a call did not help, change your "
                    "approach instead of retrying it.",
                }
            )

    # Any recurring failure signature or low success rate -> read-before-edit rule.
    if any(i.startswith("failure_signature:") for i in seen_ids) or (
        "low_success_rate" in seen_ids
    ):
        raw.append(
            {
                "base_version": active.version,
                "rationale": (
                    "Recurring failures / low success rate; reinforce reading the "
                    "full error and the code under change before editing."
                ),
                "prompt_text": active.prompt_text.rstrip()
                + "\n\nBefore editing, read the full error output and the relevant "
                "source so the change addresses the actual cause, not a guess.",
            }
        )

    _validate_into(raw[:max_candidates], result, known_tools)
    return result


# -- LLM proposer -------------------------------------------------------------

_SYSTEM = """\
You are an optimization meta-agent for an autonomous coding agent. Given the
agent's CURRENT prompt/policy and a diagnosis of its weaknesses, propose small,
targeted changes that should raise its benchmark success rate.

You may ONLY emit a JSON array of "patch" objects. Each patch object may contain
ONLY these fields:
  - "base_version"  (string, REQUIRED): the version you are patching.
  - "rationale"     (string, REQUIRED): why this change should help, citing the evidence.
  - "prompt_text"   (string, optional): a full replacement system prompt.
  - "tool_description_overrides" (object of toolName->description, optional).
  - "max_steps"     (integer, optional).
  - "max_total_tokens" (integer, optional).
  - "temperature"   (number, optional).
  - "parallel_tool_calls" (boolean, optional).

You must NOT include any other field. You cannot change sandboxing, timeouts,
deny-lists, the workspace, tool code, or the benchmark — those are not available.
Prefer minimal edits. Output ONLY the JSON array, nothing else.
"""


def llm_propose(
    active: PromptPolicyVersion,
    report: WeaknessReport,
    complete: Complete,
    *,
    max_candidates: int = 4,
    known_tools: set[str] | None = None,
) -> ProposeResult:
    """Ask an LLM for candidate patches and validate the response."""
    result = ProposeResult()
    user = json.dumps(
        {
            "active_version": {
                "version": active.version,
                "prompt_text": active.prompt_text,
                "max_steps": active.max_steps,
                "max_total_tokens": active.max_total_tokens,
                "temperature": active.temperature,
                "tool_descriptions": active.tool_descriptions,
            },
            "diagnosis": report.to_dict(),
            "max_candidates": max_candidates,
        },
        default=str,
    )
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user},
    ]
    try:
        message = complete(messages)
        content = getattr(message, "content", None) or ""
    except Exception as exc:  # noqa: BLE001 - a failed meta-call is not fatal
        result.rejected.append(f"llm call failed: {type(exc).__name__}: {exc}")
        return result

    raw = _extract_json_array(content)
    if raw is None:
        result.rejected.append("LLM response was not a JSON array of patches")
        return result

    _validate_into(raw[:max_candidates], result, known_tools)
    return result


def propose(
    active: PromptPolicyVersion,
    report: WeaknessReport,
    *,
    complete: Complete | None = None,
    max_candidates: int = 4,
    known_tools: set[str] | None = None,
) -> ProposeResult:
    """Propose candidate patches, preferring the LLM and falling back to heuristics."""
    if complete is not None:
        result = llm_propose(
            active, report, complete, max_candidates=max_candidates, known_tools=known_tools
        )
        if result.patches:
            return result
        # Fall back so a flaky/empty meta-call still yields something to evaluate.
        fb = heuristic_propose(
            active, report, max_candidates=max_candidates, known_tools=known_tools
        )
        fb.rejected = result.rejected + fb.rejected
        return fb
    return heuristic_propose(
        active, report, max_candidates=max_candidates, known_tools=known_tools
    )


# -- helpers ------------------------------------------------------------------


def _extract_json_array(content: str) -> list[dict[str, Any]] | None:
    text = content.strip()
    m = _JSON_BLOCK.search(text)
    if m:
        text = m.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return None
    return [d for d in data if isinstance(d, dict)]


def _validate_into(
    raw: list[dict[str, Any]], result: ProposeResult, known_tools: set[str] | None
) -> None:
    for item in raw:
        try:
            result.patches.append(PromptPolicyPatch.parse(item, known_tools=known_tools))
        except PatchValidationError as exc:
            result.rejected.append(str(exc))
