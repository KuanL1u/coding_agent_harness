"""Tests for LLMClient request construction and retry behavior.

These use a hand-rolled fake stand-in for the OpenAI client so no network or
real ``openai`` types are required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from coding_harness.llm_client import EmptyResponseError, LLMClient


@dataclass
class _FakeMessage:
    content: str = "ok"
    tool_calls: Any = None


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakeUsage:
    prompt_tokens: int = 1
    completion_tokens: int = 1
    total_tokens: int = 2


@dataclass
class _FakeResponse:
    choices: list[_FakeChoice]
    usage: _FakeUsage = field(default_factory=_FakeUsage)


@dataclass
class _FakeCompletions:
    """Records kwargs and returns a scripted sequence of responses."""

    scripted: list[Any]
    seen_kwargs: list[dict] = field(default_factory=list)
    calls: int = 0

    def create(self, **kwargs):
        self.seen_kwargs.append(kwargs)
        item = self.scripted[min(self.calls, len(self.scripted) - 1)]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return item


class _FakeClient:
    def __init__(self, completions: _FakeCompletions):
        self.chat = type("Chat", (), {"completions": completions})()


def _client(completions: _FakeCompletions, **overrides) -> LLMClient:
    return LLMClient(model="test-model", _client=_FakeClient(completions), **overrides)


_TOOLS = [{"type": "function", "function": {"name": "noop", "parameters": {}}}]


def test_parallel_tool_calls_omitted_when_disabled():
    completions = _FakeCompletions(scripted=[_FakeResponse([_FakeChoice(_FakeMessage())])])
    client = _client(completions, parallel_tool_calls=False)
    client.complete([{"role": "user", "content": "hi"}], tools=_TOOLS)
    # Disabled -> the param must not be sent at all (some servers reject it).
    assert "parallel_tool_calls" not in completions.seen_kwargs[0]


def test_parallel_tool_calls_sent_when_enabled():
    completions = _FakeCompletions(scripted=[_FakeResponse([_FakeChoice(_FakeMessage())])])
    client = _client(completions, parallel_tool_calls=True)
    client.complete([{"role": "user", "content": "hi"}], tools=_TOOLS)
    assert completions.seen_kwargs[0]["parallel_tool_calls"] is True


def test_parallel_tool_calls_omitted_when_no_tools():
    completions = _FakeCompletions(scripted=[_FakeResponse([_FakeChoice(_FakeMessage())])])
    client = _client(completions, parallel_tool_calls=True)
    client.complete([{"role": "user", "content": "hi"}])
    # No tools -> no tool params at all.
    assert "parallel_tool_calls" not in completions.seen_kwargs[0]
    assert "tools" not in completions.seen_kwargs[0]


def test_empty_choices_is_retried_then_succeeds():
    good = _FakeResponse([_FakeChoice(_FakeMessage(content="recovered"))])
    completions = _FakeCompletions(scripted=[_FakeResponse([]), good])
    client = _client(completions, base_backoff=0.0)
    msg = client.complete([{"role": "user", "content": "hi"}])
    assert msg.content == "recovered"
    assert completions.calls == 2  # first empty response was retried


def test_empty_choices_raises_after_exhausting_retries():
    completions = _FakeCompletions(scripted=[_FakeResponse([])])
    client = _client(completions, max_retries=2, base_backoff=0.0)
    with pytest.raises(EmptyResponseError):
        client.complete([{"role": "user", "content": "hi"}])
    assert completions.calls == 3  # initial try + 2 retries


def test_retries_emit_observability_events():
    good = _FakeResponse([_FakeChoice(_FakeMessage(content="recovered"))])
    completions = _FakeCompletions(scripted=[_FakeResponse([]), _FakeResponse([]), good])
    events: list[tuple[str, dict]] = []
    client = _client(completions, base_backoff=0.0)
    client.event_hook = lambda event, fields: events.append((event, fields))

    client.complete([{"role": "user", "content": "hi"}])

    # One retry event per backoff (two empty responses), no terminal error.
    retries = [f for e, f in events if e == "llm_retry"]
    assert [f["attempt"] for f in retries] == [1, 2]
    assert all(f["error_type"] == "EmptyResponseError" for f in retries)
    assert not any(e == "llm_error" for e, _ in events)


def test_exhausted_retries_emit_llm_error_event():
    completions = _FakeCompletions(scripted=[_FakeResponse([])])
    events: list[tuple[str, dict]] = []
    client = _client(completions, max_retries=2, base_backoff=0.0)
    client.event_hook = lambda event, fields: events.append((event, fields))

    with pytest.raises(EmptyResponseError):
        client.complete([{"role": "user", "content": "hi"}])

    errors = [f for e, f in events if e == "llm_error"]
    assert len(errors) == 1
    assert errors[0]["error_type"] == "EmptyResponseError"
    assert errors[0]["attempts"] == 3  # initial try + 2 retries
