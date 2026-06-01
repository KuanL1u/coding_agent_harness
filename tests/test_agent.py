"""Tests for the ReAct loop using a mocked LLM client (no network)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from coding_harness.agent import Agent
from coding_harness.config import Config
from coding_harness.llm_client import Usage


# -- minimal fakes mirroring the OpenAI message/tool-call shape ---------------


@dataclass
class FakeFunction:
    name: str
    arguments: str


@dataclass
class FakeToolCall:
    id: str
    function: FakeFunction


@dataclass
class FakeMessage:
    content: str | None = None
    tool_calls: list[FakeToolCall] | None = None


@dataclass
class FakeLLM:
    """Returns a pre-scripted sequence of assistant messages."""

    scripted: list[FakeMessage]
    usage: Usage = field(default_factory=Usage)
    calls: int = 0

    def complete(self, messages, tools=None, tool_choice="auto") -> FakeMessage:
        msg = self.scripted[self.calls]
        self.calls += 1
        # Pretend each request consumed some tokens.
        self.usage.add(10, 5, 15)
        return msg


def _tool_call(call_id: str, name: str, **args: Any) -> FakeToolCall:
    return FakeToolCall(id=call_id, function=FakeFunction(name, json.dumps(args)))


def _config(tmp_path) -> Config:
    cfg = Config()
    cfg.sandbox.workspace_root = str(tmp_path / "ws")
    cfg.logging.trace_file = str(tmp_path / "trace.jsonl")
    cfg.logging.console = False
    return cfg


def test_loop_terminates_on_task_done(tmp_path):
    cfg = _config(tmp_path)
    script = [
        FakeMessage(
            content="I'll create the file.",
            tool_calls=[_tool_call("c1", "write_file", path="out.txt", content="hi")],
        ),
        FakeMessage(
            content="Done.",
            tool_calls=[_tool_call("c2", "task_done", summary="created out.txt")],
        ),
    ]
    agent = Agent.create(cfg, llm=FakeLLM(scripted=script))
    result = agent.run("create out.txt")

    assert result.status == "done"
    assert result.summary == "created out.txt"
    assert (tmp_path / "ws" / "out.txt").read_text() == "hi"
    # system + user + (assistant + tool) * 2
    assert result.messages[0]["role"] == "system"
    assert any(m["role"] == "tool" for m in result.messages)


def test_loop_handles_parallel_tool_calls(tmp_path):
    cfg = _config(tmp_path)
    script = [
        FakeMessage(
            content="Writing two files at once.",
            tool_calls=[
                _tool_call("a", "write_file", path="a.txt", content="A"),
                _tool_call("b", "write_file", path="b.txt", content="B"),
            ],
        ),
        FakeMessage(tool_calls=[_tool_call("d", "task_done", summary="wrote both")]),
    ]
    agent = Agent.create(cfg, llm=FakeLLM(scripted=script))
    result = agent.run("write a and b")

    assert result.status == "done"
    assert (tmp_path / "ws" / "a.txt").read_text() == "A"
    assert (tmp_path / "ws" / "b.txt").read_text() == "B"
    tool_msgs = [m for m in result.messages if m["role"] == "tool"]
    # Both parallel calls produced their own id-keyed tool message.
    assert {"a", "b"}.issubset({m["tool_call_id"] for m in tool_msgs})


def test_loop_breaks_on_prose_answer(tmp_path):
    cfg = _config(tmp_path)
    script = [FakeMessage(content="The answer is 42.", tool_calls=None)]
    agent = Agent.create(cfg, llm=FakeLLM(scripted=script))
    result = agent.run("what is the answer")

    assert result.status == "answered"
    assert "42" in result.summary


def test_loop_enforces_max_steps(tmp_path):
    cfg = _config(tmp_path)
    cfg.loop.max_steps = 2
    # Always asks to read a file; never calls task_done -> should hit max_steps.
    looping = FakeMessage(
        content="reading",
        tool_calls=[_tool_call("c", "list_dir", path=".")],
    )
    agent = Agent.create(cfg, llm=FakeLLM(scripted=[looping, looping, looping]))
    result = agent.run("loop forever")

    assert result.status == "max_steps"
    assert result.steps == 2


def test_loop_enforces_token_budget(tmp_path):
    cfg = _config(tmp_path)
    cfg.loop.max_total_tokens = 15  # one request's worth
    script = [
        FakeMessage(content="x", tool_calls=[_tool_call("c", "list_dir", path=".")]),
        FakeMessage(content="x", tool_calls=[_tool_call("c2", "list_dir", path=".")]),
    ]
    agent = Agent.create(cfg, llm=FakeLLM(scripted=script))
    result = agent.run("burn tokens")

    assert result.status == "max_tokens"


def test_tool_error_does_not_crash_loop(tmp_path):
    cfg = _config(tmp_path)
    script = [
        # First call references a missing file -> tool error observation.
        FakeMessage(
            content="reading",
            tool_calls=[_tool_call("c1", "read_file", path="missing.txt")],
        ),
        FakeMessage(tool_calls=[_tool_call("c2", "task_done", summary="recovered")]),
    ]
    agent = Agent.create(cfg, llm=FakeLLM(scripted=script))
    result = agent.run("read a missing file then finish")

    assert result.status == "done"
    tool_msgs = [m for m in result.messages if m["role"] == "tool"]
    assert any(m["content"].startswith("ERROR: ") for m in tool_msgs)


def test_trace_file_written(tmp_path):
    cfg = _config(tmp_path)
    script = [FakeMessage(tool_calls=[_tool_call("c", "task_done", summary="ok")])]
    agent = Agent.create(cfg, llm=FakeLLM(scripted=script))
    agent.run("finish immediately")

    trace = (tmp_path / "trace.jsonl").read_text().strip().splitlines()
    events = [json.loads(line)["event"] for line in trace]
    assert "llm_request" in events
    assert "assistant_message" in events
    assert "loop_end" in events
