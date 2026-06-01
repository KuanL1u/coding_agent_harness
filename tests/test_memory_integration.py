"""End-to-end memory wiring: agent records episodes and injects experience.

Uses a scripted FakeLLM (no network) and the local embedder, with the store
pointed at tmp_path so nothing touches the repo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from coding_harness.agent import Agent
from coding_harness.config import Config
from coding_harness.llm_client import Usage
from coding_harness.evolve.memory.store import LocalFileStore


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
    scripted: list[FakeMessage]
    usage: Usage = field(default_factory=Usage)
    calls: int = 0
    # Captures the messages sent on the *first* request so the test can assert
    # what experience (if any) was injected into the prompt.
    first_request: list[dict[str, Any]] = field(default_factory=list)

    def complete(self, messages, tools=None, tool_choice="auto") -> FakeMessage:
        if not self.first_request:
            self.first_request = [dict(m) for m in messages]
        msg = self.scripted[self.calls]
        self.calls += 1
        self.usage.add(10, 5, 15)
        return msg


def _tool_call(cid, name, **args):
    return FakeToolCall(id=cid, function=FakeFunction(name, json.dumps(args)))


def _memory_config(tmp_path) -> Config:
    cfg = Config()
    cfg.sandbox.workspace_root = str(tmp_path / "ws")
    cfg.logging.trace_file = str(tmp_path / "trace.jsonl")
    cfg.logging.console = False
    cfg.memory.enabled = True
    cfg.memory.store_path = str(tmp_path / "mem")
    cfg.memory.embedding_model = "local"
    cfg.memory.distill_every_n_episodes = 0  # disable cadence for determinism
    return cfg


def test_episode_written_after_run(tmp_path):
    cfg = _memory_config(tmp_path)
    script = [FakeMessage(tool_calls=[_tool_call("c", "task_done", summary="done")])]
    agent = Agent.create(cfg, llm=FakeLLM(scripted=script))
    result = agent.run("create a greeting file")
    assert result.status == "done"

    store = LocalFileStore(cfg.memory.store_path)
    episodes = store.iter_episodes()
    assert len(episodes) == 1
    ep = episodes[0]
    assert ep.task == "create a greeting file"
    assert ep.outcome == "success"
    assert ep.task_embedding  # embedding persisted for later retrieval
    assert ep.config_snapshot.get("prompt_version", "").startswith("p_")


def test_experience_injected_on_second_similar_run(tmp_path):
    cfg = _memory_config(tmp_path)

    # First run seeds the store with a successful episode.
    script1 = [FakeMessage(tool_calls=[_tool_call("c", "task_done", summary="fixed calc.py")])]
    Agent.create(cfg, llm=FakeLLM(scripted=script1)).run("fix the failing test in calc.py")

    # Second, similar task: experience should be injected as a system message.
    fake2 = FakeLLM(scripted=[FakeMessage(tool_calls=[_tool_call("c", "task_done", summary="ok")])])
    Agent.create(cfg, llm=fake2).run("fix the failing test in calc.py again")

    system_msgs = [m for m in fake2.first_request if m["role"] == "system"]
    injected = "\n".join(m["content"] for m in system_msgs)
    assert "Relevant experience" in injected
    assert "calc.py" in injected


def test_memory_disabled_by_default_writes_nothing(tmp_path):
    cfg = Config()
    cfg.sandbox.workspace_root = str(tmp_path / "ws")
    cfg.logging.trace_file = str(tmp_path / "trace.jsonl")
    cfg.logging.console = False
    # memory.enabled defaults to False
    script = [FakeMessage(tool_calls=[_tool_call("c", "task_done", summary="done")])]
    agent = Agent.create(cfg, llm=FakeLLM(scripted=script))
    assert agent.memory is None
    agent.run("do a thing")
    assert not (tmp_path / "mem").exists()
