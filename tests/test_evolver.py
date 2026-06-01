"""Tests for the Layer 3 evolver: diagnose, decide, agent policy wiring, and a
full end-to-end cycle (heuristic proposer + scripted FakeLLM, no network)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from coding_harness.agent import Agent
from coding_harness.config import Config
from coding_harness.llm_client import Usage
from coding_harness.evolve.memory.episode import build_episode
from coding_harness.evolve.memory.store import LocalFileStore
from coding_harness.evolve.policy import (
    PromptPolicyRegistry,
    PromptPolicyVersion,
    seed_version,
)
from coding_harness.evolve.evolver.diagnose import diagnose
from coding_harness.evolve.evolver.decide import decide_one
from coding_harness.evolve.evolver.cycle import run_cycle


# -- shared fakes -------------------------------------------------------------


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


def _tc(cid, name, **args):
    return FakeToolCall(id=cid, function=FakeFunction(name, json.dumps(args)))


def _assistant(tool_calls):
    return {"role": "assistant", "content": "", "tool_calls": tool_calls}


def _call_dict(cid, name, **args):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


# -- diagnose -----------------------------------------------------------------


def _premature_episode():
    # outcome success (status done) but the last run_tests observation failed.
    msgs = [
        _assistant([_call_dict("c1", "run_tests")]),
        {"role": "tool", "tool_call_id": "c1", "content": "ERROR: exit_code: 1\nE   AssertionError"},
        _assistant([_call_dict("c2", "task_done", summary="done")]),
    ]
    return build_episode("fix calc", msgs, status="done", steps=2, tokens=10,
                         wall_clock_s=1.0, trace_path="t", config_snapshot={})


def test_diagnose_detects_premature_task_done_and_budget(tmp_path):
    store = LocalFileStore(tmp_path / "mem")
    for _ in range(3):
        store.add_episode(_premature_episode())
    # a budget-stopped run
    store.add_episode(build_episode("loop", [], status="max_steps", steps=30,
                                    tokens=99, wall_clock_s=5.0, trace_path="t",
                                    config_snapshot={}))
    report = diagnose(store)
    ids = {w.id for w in report.weaknesses}
    assert "premature_task_done" in ids
    assert "budget_exhaustion" in ids
    assert report.sample_size == 4


def test_diagnose_empty_store():
    store = LocalFileStore  # not instantiated
    # use a real empty store
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        report = diagnose(LocalFileStore(d))
    assert report.sample_size == 0
    assert report.weaknesses == []


# -- decide -------------------------------------------------------------------


def test_decide_adopts_clear_win_within_budget():
    active = SimpleNamespace(success_rate=0.5, avg_tokens=100, avg_steps=10)
    cand = SimpleNamespace(success_rate=0.7, avg_tokens=105, avg_steps=10)
    d = decide_one(active, cand, adopt_epsilon=0.03, max_cost_regression=0.10)
    assert d.adopt is True


def test_decide_rejects_marginal_gain():
    active = SimpleNamespace(success_rate=0.50, avg_tokens=100, avg_steps=10)
    cand = SimpleNamespace(success_rate=0.51, avg_tokens=100, avg_steps=10)
    d = decide_one(active, cand, adopt_epsilon=0.03, max_cost_regression=0.10)
    assert d.adopt is False
    assert "success gain" in d.reason


def test_decide_rejects_cost_regression():
    active = SimpleNamespace(success_rate=0.5, avg_tokens=100, avg_steps=10)
    cand = SimpleNamespace(success_rate=0.9, avg_tokens=200, avg_steps=10)  # +100% tokens
    d = decide_one(active, cand, adopt_epsilon=0.03, max_cost_regression=0.10)
    assert d.adopt is False
    assert "cost regression" in d.reason


# -- agent applies the active policy ------------------------------------------


@dataclass
class ScriptLLM:
    scripted: list[FakeMessage]
    usage: Usage = field(default_factory=Usage)
    calls: int = 0
    first_request: list[dict[str, Any]] = field(default_factory=list)

    def complete(self, messages, tools=None, tool_choice="auto"):
        if not self.first_request:
            self.first_request = [dict(m) for m in messages]
        msg = self.scripted[self.calls]
        self.calls += 1
        self.usage.add(10, 5, 15)
        return msg


def test_agent_applies_active_policy(tmp_path):
    cfg = Config()
    cfg.sandbox.workspace_root = str(tmp_path / "ws")
    cfg.logging.trace_file = str(tmp_path / "trace.jsonl")
    cfg.logging.console = False
    cfg.evolve.enabled = True
    cfg.evolve.policy_dir = str(tmp_path / "policy")

    reg = PromptPolicyRegistry(cfg.evolve.policy_dir)
    reg.seed(
        PromptPolicyVersion(
            version="p1",
            prompt_text="CUSTOM ACTIVE PROMPT",
            max_steps=7,
            max_total_tokens=12345,
            temperature=0.33,
            parallel_tool_calls=True,
            tool_descriptions={"task_done": "OVERRIDDEN task_done description"},
        )
    )

    llm = ScriptLLM(scripted=[FakeMessage(tool_calls=[_tc("c", "task_done", summary="ok")])])
    agent = Agent.create(cfg, llm=llm)

    assert agent.system_prompt == "CUSTOM ACTIVE PROMPT"
    assert agent.policy_version == "p1"
    assert cfg.loop.max_steps == 7
    assert cfg.llm.temperature == 0.33
    assert agent.registry.get("task_done").description == "OVERRIDDEN task_done description"

    agent.run("do it")
    # The custom prompt actually reached the model, and the episode is stamped p1.
    assert agent.system_prompt in (m["content"] for m in llm.first_request if m["role"] == "system")


def test_agent_seeds_registry_when_enabled_and_empty(tmp_path):
    cfg = Config()
    cfg.sandbox.workspace_root = str(tmp_path / "ws")
    cfg.logging.trace_file = str(tmp_path / "trace.jsonl")
    cfg.logging.console = False
    cfg.evolve.enabled = True
    cfg.evolve.policy_dir = str(tmp_path / "policy")

    llm = ScriptLLM(scripted=[FakeMessage(tool_calls=[_tc("c", "task_done", summary="ok")])])
    agent = Agent.create(cfg, llm=llm)
    # Bootstrapped p1 reproduces v1 behaviour (the hardcoded system prompt).
    assert agent.policy_version == "p1"
    reg = PromptPolicyRegistry(cfg.evolve.policy_dir)
    assert reg.get_active().version == "p1"


# -- end-to-end cycle ---------------------------------------------------------


def _make_task(tmp_path):
    """A one-task benchmark: pass iff the agent creates done.txt."""
    tasks = tmp_path / "tasks"
    t = tasks / "make_done"
    (t / "setup").mkdir(parents=True)
    (t / "task.md").write_text("Create a file done.txt, then call task_done.")
    (t / "grade.py").write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "def grade(ws):\n"
        "    ok = (Path(ws) / 'done.txt').is_file()\n"
        "    return {'passed': ok, 'score': 1.0 if ok else 0.0,\n"
        "            'details': '' if ok else 'missing done.txt'}\n"
        "if __name__ == '__main__':\n"
        "    print(json.dumps(grade(Path(sys.argv[1]))))\n"
    )
    return tasks


class MarkerLLM:
    """Solves the task only when the system prompt contains ``marker``.

    This lets a candidate whose prompt gained the marker outscore the active
    version — exercising the full DIAGNOSE->...->ADOPT path deterministically.
    """

    def __init__(self, marker: str = "CRITICAL") -> None:
        self.usage = Usage()
        self.marker = marker
        self.calls = 0

    def complete(self, messages, tools=None, tool_choice="auto"):
        self.usage.add(10, 5, 15)
        self.calls += 1
        system = " ".join(
            (m.get("content") or "") for m in messages if m.get("role") == "system"
        )
        if self.marker in system:
            if self.calls == 1:
                return FakeMessage(tool_calls=[_tc("c1", "write_file", path="done.txt", content="ok")])
            return FakeMessage(tool_calls=[_tc("c2", "task_done", summary="created")])
        return FakeMessage(tool_calls=[_tc("c1", "task_done", summary="skipped")])


class AlwaysSolveLLM:
    def __init__(self) -> None:
        self.usage = Usage()
        self.calls = 0

    def complete(self, messages, tools=None, tool_choice="auto"):
        self.usage.add(10, 5, 15)
        self.calls += 1
        if self.calls == 1:
            return FakeMessage(tool_calls=[_tc("c1", "write_file", path="done.txt", content="ok")])
        return FakeMessage(tool_calls=[_tc("c2", "task_done", summary="created")])


def _cycle_config(tmp_path) -> Config:
    cfg = Config()
    cfg.logging.console = False
    cfg.memory.enabled = False  # we inject a seeded store; keep benchmark runs clean
    cfg.evolve.policy_dir = str(tmp_path / "policy")
    return cfg


def _seed_premature_store(tmp_path) -> LocalFileStore:
    store = LocalFileStore(tmp_path / "mem")
    for _ in range(3):
        store.add_episode(_premature_episode())
    return store


def test_cycle_adopts_when_candidate_beats_active(tmp_path):
    cfg = _cycle_config(tmp_path)
    cfg.evolve.adopt_epsilon = 0.03
    cfg.evolve.max_cost_regression = 10.0  # synthetic: ignore the extra step's cost
    store = _seed_premature_store(tmp_path)
    reg = PromptPolicyRegistry(cfg.evolve.policy_dir)
    tasks = _make_task(tmp_path)

    result = run_cycle(
        cfg,
        complete=None,  # heuristic proposer
        llm_factory=lambda c: MarkerLLM(marker="CRITICAL"),
        tasks_dir=tasks,
        store=store,
        registry=reg,
        work_root=tmp_path,
        do_commit=False,
    )

    assert any(w["id"] == "premature_task_done" for w in result.weaknesses)
    assert result.proposed >= 1
    assert result.adopted_version == "p2"
    # The adopted version is now active and carries the new prompt rule.
    active = reg.get_active()
    assert active.version == "p2"
    assert "CRITICAL" in active.prompt_text
    assert active.parent == "p1"


def test_cycle_rejects_when_no_candidate_beats_active(tmp_path):
    cfg = _cycle_config(tmp_path)
    cfg.evolve.adopt_epsilon = 0.03
    cfg.evolve.max_cost_regression = 0.10
    store = _seed_premature_store(tmp_path)
    reg = PromptPolicyRegistry(cfg.evolve.policy_dir)
    tasks = _make_task(tmp_path)

    # Active already solves the task, so no candidate can beat it -> reject.
    result = run_cycle(
        cfg,
        complete=None,
        llm_factory=lambda c: AlwaysSolveLLM(),
        tasks_dir=tasks,
        store=store,
        registry=reg,
        work_root=tmp_path,
        do_commit=False,
    )

    assert result.adopted_version is None
    assert reg.get_active().version == "p1"
    # Rejected candidates are archived for audit.
    assert (reg.archive_dir).is_dir()
    assert any(reg.archive_dir.glob("*.json"))
