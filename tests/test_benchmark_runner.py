"""Smoke tests for the Evaluation Gate runner (scripted FakeLLM, no network).

Exercises the full runner path: isolated workspace copy, real tool execution in
the jail, and the out-of-jail grader subprocess.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from coding_harness.config import Config
from coding_harness.llm_client import Usage
from coding_harness.evolve.benchmark.runner import (
    TASKS_DIR,
    discover_tasks,
    run_task,
)


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

    def complete(self, messages, tools=None, tool_choice="auto") -> FakeMessage:
        msg = self.scripted[self.calls]
        self.calls += 1
        self.usage.add(10, 5, 15)
        return msg


def _call(cid, name, **args):
    return FakeToolCall(id=cid, function=FakeFunction(name, json.dumps(args)))


def _base_config() -> Config:
    cfg = Config()
    cfg.logging.console = False
    # memory disabled by default; runner overrides workspace + trace per task.
    return cfg


def test_discover_tasks_finds_the_suite():
    names = {p.name for p in discover_tasks()}
    assert {"create_file", "fix_failing_test", "implement_function"} <= names


def test_run_task_create_file_passes(tmp_path):
    script = [
        FakeMessage(tool_calls=[_call("c1", "write_file", path="greeting.txt",
                                      content="Hello, world!\n")]),
        FakeMessage(tool_calls=[_call("c2", "task_done", summary="created greeting")]),
    ]
    task_dir = TASKS_DIR / "create_file"
    result = run_task(
        task_dir, _base_config(),
        llm_factory=lambda cfg: FakeLLM(scripted=script),
        work_root=tmp_path,
    )
    assert result.passed is True
    assert result.score == 1.0
    assert result.status == "done"


def test_run_task_create_file_fails_when_not_done(tmp_path):
    # Agent declares done without creating the file -> grader fails it.
    script = [FakeMessage(tool_calls=[_call("c1", "task_done", summary="nothing")])]
    task_dir = TASKS_DIR / "create_file"
    result = run_task(
        task_dir, _base_config(),
        llm_factory=lambda cfg: FakeLLM(scripted=script),
        work_root=tmp_path,
    )
    assert result.passed is False
    assert result.score == 0.0
    assert "not found" in result.details


def test_run_task_fix_failing_test_passes(tmp_path):
    # Replace the buggy subtraction with addition via edit_file, then verify.
    script = [
        FakeMessage(tool_calls=[_call("c1", "edit_file", path="calc.py",
                                      old_string="return a - b", new_string="return a + b")]),
        FakeMessage(tool_calls=[_call("c2", "run_tests")]),
        FakeMessage(tool_calls=[_call("c3", "task_done", summary="fixed add")]),
    ]
    task_dir = TASKS_DIR / "fix_failing_test"
    result = run_task(
        task_dir, _base_config(),
        llm_factory=lambda cfg: FakeLLM(scripted=script),
        work_root=tmp_path,
    )
    assert result.passed is True
    assert result.status == "done"
