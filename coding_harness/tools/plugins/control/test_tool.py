"""Unit tests for the built-in ``control`` plugin (run by the PT-M2 gate)."""

from __future__ import annotations

from coding_harness.tools.control import TaskState, build_control_tools


def test_task_done_sets_state():
    state = TaskState()
    task_done = {t.name: t for t in build_control_tools(state)}["task_done"]
    result = task_done.func(summary="all done")
    assert state.done is True
    assert state.summary == "all done"
    assert not result.is_error
