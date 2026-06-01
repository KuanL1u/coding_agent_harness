"""Unit tests for the built-in ``exec`` plugin (run by the PT-M2 gate)."""

from __future__ import annotations

from coding_harness.sandbox import WorkspaceJail
from coding_harness.tools.exec import build_exec_tools


def _tools(tmp_path):
    tools = build_exec_tools(
        WorkspaceJail(tmp_path),
        command_timeout_s=30.0,
        max_output_bytes=10_000,
        deny_patterns=None,
        dry_run=False,
    )
    return {t.name: t for t in tools}


def test_run_python_inline(tmp_path):
    out = _tools(tmp_path)["run_python"].func(code="print('hi from snippet')")
    assert not out.is_error
    assert "hi from snippet" in out.content
