"""Tests for execution tools (subprocess wiring and temp-file hygiene)."""

from __future__ import annotations

import pytest

from coding_harness.sandbox import WorkspaceJail
from coding_harness.tools.exec import build_exec_tools


@pytest.fixture
def jail(tmp_path):
    return WorkspaceJail(tmp_path)


def _exec_tools(jail, *, dry_run=False):
    tools = build_exec_tools(
        jail,
        command_timeout_s=30.0,
        max_output_bytes=10_000,
        deny_patterns=None,
        dry_run=dry_run,
    )
    return {t.name: t for t in tools}


def test_run_python_inline_executes(jail):
    run_python = _exec_tools(jail)["run_python"]
    result = run_python.func(code="print('hi from snippet')")
    assert not result.is_error
    assert "hi from snippet" in result.content


def test_run_python_inline_leaves_no_temp_file(jail, tmp_path):
    run_python = _exec_tools(jail)["run_python"]
    for _ in range(3):
        run_python.func(code="print('x')")
    # No tmp*.py snippets should be left behind in the workspace.
    assert list(tmp_path.glob("*.py")) == []


def test_run_python_inline_temp_file_cleaned_on_dry_run(jail, tmp_path):
    run_python = _exec_tools(jail, dry_run=True)["run_python"]
    result = run_python.func(code="print('x')")
    assert "dry_run" in result.content
    # Even though the snippet is written before the dry-run check, nothing leaks.
    assert list(tmp_path.glob("*.py")) == []
