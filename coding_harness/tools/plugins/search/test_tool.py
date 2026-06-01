"""Unit tests for the built-in ``search`` plugin (run by the validation gate)."""

from __future__ import annotations

from coding_harness.config import Config
from coding_harness.sandbox import WorkspaceJail
from coding_harness.tools.context import make_tool_context
from coding_harness.tools.search import build_search_tools


def _tools(tmp_path):
    ctx = make_tool_context(Config(), WorkspaceJail(tmp_path))
    return {t.name: t for t in build_search_tools(ctx.jail, ctx.config.max_output_bytes)}


def test_grep_finds_match(tmp_path):
    (tmp_path / "f.py").write_text("def hello():\n    pass\n")
    tools = _tools(tmp_path)
    out = tools["grep_search"].func(pattern="def hello")
    assert "f.py" in out.content


def test_glob_lists_files(tmp_path):
    (tmp_path / "f.py").write_text("x = 1\n")
    tools = _tools(tmp_path)
    assert "f.py" in tools["glob_search"].func(pattern="**/*.py").content
