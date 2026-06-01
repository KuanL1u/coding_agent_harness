"""Unit tests for the built-in ``files`` plugin.

Run by the tool-validation gate, not by the harness's own test suite
(``conftest.py`` excludes ``coding_harness/tools/plugins/*`` from collection).
The file tools themselves are also covered by ``tests/test_tools.py``.
"""

from __future__ import annotations

from coding_harness.config import Config
from coding_harness.sandbox import WorkspaceJail
from coding_harness.tools.context import make_tool_context
from coding_harness.tools.files import build_file_tools


def _tools(tmp_path):
    ctx = make_tool_context(Config(), WorkspaceJail(tmp_path))
    return {t.name: t for t in build_file_tools(ctx.jail, ctx.config.max_output_bytes)}


def test_write_then_read(tmp_path):
    tools = _tools(tmp_path)
    tools["write_file"].func(path="a.txt", content="hi")
    assert "hi" in tools["read_file"].func(path="a.txt").content


def test_path_escape_is_rejected(tmp_path):
    tools = _tools(tmp_path)
    assert tools["read_file"].func(path="../../etc/passwd").is_error
