"""Tests for file tools, the registry schema generation, and dispatch."""

from __future__ import annotations

import json

import pytest

from coding_harness.sandbox import WorkspaceJail
from coding_harness.tools.base import Tool, ToolParam, ToolRegistry, ToolResult
from coding_harness.tools.files import build_file_tools


@pytest.fixture
def jail(tmp_path):
    return WorkspaceJail(tmp_path)


@pytest.fixture
def registry(jail):
    reg = ToolRegistry()
    for tool in build_file_tools(jail, max_output_bytes=10_000):
        reg.register(tool)
    return reg


def _edit_tool(jail):
    return {t.name: t for t in build_file_tools(jail, 10_000)}["edit_file"]


def test_edit_file_unique_match(jail, tmp_path):
    target = tmp_path / "f.txt"
    target.write_text("alpha beta gamma")
    edit = _edit_tool(jail)
    result = edit.func(path="f.txt", old_string="beta", new_string="BETA")
    assert not result.is_error
    assert target.read_text() == "alpha BETA gamma"


def test_edit_file_non_unique_without_replace_all(jail, tmp_path):
    target = tmp_path / "f.txt"
    target.write_text("x x x")
    edit = _edit_tool(jail)
    result = edit.func(path="f.txt", old_string="x", new_string="y")
    assert result.is_error
    assert "not unique" in result.content
    assert target.read_text() == "x x x"  # unchanged


def test_edit_file_replace_all(jail, tmp_path):
    target = tmp_path / "f.txt"
    target.write_text("x x x")
    edit = _edit_tool(jail)
    result = edit.func(path="f.txt", old_string="x", new_string="y", replace_all=True)
    assert not result.is_error
    assert target.read_text() == "y y y"


def test_edit_file_missing_string(jail, tmp_path):
    target = tmp_path / "f.txt"
    target.write_text("hello")
    edit = _edit_tool(jail)
    result = edit.func(path="f.txt", old_string="nope", new_string="x")
    assert result.is_error
    assert "not found" in result.content


def test_read_file_line_range(jail, tmp_path):
    (tmp_path / "f.txt").write_text("l1\nl2\nl3\nl4\n")
    read = {t.name: t for t in build_file_tools(jail, 10_000)}["read_file"]
    result = read.func(path="f.txt", start_line=2, end_line=3)
    assert "l2" in result.content
    assert "l3" in result.content
    assert "l4" not in result.content


def test_schema_has_strict_shape(registry):
    schema = registry.get("edit_file").schema()
    params = schema["function"]["parameters"]
    assert params["additionalProperties"] is False
    assert set(params["required"]) == {"path", "old_string", "new_string"}
    assert "replace_all" in params["properties"]
    assert "replace_all" not in params["required"]


def test_dispatch_runs_tool_and_keys_by_id(registry, tmp_path):
    (tmp_path / "f.txt").write_text("content")
    call = {
        "id": "call_123",
        "function": {"name": "read_file", "arguments": json.dumps({"path": "f.txt"})},
    }
    msg = registry.dispatch(call)
    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "call_123"
    assert "content" in msg["content"]


def test_dispatch_unknown_tool_returns_error(registry):
    call = {"id": "c1", "function": {"name": "nope", "arguments": "{}"}}
    msg = registry.dispatch(call)
    assert msg["content"].startswith("ERROR: ")
    assert "unknown tool" in msg["content"]


def test_dispatch_bad_json_returns_error(registry):
    call = {"id": "c1", "function": {"name": "read_file", "arguments": "{not json"}}
    msg = registry.dispatch(call)
    assert msg["content"].startswith("ERROR: ")


def test_dispatch_missing_required_arg(registry):
    call = {"id": "c1", "function": {"name": "read_file", "arguments": "{}"}}
    msg = registry.dispatch(call)
    assert msg["content"].startswith("ERROR: ")
    assert "missing required" in msg["content"]


def test_dispatch_unexpected_arg(registry):
    call = {
        "id": "c1",
        "function": {
            "name": "read_file",
            "arguments": json.dumps({"path": "f.txt", "bogus": 1}),
        },
    }
    msg = registry.dispatch(call)
    assert msg["content"].startswith("ERROR: ")
    assert "unexpected" in msg["content"]


def test_dispatch_tool_exception_becomes_error():
    reg = ToolRegistry()

    def boom():
        raise RuntimeError("kaboom")

    reg.register(Tool(name="boom", description="", params=[], func=boom))
    msg = reg.dispatch({"id": "c1", "function": {"name": "boom", "arguments": "{}"}})
    assert msg["content"].startswith("ERROR: ")
    assert "kaboom" in msg["content"]


def test_registry_rejects_duplicate():
    reg = ToolRegistry()
    t = Tool(name="dup", description="", params=[], func=lambda: ToolResult("ok"))
    reg.register(t)
    with pytest.raises(ValueError):
        reg.register(t)


def test_path_escape_in_tool_returns_error(jail):
    read = {t.name: t for t in build_file_tools(jail, 10_000)}["read_file"]
    result = read.func(path="../../etc/passwd")
    assert result.is_error
