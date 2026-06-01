"""Tests for the pluggable tool registry (PT-M1).

Covers: the built-in plugins are discovered (and keep their v1 order); the
loader filters by manifest status; and every rejection path (bad manifest,
import error, build error, malformed schema, duplicate name, missing files)
rejects only the offending plugin and is reported via the event hook, never
aborting the load.
"""

from __future__ import annotations

import json

import pytest

from coding_harness.agent import build_registry
from coding_harness.config import Config
from coding_harness.sandbox import WorkspaceJail
from coding_harness.tools import (
    PluginLoader,
    TaskState,
    Tool,
    ToolParam,
    make_tool_context,
)

# The exact v1 tool set, in registration order — discovery must preserve it.
V1_TOOLS = [
    "read_file",
    "write_file",
    "edit_file",
    "list_dir",
    "grep_search",
    "glob_search",
    "run_shell",
    "run_python",
    "run_tests",
    "task_done",
]


# -- helpers ------------------------------------------------------------------


def _write_plugin(root, name, *, tool_py, manifest=None, status="active", priority=100):
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "tool.py").write_text(tool_py, encoding="utf-8")
    if manifest is None:
        manifest = {"name": name, "status": status, "load_priority": priority}
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return folder


def _ctx(tmp_path):
    return make_tool_context(Config(), WorkspaceJail(tmp_path / "ws"), TaskState())


def _loader(tmp_path, plugins_dir):
    events: list[tuple[str, dict]] = []
    loader = PluginLoader(
        plugins_dir, _ctx(tmp_path), on_event=lambda e, f: events.append((e, f))
    )
    return loader, events


_OK_PLUGIN = """
from coding_harness.tools.base import Tool, ToolParam, ToolResult

def build(ctx):
    return Tool(
        name="{name}",
        description="a test tool",
        params=[ToolParam("x", "string", "an arg")],
        func=lambda x: ToolResult("ok"),
    )
"""


# -- built-in discovery -------------------------------------------------------


def test_builtins_discovered_in_v1_order(tmp_path):
    """build_registry must yield exactly the v1 tools, in the v1 order."""
    cfg = Config()
    registry = build_registry(cfg, WorkspaceJail(tmp_path / "ws"), TaskState())
    assert registry.names() == V1_TOOLS


def test_builtin_schemas_are_strict(tmp_path):
    registry = build_registry(Config(), WorkspaceJail(tmp_path / "ws"), TaskState())
    for schema in registry.schemas():
        params = schema["function"]["parameters"]
        assert params["type"] == "object"
        assert params["additionalProperties"] is False
        assert isinstance(params["required"], list)


# -- status filtering ---------------------------------------------------------


def test_staged_and_disabled_plugins_are_skipped(tmp_path):
    plugins = tmp_path / "plugins"
    _write_plugin(plugins, "live", tool_py=_OK_PLUGIN.format(name="live_tool"))
    _write_plugin(plugins, "pending", tool_py=_OK_PLUGIN.format(name="staged_tool"), status="staged")
    _write_plugin(plugins, "off", tool_py=_OK_PLUGIN.format(name="disabled_tool"), status="disabled")

    loader, events = _loader(tmp_path, plugins)
    registry = loader.load()

    assert registry.names() == ["live_tool"]
    skipped = {f["plugin"]: f["status"] for e, f in events if e == "plugin_skipped"}
    assert skipped == {"pending": "staged", "off": "disabled"}


def test_underscore_prefixed_folders_are_ignored(tmp_path):
    """A ``_staging`` quarantine dir under plugins/ is never discovered."""
    plugins = tmp_path / "plugins"
    _write_plugin(plugins, "live", tool_py=_OK_PLUGIN.format(name="live_tool"))
    _write_plugin(plugins, "_staging", tool_py=_OK_PLUGIN.format(name="quarantined"))

    loader, _ = _loader(tmp_path, plugins)
    assert loader.load().names() == ["live_tool"]


# -- load order ---------------------------------------------------------------


def test_load_priority_orders_tools(tmp_path):
    plugins = tmp_path / "plugins"
    _write_plugin(plugins, "b", tool_py=_OK_PLUGIN.format(name="second"), priority=20)
    _write_plugin(plugins, "a", tool_py=_OK_PLUGIN.format(name="first"), priority=10)
    loader, _ = _loader(tmp_path, plugins)
    assert loader.load().names() == ["first", "second"]


# -- rejection paths (each isolates one plugin, never fatal) ------------------


def test_import_error_rejects_only_that_plugin(tmp_path):
    plugins = tmp_path / "plugins"
    _write_plugin(plugins, "good", tool_py=_OK_PLUGIN.format(name="good_tool"))
    _write_plugin(plugins, "bad", tool_py="import a_module_that_does_not_exist_xyz\n")

    loader, events = _loader(tmp_path, plugins)
    registry = loader.load()

    assert registry.names() == ["good_tool"]
    rejects = {f["plugin"] for e, f in events if e == "plugin_rejected"}
    assert "bad" in rejects
    assert any("import error" in f["reason"] for e, f in events if e == "plugin_rejected")


def test_build_error_is_isolated(tmp_path):
    plugins = tmp_path / "plugins"
    _write_plugin(plugins, "good", tool_py=_OK_PLUGIN.format(name="good_tool"))
    _write_plugin(plugins, "boom", tool_py="def build(ctx):\n    raise RuntimeError('nope')\n")

    loader, _ = _loader(tmp_path, plugins)
    registry = loader.load()
    assert registry.names() == ["good_tool"]
    assert any("build() failed" in r["reason"] for r in loader.rejections)


def test_non_tool_build_result_rejected(tmp_path):
    plugins = tmp_path / "plugins"
    _write_plugin(plugins, "good", tool_py=_OK_PLUGIN.format(name="good_tool"))
    _write_plugin(plugins, "weird", tool_py="def build(ctx):\n    return 'not a tool'\n")

    loader, _ = _loader(tmp_path, plugins)
    registry = loader.load()
    assert registry.names() == ["good_tool"]


def test_duplicate_tool_name_rejected(tmp_path):
    plugins = tmp_path / "plugins"
    _write_plugin(plugins, "a", tool_py=_OK_PLUGIN.format(name="dup"), priority=10)
    _write_plugin(plugins, "b", tool_py=_OK_PLUGIN.format(name="dup"), priority=20)

    loader, _ = _loader(tmp_path, plugins)
    registry = loader.load()
    # First (lower priority) wins; the second is rejected as a duplicate.
    assert registry.names() == ["dup"]
    assert any("duplicate tool name" in r["reason"] for r in loader.rejections)


def test_missing_manifest_or_tool_py_rejected(tmp_path):
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    # Folder with tool.py but no manifest.json.
    (plugins / "no_manifest").mkdir()
    (plugins / "no_manifest" / "tool.py").write_text(_OK_PLUGIN.format(name="x"))
    # Folder with manifest.json but no tool.py.
    (plugins / "no_tool").mkdir()
    (plugins / "no_tool" / "manifest.json").write_text(json.dumps({"name": "no_tool"}))

    loader, _ = _loader(tmp_path, plugins)
    registry = loader.load()
    assert registry.names() == []
    reasons = {r["plugin"]: r["reason"] for r in loader.rejections}
    assert "missing manifest.json" in reasons["no_manifest"]
    assert "missing tool.py" in reasons["no_tool"]


def test_malformed_manifest_rejected(tmp_path):
    plugins = tmp_path / "plugins"
    folder = plugins / "broken"
    folder.mkdir(parents=True)
    (folder / "tool.py").write_text(_OK_PLUGIN.format(name="x"))
    (folder / "manifest.json").write_text("{not valid json")

    loader, _ = _loader(tmp_path, plugins)
    assert loader.load().names() == []
    assert any("invalid manifest" in r["reason"] for r in loader.rejections)


def test_missing_plugins_dir_is_not_fatal(tmp_path):
    loader, events = _loader(tmp_path, tmp_path / "does_not_exist")
    registry = loader.load()
    assert registry.names() == []
    assert any(e == "plugins_dir_missing" for e, _ in events)


# -- validate_tool ------------------------------------------------------------


def test_validate_tool_rejects_bad_name():
    from coding_harness.tools.loader import validate_tool

    bad = Tool(name="9bad", description="x", params=[], func=lambda: None)
    assert validate_tool(bad) is not None


def test_validate_tool_rejects_bad_param_type():
    from coding_harness.tools.loader import validate_tool

    bad = Tool(
        name="t",
        description="x",
        params=[ToolParam("p", "not_a_json_type", "d")],
        func=lambda p: None,
    )
    assert "invalid type" in validate_tool(bad)


def test_validate_tool_accepts_well_formed():
    from coding_harness.tools.loader import validate_tool

    ok = Tool(
        name="t",
        description="x",
        params=[ToolParam("p", "string", "d")],
        func=lambda p: None,
    )
    assert validate_tool(ok) is None
