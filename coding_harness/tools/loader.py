"""Runtime, directory-based tool discovery — the pluggable registry.

Instead of hardcoding which tools exist, the harness *discovers* them: every
tool (built-in or agent-authored) lives in a self-describing plugin folder under
a ``plugins/`` directory::

    plugins/<name>/
        tool.py        # exposes ``build(ctx) -> Tool | list[Tool]``
        manifest.json  # metadata + lifecycle status
        test_tool.py   # unit tests the staging gate runs

:class:`PluginLoader` scans that directory, loads only plugins whose manifest
``status == "active"``, calls each plugin's ``build(ctx)`` with the sandboxed
:class:`~coding_harness.tools.context.ToolContext`, validates every produced
:class:`~coding_harness.tools.base.Tool`'s schema, and registers the survivors
into a :class:`~coding_harness.tools.base.ToolRegistry`.

Discovery is **failure-isolated**: a malformed manifest, an import error, a
build error, a malformed schema, or a duplicate name rejects only that one
plugin (recorded on :attr:`PluginLoader.rejections` and emitted as a
``plugin_rejected`` event) — it never aborts the load. A clean load emits
nothing, so the trace of a normal run is unchanged.

The registry is pluggable, but the agent loop is untouched — it still just
calls ``registry.schemas()`` / ``registry.dispatch()``. The forbidden-import
scan and the staging/validation gate that gate *agent-authored* tools are not
built yet; the loader here trusts that only human-approved (merged) plugins ever
carry ``status == "active"``.
"""

from __future__ import annotations

import importlib.util
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .base import Tool, ToolRegistry
from .context import ToolContext

OnEvent = Callable[[str, dict[str, Any]], None]

# A tool name must be a plain identifier so it maps cleanly to an OpenAI tool.
_VALID_NAME = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]*$")
# JSON-schema scalar/container types a tool parameter may declare.
_JSON_TYPES = {"string", "integer", "number", "boolean", "object", "array"}


def default_plugins_dir() -> Path:
    """The packaged built-in plugins directory (``coding_harness/tools/plugins``)."""
    return Path(__file__).parent / "plugins"


@dataclass
class Manifest:
    """A plugin's ``manifest.json`` — its audit record and lifecycle status.

    Only ``name``/``status`` drive loading; the rest is provenance the
    self-extension loop fills in (why the tool exists, which PR approved it, its
    measured benchmark impact). Parsing is tolerant: unknown keys are kept in
    :attr:`raw` rather than rejected, so the manifest can grow without breaking
    older loaders.
    """

    name: str
    status: str = "active"  # "staged" | "active" | "disabled"
    author: str = "agent"  # "builtin" | "agent"
    version: int = 1
    # Lower loads first; ties broken by folder name. Built-ins pin an explicit
    # priority so the tool list keeps its v1 order (files, search, exec, control).
    load_priority: int = 100
    provides: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: Path) -> "Manifest":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("manifest is not a JSON object")
        name = data.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("manifest 'name' must be a non-empty string")
        return cls(
            name=name,
            status=str(data.get("status", "active")),
            author=str(data.get("author", "agent")),
            version=int(data.get("version", 1)),
            load_priority=int(data.get("load_priority", 100)),
            provides=list(data.get("provides", []) or []),
            raw=data,
        )


def validate_tool(tool: Any) -> str | None:
    """Return an error string if ``tool`` is not a well-formed Tool, else None.

    Enforces the contract a discovered tool must satisfy: it is a
    :class:`Tool`, its name is a plain identifier, it has a non-empty
    description, every parameter declares a valid JSON-schema type, and the
    generated parameter schema is a strict object schema
    (``additionalProperties: false`` with a ``required`` list).
    """
    if not isinstance(tool, Tool):
        return f"build() produced a non-Tool object: {type(tool).__name__}"
    if not isinstance(tool.name, str) or not _VALID_NAME.match(tool.name):
        return f"invalid tool name: {tool.name!r}"
    if not isinstance(tool.description, str) or not tool.description.strip():
        return f"tool {tool.name!r} has an empty description"
    if not callable(tool.func):
        return f"tool {tool.name!r} has no callable func"
    for p in tool.params:
        if p.type not in _JSON_TYPES:
            return f"tool {tool.name!r} parameter {p.name!r} has invalid type {p.type!r}"

    schema = tool.schema().get("function", {}).get("parameters", {})
    if schema.get("type") != "object":
        return f"tool {tool.name!r} schema is not an object schema"
    if schema.get("additionalProperties") is not False:
        return f"tool {tool.name!r} schema must set additionalProperties: false"
    if not isinstance(schema.get("required"), list):
        return f"tool {tool.name!r} schema is missing a 'required' list"
    return None


def _load_module(tool_py: Path, mod_name: str) -> Any:
    """Import ``tool.py`` from a file path under a unique synthetic module name."""
    spec = importlib.util.spec_from_file_location(mod_name, tool_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create import spec for {tool_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PluginLoader:
    """Discovers and registers active tool plugins from a directory."""

    def __init__(
        self,
        plugins_dir: str | Path,
        ctx: ToolContext,
        *,
        on_event: OnEvent | None = None,
    ) -> None:
        self.plugins_dir = Path(plugins_dir)
        self.ctx = ctx
        self.on_event = on_event
        self.rejections: list[dict[str, str]] = []
        self.skipped: list[dict[str, str]] = []
        self.loaded: list[str] = []

    def _emit(self, event: str, **fields: Any) -> None:
        if self.on_event is not None:
            self.on_event(event, fields)

    def _reject(self, plugin: str, reason: str) -> None:
        self.rejections.append({"plugin": plugin, "reason": reason})
        self._emit("plugin_rejected", plugin=plugin, reason=reason)

    def _skip(self, plugin: str, status: str) -> None:
        self.skipped.append({"plugin": plugin, "status": status})
        self._emit("plugin_skipped", plugin=plugin, status=status)

    def load(self) -> ToolRegistry:
        """Scan the directory and return a registry of every active, valid tool."""
        registry = ToolRegistry()
        if not self.plugins_dir.is_dir():
            self._emit("plugins_dir_missing", path=str(self.plugins_dir))
            return registry

        entries = self._discover()
        # Deterministic load order: by manifest priority, then folder name. This
        # keeps the built-in tool list in its v1 order so runs are unchanged.
        entries.sort(key=lambda e: (e[0].load_priority, e[0].name))

        for manifest, folder in entries:
            self._load_one(manifest, folder, registry)
        return registry

    def _discover(self) -> list[tuple[Manifest, Path]]:
        """Read manifests for every candidate folder, filtering to active ones."""
        out: list[tuple[Manifest, Path]] = []
        for folder in sorted(self.plugins_dir.iterdir()):
            if not folder.is_dir() or folder.name.startswith((".", "_")):
                continue
            manifest_path = folder / "manifest.json"
            tool_py = folder / "tool.py"
            if not manifest_path.is_file():
                self._reject(folder.name, "missing manifest.json")
                continue
            if not tool_py.is_file():
                self._reject(folder.name, "missing tool.py")
                continue
            try:
                manifest = Manifest.from_file(manifest_path)
            except Exception as exc:  # noqa: BLE001 - a bad manifest skips one plugin
                self._reject(folder.name, f"invalid manifest.json: {exc}")
                continue
            if manifest.status != "active":
                self._skip(folder.name, manifest.status)
                continue
            out.append((manifest, folder))
        return out

    def _load_one(
        self, manifest: Manifest, folder: Path, registry: ToolRegistry
    ) -> None:
        tool_py = folder / "tool.py"
        try:
            module = _load_module(tool_py, f"ch_plugin_{folder.name}")
        except Exception as exc:  # noqa: BLE001 - import error rejects one plugin
            self._reject(manifest.name, f"import error: {type(exc).__name__}: {exc}")
            return

        build_fn = getattr(module, "build", None)
        if not callable(build_fn):
            self._reject(manifest.name, "tool.py has no callable build(ctx)")
            return

        try:
            produced = build_fn(self.ctx)
        except Exception as exc:  # noqa: BLE001 - build error rejects one plugin
            self._reject(manifest.name, f"build() failed: {type(exc).__name__}: {exc}")
            return

        tools = [produced] if isinstance(produced, Tool) else list(produced or [])
        if not tools:
            self._reject(manifest.name, "build() produced no tools")
            return

        for tool in tools:
            err = validate_tool(tool)
            if err is not None:
                self._reject(manifest.name, err)
                continue
            if registry.get(tool.name) is not None:
                self._reject(manifest.name, f"duplicate tool name: {tool.name!r}")
                continue
            registry.register(tool)
            self.loaded.append(tool.name)
