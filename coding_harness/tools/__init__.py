"""Tool definitions and the registry that exposes them to the model.

Tools are no longer registered at import time: they are *discovered* at runtime
by :class:`PluginLoader` from a plugins directory (see :mod:`.loader`). The
built-in tool factories below remain the implementations — each built-in plugin
under ``plugins/`` is a thin adapter that calls one of them.
"""

from .base import Tool, ToolParam, ToolRegistry, ToolResult
from .context import ToolConfigView, ToolContext, make_tool_context
from .control import TaskState, build_control_tools
from .exec import build_exec_tools
from .files import build_file_tools
from .loader import Manifest, PluginLoader, default_plugins_dir, validate_tool
from .search import build_search_tools

__all__ = [
    "Tool",
    "ToolParam",
    "ToolRegistry",
    "ToolResult",
    "ToolContext",
    "ToolConfigView",
    "make_tool_context",
    "PluginLoader",
    "Manifest",
    "default_plugins_dir",
    "validate_tool",
    "TaskState",
    "build_control_tools",
    "build_exec_tools",
    "build_file_tools",
    "build_search_tools",
]
