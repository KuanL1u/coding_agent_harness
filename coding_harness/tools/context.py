"""The capability context handed to every tool plugin's ``build()``.

A :class:`ToolContext` is the *only* thing a plugin receives. It carries the
sandboxed primitives a tool needs — the :class:`~coding_harness.sandbox.WorkspaceJail`,
a pre-wrapped ``run_subprocess`` runner, and a read-only view of the relevant
config — so an agent-authored tool inherits all of v1's safety (path
confinement, the command deny-list, timeouts, output truncation, dry-run)
automatically and has **no privileged path** to bypass it.

``state`` is the one harness-internal handle on the context: the loop's mutable
:class:`~coding_harness.tools.control.TaskState`, used only by the built-in
``control`` plugin (``task_done``). It is deliberately not part of the
capability surface agent-authored tools are meant to use; the staging gate
(PT-M2) forbids agent tools from reaching into loop internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..sandbox import SubprocessResult, WorkspaceJail, run_subprocess


@dataclass(frozen=True)
class ToolConfigView:
    """A read-only snapshot of the config a tool may legitimately observe.

    Only the sandbox-relevant knobs are exposed; nothing here lets a tool change
    safety behaviour — the wrapped :attr:`ToolContext.run_subprocess` already has
    the timeout, byte budget, deny-list, and dry-run baked in.
    """

    max_output_bytes: int
    command_timeout_s: float
    dry_run: bool
    workspace_root: str
    deny_patterns: tuple[str, ...] = ()


# A tool's view of the sandboxed runner: ``runner(command, timeout=None)``.
RunSubprocess = Callable[..., SubprocessResult]


@dataclass
class ToolContext:
    """Everything a plugin's ``build(ctx)`` is allowed to close over."""

    jail: WorkspaceJail
    run_subprocess: RunSubprocess
    config: ToolConfigView
    # Harness-internal loop state (TaskState). Present so the built-in control
    # plugin can be discovered through the same uniform path; agent-authored
    # tools should not depend on it.
    state: Any = field(default=None)


def make_tool_context(
    config: Any, jail: WorkspaceJail, state: Any = None
) -> ToolContext:
    """Build a :class:`ToolContext` from the harness config + a jail.

    The returned ``run_subprocess`` is the same deny-listed, timed, truncated,
    dry-run-aware runner the built-in execution tools use — bound to this jail's
    root as cwd — so any tool that calls it is confined exactly like v1.
    """
    sb = config.sandbox
    view = ToolConfigView(
        max_output_bytes=sb.max_output_bytes,
        command_timeout_s=sb.command_timeout_s,
        dry_run=sb.dry_run,
        workspace_root=str(jail.root),
        deny_patterns=tuple(sb.deny_patterns or ()),
    )

    def runner(command: str | list[str], timeout: float | None = None) -> SubprocessResult:
        return run_subprocess(
            command,
            cwd=jail.root,
            timeout=timeout or sb.command_timeout_s,
            max_output_bytes=sb.max_output_bytes,
            deny_patterns=sb.deny_patterns,
            dry_run=sb.dry_run,
        )

    return ToolContext(jail=jail, run_subprocess=runner, config=view, state=state)
