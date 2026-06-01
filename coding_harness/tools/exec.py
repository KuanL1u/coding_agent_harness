"""Execution tools: shell, Python, and tests.

All execution goes through :func:`~coding_harness.sandbox.run_subprocess`, which
enforces the workspace cwd, a hard timeout with process-group kill, output
truncation, the deny-list, and dry-run mode.
"""

from __future__ import annotations

import sys
import tempfile

from ..sandbox import (
    DeniedCommandError,
    PathEscapeError,
    SubprocessResult,
    WorkspaceJail,
    run_subprocess,
)
from .base import Tool, ToolParam, ToolResult


def _format(result: SubprocessResult) -> ToolResult:
    """Render a :class:`SubprocessResult` as a tool observation."""
    parts = [f"exit_code: {result.exit_code}"]
    if result.timed_out:
        parts.append("(timed out)")
    if result.stdout:
        parts.append(f"--- stdout ---\n{result.stdout}")
    if result.stderr:
        parts.append(f"--- stderr ---\n{result.stderr}")
    content = "\n".join(parts)
    # A non-zero exit or timeout is surfaced as an error observation so the model
    # treats it as a failure to recover from.
    return ToolResult(content=content, is_error=result.exit_code != 0 or result.timed_out)


def build_exec_tools(
    jail: WorkspaceJail,
    *,
    command_timeout_s: float,
    max_output_bytes: int,
    deny_patterns: list[str] | None,
    dry_run: bool,
) -> list[Tool]:
    """Construct the execution tools bound to ``jail`` and sandbox settings."""

    def _run(command: str | list[str], timeout: float | None = None) -> ToolResult:
        try:
            result = run_subprocess(
                command,
                cwd=jail.root,
                timeout=timeout or command_timeout_s,
                max_output_bytes=max_output_bytes,
                deny_patterns=deny_patterns,
                dry_run=dry_run,
            )
        except DeniedCommandError as exc:
            return ToolResult.error(str(exc))
        return _format(result)

    def run_shell(command: str, timeout: int | None = None) -> ToolResult:
        return _run(command, timeout)

    def run_python(
        code: str | None = None, path: str | None = None, timeout: int | None = None
    ) -> ToolResult:
        if code is None and path is None:
            return ToolResult.error("provide either 'code' or 'path'")
        if code is not None and path is not None:
            return ToolResult.error("provide only one of 'code' or 'path'")

        if path is not None:
            try:
                resolved = jail.resolve(path)
            except PathEscapeError as exc:
                return ToolResult.error(str(exc))
            return _run([sys.executable, str(resolved)], timeout)

        # Write the snippet to a temp file inside the workspace so it shares the
        # jailed cwd and is cleaned up by the OS temp dir lifecycle.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", dir=jail.root, delete=False, encoding="utf-8"
        ) as fh:
            fh.write(code or "")
            tmp_path = fh.name
        return _run([sys.executable, tmp_path], timeout)

    def run_tests(
        path: str | None = None, pattern: str | None = None, timeout: int | None = None
    ) -> ToolResult:
        cmd = [sys.executable, "-m", "pytest", "-q"]
        if path is not None:
            try:
                jail.resolve(path)  # confinement check
            except PathEscapeError as exc:
                return ToolResult.error(str(exc))
            cmd.append(path)
        if pattern is not None:
            cmd.extend(["-k", pattern])
        # pytest can run long; give it a generous default relative to the shell timeout.
        return _run(cmd, timeout or max(command_timeout_s, 120))

    return [
        Tool(
            name="run_shell",
            description=(
                "Run a shell command in the workspace root. Subject to a timeout, "
                "output truncation, and the command deny-list."
            ),
            params=[
                ToolParam("command", "string", "Shell command to execute."),
                ToolParam(
                    "timeout", "integer", "Optional timeout in seconds.",
                    required=False,
                ),
            ],
            func=run_shell,
        ),
        Tool(
            name="run_python",
            description=(
                "Run Python code: either an inline 'code' snippet or an existing "
                "'path' in the workspace. Provide exactly one."
            ),
            params=[
                ToolParam("code", "string", "Inline Python source to execute.", required=False),
                ToolParam("path", "string", "Path to a .py file in the workspace.", required=False),
                ToolParam("timeout", "integer", "Optional timeout in seconds.", required=False),
            ],
            func=run_python,
        ),
        Tool(
            name="run_tests",
            description=(
                "Run pytest in the workspace. Optionally scope to a path and/or a "
                "-k pattern."
            ),
            params=[
                ToolParam("path", "string", "Optional test path to run.", required=False),
                ToolParam("pattern", "string", "Optional -k expression to select tests.", required=False),
                ToolParam("timeout", "integer", "Optional timeout in seconds.", required=False),
            ],
            func=run_tests,
        ),
    ]
