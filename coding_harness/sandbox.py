"""Workspace confinement and safe subprocess execution.

This module is the security boundary of the harness. Two responsibilities:

* :class:`WorkspaceJail` resolves and confines every filesystem path the agent
  touches to a single ``workspace_root``, rejecting ``..`` traversal, absolute
  paths outside the root, and symlinks that escape the root.
* :func:`run_subprocess` runs a command with a hard wall-clock timeout, kills the
  whole process group on timeout, captures stdout/stderr, and truncates output to
  a byte budget so a runaway command cannot exhaust memory or the context window.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path


class PathEscapeError(ValueError):
    """Raised when a requested path resolves outside the workspace jail."""


class DeniedCommandError(ValueError):
    """Raised when a command matches a configured deny pattern."""


class WorkspaceJail:
    """Confines all path access to ``root``.

    All public methods return a resolved absolute :class:`~pathlib.Path` that is
    guaranteed to live inside ``root``. Any attempt to escape raises
    :class:`PathEscapeError`.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        # ``resolve()`` collapses symlinks and ``..`` so the stored root is the
        # canonical real path. The directory is created if missing.
        self.root: Path = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, path: str | os.PathLike[str]) -> Path:
        """Resolve ``path`` (relative to root) and confine it to the jail.

        Relative paths are interpreted against the workspace root. Absolute
        paths are allowed only if they already live inside the root. Symlinks
        are followed during resolution, so a symlink pointing outside the root
        is rejected just like any other escape.
        """
        candidate = Path(path)
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = (self.root / candidate).resolve()

        if not self._is_within_root(resolved):
            raise PathEscapeError(
                f"path {path!r} resolves to {resolved} which is outside "
                f"the workspace root {self.root}"
            )
        return resolved

    def _is_within_root(self, resolved: Path) -> bool:
        # ``is_relative_to`` (3.9+) is the canonical containment check. The root
        # itself counts as within the jail.
        return resolved == self.root or resolved.is_relative_to(self.root)

    def relative(self, path: str | os.PathLike[str]) -> str:
        """Return ``path`` as a string relative to the workspace root."""
        resolved = self.resolve(path)
        return str(resolved.relative_to(self.root))


@dataclass
class SubprocessResult:
    """Outcome of a :func:`run_subprocess` call."""

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    truncated: bool


def truncate_text(text: str, max_bytes: int) -> tuple[str, bool]:
    """Truncate ``text`` so its UTF-8 encoding is at most ``max_bytes``.

    Returns the (possibly truncated) text and a flag indicating truncation.
    """
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text, False
    # Decode the truncated byte slice, dropping any partial trailing char.
    clipped = encoded[:max_bytes].decode("utf-8", errors="ignore")
    notice = f"\n...[truncated to {max_bytes} bytes]..."
    return clipped + notice, True


def check_denied(command: str, deny_patterns: list[str] | None) -> None:
    """Raise :class:`DeniedCommandError` if ``command`` matches a deny pattern.

    Patterns are treated as regular expressions and matched (searched)
    case-sensitively against the full command string.
    """
    for pattern in deny_patterns or []:
        if re.search(pattern, command):
            raise DeniedCommandError(
                f"command blocked by deny pattern {pattern!r}: {command!r}"
            )


def run_subprocess(
    command: str | list[str],
    cwd: str | os.PathLike[str],
    timeout: float,
    max_output_bytes: int,
    *,
    deny_patterns: list[str] | None = None,
    dry_run: bool = False,
    env: dict[str, str] | None = None,
) -> SubprocessResult:
    """Run ``command`` under a hard timeout with output capture and truncation.

    The command runs in its own process group so that, on timeout, the whole
    group (including child processes) can be killed rather than leaking orphans.

    Args:
        command: A shell string (run via ``/bin/sh -c``) or an argv list.
        cwd: Working directory for the command.
        timeout: Wall-clock timeout in seconds.
        max_output_bytes: Per-stream truncation budget for stdout and stderr.
        deny_patterns: Optional regexes; a matching command is refused.
        dry_run: If true, do not execute — echo what would have run.
        env: Optional environment overrides merged onto the current environment.
    """
    command_str = command if isinstance(command, str) else " ".join(command)
    check_denied(command_str, deny_patterns)

    if dry_run:
        return SubprocessResult(
            stdout=f"[dry_run] would run: {command_str}",
            stderr="",
            exit_code=0,
            timed_out=False,
            truncated=False,
        )

    use_shell = isinstance(command, str)
    full_env = {**os.environ, **(env or {})}

    # start_new_session=True puts the child in its own process group, letting us
    # signal the entire group (negative pid) on timeout.
    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        shell=use_shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        env=full_env,
    )

    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_process_group(proc)
        # Drain whatever output was produced before the kill.
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""

    stdout = stdout or ""
    stderr = stderr or ""
    stdout, t1 = truncate_text(stdout, max_output_bytes)
    stderr, t2 = truncate_text(stderr, max_output_bytes)

    if timed_out:
        stderr = (stderr + f"\n[timed out after {timeout}s]").strip()

    return SubprocessResult(
        stdout=stdout,
        stderr=stderr,
        exit_code=proc.returncode if proc.returncode is not None else -1,
        timed_out=timed_out,
        truncated=t1 or t2,
    )


def _kill_process_group(proc: subprocess.Popen) -> None:
    """Terminate the process group led by ``proc``, escalating to SIGKILL."""
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=2)
            return
        except subprocess.TimeoutExpired:
            continue
