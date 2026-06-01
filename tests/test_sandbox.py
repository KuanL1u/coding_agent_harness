"""Tests for path confinement and safe subprocess execution."""

from __future__ import annotations

import os

import pytest

from coding_harness.sandbox import (
    DeniedCommandError,
    PathEscapeError,
    WorkspaceJail,
    check_denied,
    run_subprocess,
)


def test_jail_resolves_relative_path(tmp_path):
    jail = WorkspaceJail(tmp_path)
    resolved = jail.resolve("sub/file.txt")
    assert str(resolved).startswith(str(tmp_path))
    assert resolved.name == "file.txt"


def test_jail_rejects_parent_traversal(tmp_path):
    jail = WorkspaceJail(tmp_path)
    with pytest.raises(PathEscapeError):
        jail.resolve("../../etc/passwd")


def test_jail_rejects_absolute_outside(tmp_path):
    jail = WorkspaceJail(tmp_path)
    with pytest.raises(PathEscapeError):
        jail.resolve("/etc/passwd")


def test_jail_allows_absolute_inside(tmp_path):
    jail = WorkspaceJail(tmp_path)
    inside = tmp_path / "a" / "b.txt"
    assert jail.resolve(str(inside)).name == "b.txt"


def test_jail_rejects_symlink_escape(tmp_path):
    jail = WorkspaceJail(tmp_path)
    outside_dir = tmp_path.parent / "outside_target"
    outside_dir.mkdir(exist_ok=True)
    link = tmp_path / "escape_link"
    link.symlink_to(outside_dir)
    # Resolving through the symlink lands outside the root and must be rejected.
    with pytest.raises(PathEscapeError):
        jail.resolve("escape_link/secret.txt")


def test_run_subprocess_captures_stdout(tmp_path):
    result = run_subprocess(
        "echo hello", cwd=tmp_path, timeout=10, max_output_bytes=1000
    )
    assert result.exit_code == 0
    assert "hello" in result.stdout
    assert not result.timed_out


def test_run_subprocess_timeout_kills(tmp_path):
    result = run_subprocess(
        "sleep 30", cwd=tmp_path, timeout=1, max_output_bytes=1000
    )
    assert result.timed_out
    assert result.exit_code != 0


def test_run_subprocess_truncates_output(tmp_path):
    # Produce far more bytes than the budget allows.
    result = run_subprocess(
        "python3 -c \"print('x' * 100000)\"",
        cwd=tmp_path,
        timeout=10,
        max_output_bytes=500,
    )
    assert result.truncated
    assert "truncated" in result.stdout
    assert len(result.stdout.encode("utf-8")) < 1000


def test_run_subprocess_dry_run_does_not_execute(tmp_path):
    marker = tmp_path / "created"
    result = run_subprocess(
        f"touch {marker}", cwd=tmp_path, timeout=10, max_output_bytes=1000, dry_run=True
    )
    assert "dry_run" in result.stdout
    assert not marker.exists()


def test_deny_pattern_blocks_command():
    with pytest.raises(DeniedCommandError):
        check_denied("rm -rf /", [r"\brm\s+-rf\s+/"])


def test_run_subprocess_respects_deny(tmp_path):
    with pytest.raises(DeniedCommandError):
        run_subprocess(
            "rm -rf /",
            cwd=tmp_path,
            timeout=10,
            max_output_bytes=1000,
            deny_patterns=[r"\brm\s+-rf\s+/"],
        )
