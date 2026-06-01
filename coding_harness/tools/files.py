"""File manipulation tools: read, write, edit, list.

Every path argument is resolved through the :class:`~coding_harness.sandbox.WorkspaceJail`
so file access cannot escape the workspace. Output that could be unbounded
(file contents, directory listings) is truncated to ``max_output_bytes``.
"""

from __future__ import annotations

import os

from ..sandbox import PathEscapeError, WorkspaceJail, truncate_text
from .base import Tool, ToolParam, ToolResult


def _clip(text: str, max_bytes: int) -> str:
    return truncate_text(text, max_bytes)[0]


def build_file_tools(jail: WorkspaceJail, max_output_bytes: int) -> list[Tool]:
    """Construct the file tools bound to ``jail``."""

    def read_file(
        path: str, start_line: int | None = None, end_line: int | None = None
    ) -> ToolResult:
        try:
            resolved = jail.resolve(path)
        except PathEscapeError as exc:
            return ToolResult.error(str(exc))
        if not resolved.is_file():
            return ToolResult.error(f"not a file: {path}")

        lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
        total = len(lines)

        # 1-based, inclusive line range. Defaults read the whole file.
        lo = (start_line - 1) if start_line else 0
        hi = end_line if end_line else total
        lo = max(lo, 0)
        hi = min(hi, total)
        if lo >= hi:
            return ToolResult.error(
                f"empty line range [{start_line}, {end_line}] for file with {total} lines"
            )

        # Prefix each line with its absolute line number for easy reference.
        numbered = [f"{i + 1}\t{lines[i]}" for i in range(lo, hi)]
        body = "\n".join(numbered)
        header = f"{jail.relative(resolved)} (lines {lo + 1}-{hi} of {total})\n"
        return ToolResult(content=_clip(header + body, max_output_bytes))

    def write_file(path: str, content: str) -> ToolResult:
        try:
            resolved = jail.resolve(path)
        except PathEscapeError as exc:
            return ToolResult.error(str(exc))
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return ToolResult(
            content=f"wrote {len(content)} bytes to {jail.relative(resolved)}"
        )

    def edit_file(
        path: str, old_string: str, new_string: str, replace_all: bool = False
    ) -> ToolResult:
        try:
            resolved = jail.resolve(path)
        except PathEscapeError as exc:
            return ToolResult.error(str(exc))
        if not resolved.is_file():
            return ToolResult.error(f"not a file: {path}")

        text = resolved.read_text(encoding="utf-8", errors="replace")
        count = text.count(old_string)
        if count == 0:
            return ToolResult.error("old_string not found in file")
        if count > 1 and not replace_all:
            return ToolResult.error(
                f"old_string is not unique ({count} matches); pass replace_all=true "
                "or include more surrounding context to make it unique"
            )

        updated = (
            text.replace(old_string, new_string)
            if replace_all
            else text.replace(old_string, new_string, 1)
        )
        resolved.write_text(updated, encoding="utf-8")
        replaced = count if replace_all else 1
        return ToolResult(
            content=f"replaced {replaced} occurrence(s) in {jail.relative(resolved)}"
        )

    def list_dir(path: str = ".") -> ToolResult:
        try:
            resolved = jail.resolve(path)
        except PathEscapeError as exc:
            return ToolResult.error(str(exc))
        if not resolved.is_dir():
            return ToolResult.error(f"not a directory: {path}")

        entries = []
        for entry in sorted(resolved.iterdir(), key=lambda p: (p.is_file(), p.name)):
            marker = "/" if entry.is_dir() else ""
            entries.append(entry.name + marker)
        listing = "\n".join(entries) if entries else "(empty)"
        header = f"{jail.relative(resolved)}{os.sep}\n"
        return ToolResult(content=_clip(header + listing, max_output_bytes))

    return [
        Tool(
            name="read_file",
            description=(
                "Read a UTF-8 text file from the workspace. Optionally restrict to "
                "an inclusive 1-based line range with start_line/end_line."
            ),
            params=[
                ToolParam("path", "string", "Path to the file, relative to the workspace root."),
                ToolParam(
                    "start_line", "integer", "First line to read (1-based, inclusive).",
                    required=False,
                ),
                ToolParam(
                    "end_line", "integer", "Last line to read (1-based, inclusive).",
                    required=False,
                ),
            ],
            func=read_file,
        ),
        Tool(
            name="write_file",
            description=(
                "Create or overwrite a file with the given content. Parent "
                "directories are created as needed."
            ),
            params=[
                ToolParam("path", "string", "Path to the file, relative to the workspace root."),
                ToolParam("content", "string", "Full file content to write."),
            ],
            func=write_file,
        ),
        Tool(
            name="edit_file",
            description=(
                "Replace an exact string in a file. Fails if old_string is not "
                "found, or if it occurs more than once and replace_all is false."
            ),
            params=[
                ToolParam("path", "string", "Path to the file, relative to the workspace root."),
                ToolParam("old_string", "string", "Exact text to find."),
                ToolParam("new_string", "string", "Replacement text."),
                ToolParam(
                    "replace_all", "boolean",
                    "Replace every occurrence instead of requiring a unique match.",
                    required=False,
                ),
            ],
            func=edit_file,
        ),
        Tool(
            name="list_dir",
            description="List the entries of a directory in the workspace.",
            params=[
                ToolParam(
                    "path", "string",
                    "Directory path relative to the workspace root (defaults to root).",
                    required=False,
                ),
            ],
            func=list_dir,
        ),
    ]
