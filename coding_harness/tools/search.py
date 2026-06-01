"""Search tools: content grep and filename glob.

Both are implemented with the Python stdlib (``re``, ``pathlib``) so the harness
has no external search dependency. Results are confined to the workspace and
truncated to a line/byte budget.
"""

from __future__ import annotations

import re

from ..sandbox import WorkspaceJail, truncate_text
from .base import Tool, ToolParam, ToolResult

# Hard cap on result lines so a broad pattern cannot flood the context window.
_MAX_MATCH_LINES = 200


def build_search_tools(jail: WorkspaceJail, max_output_bytes: int) -> list[Tool]:
    """Construct the search tools bound to ``jail``."""

    def _iter_files(glob: str | None):
        pattern = glob or "**/*"
        for p in jail.root.glob(pattern):
            if p.is_file():
                yield p

    def grep_search(
        pattern: str, glob: str | None = None, ignore_case: bool = False
    ) -> ToolResult:
        try:
            flags = re.IGNORECASE if ignore_case else 0
            regex = re.compile(pattern, flags)
        except re.error as exc:
            return ToolResult.error(f"invalid regex: {exc}")

        matches: list[str] = []
        for path in _iter_files(glob):
            rel = path.relative_to(jail.root)
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    matches.append(f"{rel}:{lineno}:{line}")
                    if len(matches) >= _MAX_MATCH_LINES:
                        matches.append(f"...[stopped after {_MAX_MATCH_LINES} matches]...")
                        break
            if len(matches) >= _MAX_MATCH_LINES:
                break

        if not matches:
            return ToolResult(content="no matches")
        body = truncate_text("\n".join(matches), max_output_bytes)[0]
        return ToolResult(content=body)

    def glob_search(pattern: str) -> ToolResult:
        results = [
            str(p.relative_to(jail.root))
            for p in sorted(jail.root.glob(pattern))
        ]
        if not results:
            return ToolResult(content="no files matched")
        if len(results) > _MAX_MATCH_LINES:
            results = results[:_MAX_MATCH_LINES]
            results.append(f"...[stopped after {_MAX_MATCH_LINES} paths]...")
        return ToolResult(content="\n".join(results))

    return [
        Tool(
            name="grep_search",
            description=(
                "Search file contents for a Python regular expression. Returns "
                "matching lines as path:lineno:line."
            ),
            params=[
                ToolParam("pattern", "string", "Python regular expression to search for."),
                ToolParam(
                    "glob", "string",
                    "Optional glob (e.g. '**/*.py') restricting which files are searched.",
                    required=False,
                ),
                ToolParam(
                    "ignore_case", "boolean", "Case-insensitive matching.",
                    required=False,
                ),
            ],
            func=grep_search,
        ),
        Tool(
            name="glob_search",
            description="Find files whose path matches a glob pattern (e.g. 'src/**/*.py').",
            params=[
                ToolParam("pattern", "string", "Glob pattern relative to the workspace root."),
            ],
            func=glob_search,
        ),
    ]
