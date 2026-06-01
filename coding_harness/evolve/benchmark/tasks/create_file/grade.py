"""Grader: passes iff greeting.txt contains exactly 'Hello, world!'.

A single trailing newline is tolerated; any other content fails.
"""

import json
import sys
from pathlib import Path

EXPECTED = "Hello, world!"


def grade(workspace: Path) -> dict:
    target = workspace / "greeting.txt"
    if not target.is_file():
        return {"passed": False, "score": 0.0, "details": "greeting.txt not found"}
    content = target.read_text(encoding="utf-8", errors="replace")
    passed = content.rstrip("\n") == EXPECTED
    return {
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "details": "exact match" if passed else f"unexpected content: {content!r}",
    }


if __name__ == "__main__":
    ws = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    print(json.dumps(grade(ws)))
