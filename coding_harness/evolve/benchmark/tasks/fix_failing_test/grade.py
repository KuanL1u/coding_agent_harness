"""Grader: the task passes iff pytest passes in the workspace.

Runs as ``python grade.py <workspace>`` and prints a single JSON object. Uses
only the standard library plus pytest (invoked as a subprocess) so it is fully
self-contained and runs outside the agent's jail.
"""

import json
import subprocess
import sys
from pathlib import Path


def grade(workspace: Path) -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=str(workspace),
        capture_output=True,
        text=True,
        timeout=120,
    )
    passed = proc.returncode == 0
    tail = (proc.stdout or proc.stderr or "").strip().splitlines()[-1:] or [""]
    return {
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "details": tail[0],
    }


if __name__ == "__main__":
    ws = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    print(json.dumps(grade(ws)))
