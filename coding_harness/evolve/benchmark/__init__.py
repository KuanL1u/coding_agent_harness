"""The shared Evaluation Gate: held-out tasks + a runner that scores the harness.

The benchmark is a fixed set of small, deterministic coding tasks. Each lives in
``tasks/<name>/`` and contains:

* ``task.md`` — the instruction handed to the agent.
* ``setup/`` — the initial workspace state, copied into a fresh isolated jail per
  run (so a run never mutates the canonical task).
* ``grade.py`` — a self-contained, stdlib-only grader run *outside* the agent's
  jail. Invoked as ``python grade.py <workspace>`` it prints one JSON object:
  ``{"passed": bool, "score": float, "details": str}``.

Hard rule (anti-reward-hacking): the tasks, graders, and runner live outside the
agent's writable workspace. The agent can never edit what scores it.
"""
