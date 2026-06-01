"""Pytest configuration for the repository test run.

The benchmark suite under ``coding_harness/evolve/benchmark/tasks/`` contains
``test_*.py`` files that are *fixtures for the agent*, not tests of this repo
(they import workspace modules that intentionally start broken). Exclude them
from collection so the harness's own test run is unaffected.

Tool plugins under ``coding_harness/tools/plugins/`` ship a ``test_tool.py``
that the tool-validation gate runs per-plugin in isolation; it is not
part of the harness's own suite (the built-in tools are already covered by
``tests/``), so it is excluded here too.
"""

collect_ignore_glob = [
    "coding_harness/evolve/benchmark/tasks/*",
    "coding_harness/tools/plugins/*",
]
