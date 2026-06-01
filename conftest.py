"""Pytest configuration for the repository test run.

The benchmark suite under ``coding_harness/evolve/benchmark/tasks/`` contains
``test_*.py`` files that are *fixtures for the agent*, not tests of this repo
(they import workspace modules that intentionally start broken). Exclude them
from collection so the harness's own test run is unaffected.
"""

collect_ignore_glob = ["coding_harness/evolve/benchmark/tasks/*"]
