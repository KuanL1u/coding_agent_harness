"""The versioned prompt/policy surface the evolver tunes.

* :class:`PromptPolicyVersion` / :class:`PromptPolicyRegistry` — addressable,
  swappable artifacts with an active pointer (``registry.py``).
* :class:`PromptPolicyPatch` / :func:`materialize` — the strict whitelisted edit
  shape the evolver may emit, and how it becomes a new version (``patch.py``).
"""

from __future__ import annotations

from .patch import (
    PatchValidationError,
    PromptPolicyPatch,
    materialize,
)
from .registry import PromptPolicyRegistry, PromptPolicyVersion, seed_version

__all__ = [
    "PromptPolicyVersion",
    "PromptPolicyRegistry",
    "seed_version",
    "PromptPolicyPatch",
    "PatchValidationError",
    "materialize",
]
