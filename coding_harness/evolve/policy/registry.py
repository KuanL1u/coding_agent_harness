"""Versioned, addressable prompt/policy artifacts.

A :class:`PromptPolicyVersion` is the full tunable surface the evolver may change
— the system prompt, per-tool description overrides, loop budgets, temperature,
and the parallel-tool-call flag. Versions are stored as JSON so outcomes can be
attributed to an exact version and rollback is trivial (it is just a file +
pointer change, committable to git).

:class:`PromptPolicyRegistry` is the small store the agent and the evolver share:
``get_active`` resolves the version the live path should use; ``save`` /
``set_active`` / ``archive`` let the cycle adopt or shelve candidates.

Note what is *absent* from this surface: nothing here can change the sandbox,
deny-list, timeouts, workspace root, tool execution code, or the benchmark. Those
safety surfaces are structurally off-limits to the evolver (see
:mod:`coding_harness.evolve.policy.patch`).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_VERSION_NUM = re.compile(r"^p(\d+)$")


@dataclass
class PromptPolicyVersion:
    """One addressable prompt/policy artifact."""

    version: str
    prompt_text: str
    max_steps: int
    max_total_tokens: int
    temperature: float
    parallel_tool_calls: bool = True
    # Per-tool description overrides applied on top of the code defaults. Keys
    # are tool names; an absent tool keeps its built-in description.
    tool_descriptions: dict[str, str] = field(default_factory=dict)
    parent: str | None = None
    rationale: str = ""
    created: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PromptPolicyVersion":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


def seed_version(
    *,
    prompt_text: str,
    max_steps: int,
    max_total_tokens: int,
    temperature: float,
    parallel_tool_calls: bool,
    version: str = "p1",
) -> PromptPolicyVersion:
    """Build the initial version (``p1``) from the harness's code defaults.

    ``tool_descriptions`` starts empty — meaning "use the built-in descriptions"
    — so the seed reproduces v1 behaviour exactly before any evolution.
    """
    return PromptPolicyVersion(
        version=version,
        prompt_text=prompt_text,
        max_steps=max_steps,
        max_total_tokens=max_total_tokens,
        temperature=temperature,
        parallel_tool_calls=parallel_tool_calls,
        tool_descriptions={},
        parent=None,
        rationale="seed: initial version from code defaults",
        created=time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
    )


class PromptPolicyRegistry:
    """File-backed store of :class:`PromptPolicyVersion` artifacts + an active pointer."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.archive_dir = self.root / "archive"
        self._active_file = self.root / "_active.json"

    # -- version files ------------------------------------------------------

    def _path(self, version: str) -> Path:
        return self.root / f"{version}.json"

    def get(self, version: str) -> PromptPolicyVersion | None:
        path = self._path(version)
        if not path.is_file():
            return None
        return PromptPolicyVersion.from_dict(json.loads(path.read_text("utf-8")))

    def save(self, version: PromptPolicyVersion) -> None:
        self._path(version.version).write_text(version.to_json(), encoding="utf-8")

    def list_versions(self) -> list[str]:
        return sorted(
            (p.stem for p in self.root.glob("*.json") if p.name != "_active.json"),
            key=_version_sort_key,
        )

    def next_version_id(self) -> str:
        """Return the next ``p<N>`` id not already taken."""
        nums = [
            int(m.group(1))
            for v in self.list_versions()
            if (m := _VERSION_NUM.match(v))
        ]
        return f"p{(max(nums) + 1) if nums else 1}"

    # -- active pointer -----------------------------------------------------

    def set_active(self, version: str) -> None:
        if self.get(version) is None:
            raise ValueError(f"cannot activate unknown version: {version!r}")
        self._active_file.write_text(json.dumps({"active": version}), encoding="utf-8")

    def active_id(self, pin: str = "") -> str | None:
        """The active version id: an explicit ``pin`` wins, else the pointer file."""
        if pin:
            return pin
        if self._active_file.is_file():
            return json.loads(self._active_file.read_text("utf-8")).get("active")
        return None

    def get_active(self, pin: str = "") -> PromptPolicyVersion | None:
        vid = self.active_id(pin)
        return self.get(vid) if vid else None

    def seed(self, version: PromptPolicyVersion) -> PromptPolicyVersion:
        """Save ``version`` and make it active (used to bootstrap an empty registry)."""
        self.save(version)
        self.set_active(version.version)
        return version

    # -- audit --------------------------------------------------------------

    def archive(self, version: PromptPolicyVersion, reason: str) -> Path:
        """Persist a rejected candidate (with the reason) for audit, not activated."""
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        payload = version.to_dict()
        payload["archived_reason"] = reason
        payload["archived_at"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        path = self.archive_dir / f"{version.version}.json"
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path


def _version_sort_key(name: str) -> tuple[int, Any]:
    m = _VERSION_NUM.match(name)
    return (0, int(m.group(1))) if m else (1, name)
