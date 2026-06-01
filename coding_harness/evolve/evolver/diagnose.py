"""DIAGNOSE — turn experience-store aggregates into a ranked weakness report.

Reads the episode store and computes evidence-backed weaknesses the evolver can
target: premature ``task_done`` (declared success while tests were failing),
budget-exhaustion rate, recurring failure signatures, and a low overall success
rate. Each weakness carries a numeric ``weight`` (roughly, how much of the
failure mass it explains) so PROPOSE can focus on what matters most.

This stage is pure analysis — it reads, it never changes anything.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Weakness:
    """One diagnosed weakness with its supporting evidence."""

    id: str
    description: str
    weight: float  # 0..1-ish; higher = more important to fix
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class WeaknessReport:
    """Ranked weaknesses + the raw aggregates they were derived from."""

    weaknesses: list[Weakness] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    sample_size: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "weaknesses": [asdict(w) for w in self.weaknesses],
            "stats": self.stats,
            "sample_size": self.sample_size,
        }

    def top(self, n: int) -> list[Weakness]:
        return self.weaknesses[:n]


def diagnose(store: Any) -> WeaknessReport:
    """Produce a :class:`WeaknessReport` from the experience ``store``."""
    episodes = list(store.iter_episodes())
    total = len(episodes)
    report = WeaknessReport(sample_size=total)
    if total == 0:
        return report

    failure_stats = store.failure_stats()
    report.stats = failure_stats

    successes = [e for e in episodes if e.outcome == "success"]
    budget_stops = [e for e in episodes if e.outcome == "stopped_budget"]

    weaknesses: list[Weakness] = []

    # 1) Premature task_done: declared success while the last test run failed.
    premature = [e for e in successes if e.test_result == "failed"]
    if premature:
        share = len(premature) / max(len(successes), 1)
        weaknesses.append(
            Weakness(
                id="premature_task_done",
                description=(
                    "The agent called task_done while tests were still failing in "
                    f"{len(premature)}/{len(successes)} 'successful' runs."
                ),
                weight=round(share, 3),
                evidence={
                    "count": len(premature),
                    "of_successes": len(successes),
                    "share": round(share, 3),
                    "example_tasks": [e.task for e in premature[:3]],
                },
            )
        )

    # 2) Budget exhaustion: runs that ran out of steps/tokens without finishing.
    if budget_stops:
        rate = len(budget_stops) / total
        weaknesses.append(
            Weakness(
                id="budget_exhaustion",
                description=(
                    f"{len(budget_stops)}/{total} runs hit a step/token budget "
                    "without completing the task."
                ),
                weight=round(rate, 3),
                evidence={
                    "count": len(budget_stops),
                    "of_total": total,
                    "rate": round(rate, 3),
                    "example_tasks": [e.task for e in budget_stops[:3]],
                },
            )
        )

    # 3) Recurring failure signatures from the aggregate.
    for sig, info in list(failure_stats.get("signatures", {}).items())[:3]:
        weaknesses.append(
            Weakness(
                id=f"failure_signature:{sig}",
                description=f"Recurring failure: {sig}",
                weight=round(info.get("share", 0.0) * failure_stats.get("failure_rate", 0.0), 3),
                evidence={"signature": sig, **info},
            )
        )

    # 4) Low overall success rate (a catch-all signal).
    success_rate = len(successes) / total
    if success_rate < 0.8:
        weaknesses.append(
            Weakness(
                id="low_success_rate",
                description=f"Overall success rate is {round(success_rate, 3)} across {total} runs.",
                weight=round(1.0 - success_rate, 3),
                evidence={"success_rate": round(success_rate, 3), "total": total},
            )
        )

    weaknesses.sort(key=lambda w: w.weight, reverse=True)
    report.weaknesses = weaknesses
    return report
