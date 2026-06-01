"""DECIDE — the adoption rule + regression budget.

A candidate is adopted only if it beats the active version's benchmark success
rate by at least ``adopt_epsilon`` *and* does not regress cost (avg tokens or
avg steps) by more than ``max_cost_regression``. Requiring a margin guards
against adopting a single-task fluke; the cost budget guards against buying a
small win with a large cost increase.

:func:`pick_best` chooses, among the candidates that pass, the one with the
highest success rate (ties broken by lower cost).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Decision:
    adopt: bool
    reason: str
    success_delta: float = 0.0
    cost_regression: float = 0.0


def _cost_regression(active: Any, candidate: Any) -> float:
    """Worst-case relative cost increase across avg tokens and avg steps.

    Returns e.g. 0.2 for a 20% increase; <= 0 means no regression (cheaper).
    """
    ratios = []
    for attr in ("avg_tokens", "avg_steps"):
        base = getattr(active, attr)
        cand = getattr(candidate, attr)
        if base > 0:
            ratios.append(cand / base - 1.0)
        elif cand > 0:
            ratios.append(float("inf"))  # went from free to non-free
    return max(ratios) if ratios else 0.0


def decide_one(
    active_report: Any,
    candidate_report: Any,
    *,
    adopt_epsilon: float,
    max_cost_regression: float,
) -> Decision:
    """Adopt/reject a single candidate against the active version."""
    success_delta = round(
        candidate_report.success_rate - active_report.success_rate, 4
    )
    cost_reg = round(_cost_regression(active_report, candidate_report), 4)

    if success_delta < adopt_epsilon:
        return Decision(
            adopt=False,
            reason=(
                f"success gain {success_delta:+.4f} < required +{adopt_epsilon:.4f}"
            ),
            success_delta=success_delta,
            cost_regression=cost_reg,
        )
    if cost_reg > max_cost_regression:
        return Decision(
            adopt=False,
            reason=(
                f"cost regression {cost_reg:+.2%} exceeds budget "
                f"{max_cost_regression:.2%} (success gain was {success_delta:+.4f})"
            ),
            success_delta=success_delta,
            cost_regression=cost_reg,
        )
    return Decision(
        adopt=True,
        reason=(
            f"success {success_delta:+.4f} (>= +{adopt_epsilon:.4f}); "
            f"cost {cost_reg:+.2%} within budget {max_cost_regression:.2%}"
        ),
        success_delta=success_delta,
        cost_regression=cost_reg,
    )


def pick_best(
    evaluation: Any,
    *,
    adopt_epsilon: float,
    max_cost_regression: float,
) -> tuple[Any, Any, Decision] | None:
    """Return (version, report, Decision) of the best adoptable candidate, or None.

    Each candidate is judged against the active version; among those that pass,
    the highest success rate wins, ties broken by lower average token cost.
    """
    passing: list[tuple[Any, Any, Decision]] = []
    for version, report in evaluation.candidates:
        decision = decide_one(
            evaluation.active_report,
            report,
            adopt_epsilon=adopt_epsilon,
            max_cost_regression=max_cost_regression,
        )
        if decision.adopt:
            passing.append((version, report, decision))

    if not passing:
        return None
    return max(passing, key=lambda t: (t[1].success_rate, -t[1].avg_tokens))
