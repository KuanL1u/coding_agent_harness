"""EVALUATE — drive the Evaluation Gate: active vs each candidate, head-to-head.

Runs the benchmark suite once for the active version and once per candidate, on
the same tasks, and returns each version's :class:`BenchmarkReport`. A per-cycle
token ceiling stops launching further candidate evaluations once the budget is
spent, so an evolution cycle can never run away on cost.

The benchmark, graders, and runner it calls live outside the agent's jail, so a
candidate cannot game its own score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...config import Config
from ..benchmark.runner import TASKS_DIR, BenchmarkReport, run_benchmark
from ..policy import PromptPolicyVersion


def _report_tokens(report: BenchmarkReport) -> int:
    return sum(r.tokens for r in report.results)


@dataclass
class EvaluationResult:
    active: PromptPolicyVersion
    active_report: BenchmarkReport
    candidates: list[tuple[PromptPolicyVersion, BenchmarkReport]] = field(default_factory=list)
    skipped: list[tuple[PromptPolicyVersion, str]] = field(default_factory=list)
    tokens_spent: int = 0


def evaluate(
    config: Config,
    active: PromptPolicyVersion,
    candidates: list[PromptPolicyVersion],
    *,
    tasks_dir: Path | None = None,
    llm_factory: Any = None,
    work_root: Path | None = None,
    cycle_budget_tokens: int = 0,
) -> EvaluationResult:
    """A/B the active version against each candidate on the benchmark."""
    tasks = tasks_dir or TASKS_DIR

    active_report = run_benchmark(
        config, tasks_dir=tasks, llm_factory=llm_factory, work_root=work_root, policy=active
    )
    result = EvaluationResult(active=active, active_report=active_report)
    result.tokens_spent += _report_tokens(active_report)

    for cand in candidates:
        if cycle_budget_tokens and result.tokens_spent >= cycle_budget_tokens:
            result.skipped.append((cand, "cycle token budget exhausted"))
            continue
        rep = run_benchmark(
            config, tasks_dir=tasks, llm_factory=llm_factory, work_root=work_root, policy=cand
        )
        result.tokens_spent += _report_tokens(rep)
        result.candidates.append((cand, rep))

    return result
