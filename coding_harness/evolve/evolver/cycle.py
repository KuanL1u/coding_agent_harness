"""The evolution cycle: DIAGNOSE -> PROPOSE -> MATERIALIZE -> EVALUATE -> DECIDE -> COMMIT.

This is the offline entry point. It reads the experience store, proposes
benchmark-validated prompt/policy changes, and adopts a candidate only when it
beats the active version on the Evaluation Gate within the cost budget. Adoption
is a registry change (new version file + active pointer) and, optionally, a git
commit carrying the benchmark delta — so every change is attributable and a
one-command revert away. Rejected candidates are archived with their reason.

Safety rails enforced here and below:
* The evolver can only emit whitelisted ``PromptPolicyPatch`` objects.
* The benchmark/graders/runner live outside the agent's jail.
* A hard per-cycle token ceiling bounds cost; candidates are capped per cycle.
* The git commit is opt-in (``--commit`` / ``do_commit``); the registry state is
  always persisted regardless.

CLI::

    python -m coding_harness.evolve.evolver.cycle --config config.yaml
    python -m coding_harness.evolve.evolver.cycle --heuristic   # no LLM proposer
    python -m coding_harness.evolve.evolver.cycle --commit      # also git-commit an adoption
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ...config import Config
from ...llm_client import LLMClient
from ...prompts import SYSTEM_PROMPT
from ...sandbox import WorkspaceJail
from ...tools import TaskState
from ..memory.store import LocalFileStore
from ..policy import (
    PromptPolicyRegistry,
    PromptPolicyVersion,
    materialize,
    seed_version,
)
from .decide import decide_one
from .diagnose import diagnose
from .evaluate import evaluate
from .propose import propose

OnEvent = Callable[[str, dict[str, Any]], None]


@dataclass
class CycleResult:
    active_version: str
    sample_size: int
    weaknesses: list[dict[str, Any]] = field(default_factory=list)
    proposed: int = 0
    rejected: list[str] = field(default_factory=list)
    evaluated: list[dict[str, Any]] = field(default_factory=list)
    adopted_version: str | None = None
    decision_reason: str = ""
    committed: bool = False
    tokens_spent: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_version": self.active_version,
            "sample_size": self.sample_size,
            "weaknesses": self.weaknesses,
            "proposed": self.proposed,
            "rejected": self.rejected,
            "evaluated": self.evaluated,
            "adopted_version": self.adopted_version,
            "decision_reason": self.decision_reason,
            "committed": self.committed,
            "tokens_spent": self.tokens_spent,
        }


def _known_tools(config: Config) -> set[str]:
    """The registered tool names (for validating tool_description_overrides)."""
    from ...agent import build_registry  # local import to avoid an import cycle

    with tempfile.TemporaryDirectory() as tmp:
        registry = build_registry(config, WorkspaceJail(tmp), TaskState())
        return set(registry.names())


def _ensure_active(
    config: Config, registry: PromptPolicyRegistry
) -> PromptPolicyVersion:
    active = registry.get_active(config.evolve.active_policy_version)
    if active is None:
        active = registry.seed(
            seed_version(
                prompt_text=SYSTEM_PROMPT,
                max_steps=config.loop.max_steps,
                max_total_tokens=config.loop.max_total_tokens,
                temperature=config.llm.temperature,
                parallel_tool_calls=config.llm.parallel_tool_calls,
            )
        )
    return active


def run_cycle(
    config: Config,
    *,
    complete: Callable[..., Any] | None = None,
    llm_factory: Any = None,
    tasks_dir: Path | None = None,
    store: Any = None,
    registry: PromptPolicyRegistry | None = None,
    work_root: Path | None = None,
    do_commit: bool = False,
    max_candidates: int | None = None,
    on_event: OnEvent | None = None,
) -> CycleResult:
    """Run one full evolution cycle and return a structured result.

    ``complete`` (an LLM ``complete`` callable) drives PROPOSE; when ``None`` the
    deterministic heuristic proposer is used. ``llm_factory`` drives the agent
    runs inside the benchmark; when ``None`` a real client is built per task.
    """

    def emit(event: str, **fields: Any) -> None:
        if on_event is not None:
            on_event(event, fields)

    registry = registry or PromptPolicyRegistry(config.evolve.policy_dir)
    store = store or LocalFileStore(config.memory.store_path)
    cap = max_candidates if max_candidates is not None else config.evolve.max_candidates_per_cycle

    active = _ensure_active(config, registry)
    result = CycleResult(active_version=active.version, sample_size=0)
    emit("cycle_start", active_version=active.version)

    # DIAGNOSE
    report = diagnose(store)
    result.sample_size = report.sample_size
    result.weaknesses = [w.__dict__ for w in report.weaknesses]
    emit("cycle_diagnose", sample_size=report.sample_size, weaknesses=len(report.weaknesses))
    if report.sample_size == 0:
        emit("cycle_end", adopted=None, reason="no episodes to learn from")
        result.decision_reason = "no episodes in the experience store yet"
        return result

    # PROPOSE -> MATERIALIZE
    proposal = propose(
        active, report, complete=complete, max_candidates=cap, known_tools=_known_tools(config)
    )
    result.proposed = len(proposal.patches)
    result.rejected = list(proposal.rejected)
    emit("cycle_propose", patches=len(proposal.patches), rejected=len(proposal.rejected))
    if not proposal.patches:
        result.decision_reason = "no valid candidate patches were proposed"
        emit("cycle_end", adopted=None, reason=result.decision_reason)
        return result

    candidates = [
        materialize(patch, active, new_version_id=f"cand{i + 1}")
        for i, patch in enumerate(proposal.patches)
    ]

    # EVALUATE (Evaluation Gate, with the per-cycle token ceiling)
    evaluation = evaluate(
        config,
        active,
        candidates,
        tasks_dir=tasks_dir,
        llm_factory=llm_factory,
        work_root=work_root,
        cycle_budget_tokens=config.evolve.cycle_budget_tokens,
    )
    result.tokens_spent = evaluation.tokens_spent
    emit(
        "cycle_evaluate",
        active_success=evaluation.active_report.success_rate,
        candidates=len(evaluation.candidates),
        skipped=len(evaluation.skipped),
        tokens_spent=evaluation.tokens_spent,
    )

    # DECIDE
    decisions: list[tuple[PromptPolicyVersion, Any, Any]] = []
    for version, rep in evaluation.candidates:
        decision = decide_one(
            evaluation.active_report,
            rep,
            adopt_epsilon=config.evolve.adopt_epsilon,
            max_cost_regression=config.evolve.max_cost_regression,
        )
        decisions.append((version, rep, decision))
        result.evaluated.append(
            {
                "candidate": version.version,
                "rationale": version.rationale,
                "success_rate": rep.success_rate,
                "avg_tokens": rep.avg_tokens,
                "avg_steps": rep.avg_steps,
                "adopt": decision.adopt,
                "reason": decision.reason,
            }
        )

    # Candidates skipped for budget never got a score; archive them as such.
    for version, reason in evaluation.skipped:
        registry.archive(version, reason)
        result.evaluated.append(
            {"candidate": version.version, "rationale": version.rationale,
             "adopt": False, "reason": reason}
        )

    adoptable = [(v, r, d) for v, r, d in decisions if d.adopt]
    if not adoptable:
        # COMMIT (reject path): archive every evaluated candidate with its reason.
        for version, _rep, decision in decisions:
            registry.archive(version, decision.reason)
        result.decision_reason = (
            "no candidate beat the active version within the cost budget"
        )
        emit("cycle_end", adopted=None, reason=result.decision_reason)
        return result

    # Best adoptable: highest success rate, ties broken by lower token cost.
    best_v, best_r, best_d = max(adoptable, key=lambda t: (t[1].success_rate, -t[1].avg_tokens))

    # COMMIT (adopt path): re-id under the next pN, save, activate; archive the rest.
    new_id = registry.next_version_id()
    adopted = materialize_rename(best_v, new_id)
    registry.save(adopted)
    registry.set_active(adopted.version)
    for version, _rep, decision in decisions:
        if version is not best_v:
            registry.archive(version, decision.reason)

    result.adopted_version = adopted.version
    result.decision_reason = best_d.reason
    emit(
        "cycle_adopt",
        version=adopted.version,
        parent=active.version,
        success_delta=best_d.success_delta,
        reason=best_d.reason,
    )

    if do_commit:
        result.committed = _git_commit(
            registry.root,
            f"evolve: adopt {adopted.version} (parent {active.version})\n\n"
            f"{best_d.reason}\n"
            f"active success_rate {evaluation.active_report.success_rate} -> "
            f"{best_r.success_rate}\nrationale: {adopted.rationale}",
        )
        emit("cycle_commit", committed=result.committed, version=adopted.version)

    emit("cycle_end", adopted=adopted.version, reason=best_d.reason)
    return result


def materialize_rename(version: PromptPolicyVersion, new_id: str) -> PromptPolicyVersion:
    """Copy ``version`` under a new id (used to give an adopted candidate a pN id)."""
    data = version.to_dict()
    data["version"] = new_id
    return PromptPolicyVersion.from_dict(data)


def _git_commit(path: Path, message: str) -> bool:
    """Stage ``path`` and commit. Returns True on success, False on any failure."""
    try:
        subprocess.run(["git", "add", str(path)], check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", message], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _print_result(result: CycleResult) -> None:
    print("\n" + "=" * 60)
    print(f"evolution cycle — active version: {result.active_version}")
    print("=" * 60)
    print(f"episodes analysed : {result.sample_size}")
    if result.weaknesses:
        print("top weaknesses    :")
        for w in result.weaknesses[:5]:
            print(f"  - [{w['weight']}] {w['id']}: {w['description']}")
    print(f"candidates        : proposed={result.proposed} rejected={len(result.rejected)}")
    for ev in result.evaluated:
        mark = "ADOPT?" if ev["adopt"] else "reject"
        print(
            f"  [{mark}] {ev['candidate']:<8} success={ev['success_rate']} "
            f"tokens={ev['avg_tokens']} — {ev['reason']}"
        )
    print("-" * 60)
    if result.adopted_version:
        print(f"ADOPTED {result.adopted_version}: {result.decision_reason}")
        print(f"committed to git  : {result.committed}")
    else:
        print(f"no adoption: {result.decision_reason}")
    print(f"tokens spent      : {result.tokens_spent}")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="coding_harness.evolve.evolver.cycle",
        description="Run one prompt/policy self-tuning cycle (offline).",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML config.")
    parser.add_argument("--model", help="Override the model name.")
    parser.add_argument("--base-url", help="Override the API base URL.")
    parser.add_argument(
        "--heuristic",
        action="store_true",
        help="Use the deterministic heuristic proposer instead of the LLM meta-agent.",
    )
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Also git-commit an adopted version (registry state is saved regardless).",
    )
    parser.add_argument("--max-candidates", type=int, help="Cap candidates this cycle.")
    parser.add_argument("--json", action="store_true", help="Emit the result as JSON.")
    args = parser.parse_args(argv)

    config = Config.load(args.config)
    if args.model:
        config.llm.model = args.model
    if args.base_url:
        config.llm.base_url = args.base_url

    complete = None
    if not args.heuristic:
        complete = LLMClient.from_config(config.llm).complete

    result = run_cycle(
        config,
        complete=complete,
        do_commit=args.commit,
        max_candidates=args.max_candidates,
    )

    if args.json:
        print(json.dumps(result.to_dict()))
    else:
        _print_result(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
