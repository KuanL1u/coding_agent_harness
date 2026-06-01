"""Run the harness against the benchmark suite and score it.

For each task the runner:

1. Copies the task's ``setup/`` into a fresh temp directory — the agent's
   isolated, writable workspace. The canonical task is never mutated.
2. Runs the harness on the task instruction, scoped to that workspace.
3. Invokes the task's ``grade.py`` *as a subprocess, outside the workspace*, so
   the grader code is physically separate from anything the agent could edit.
4. Records pass/fail plus cost metrics (steps, tokens, wall-clock).

It aggregates ``success_rate``, ``avg_steps``, ``avg_tokens`` and
``avg_wall_clock`` — a reliable *relative* signal (A vs B), which is all the
evolution gate needs.

CLI::

    python -m coding_harness.evolve.benchmark.runner --config config.yaml
    python -m coding_harness.evolve.benchmark.runner --compare-memory   # A/B
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from ...agent import Agent
from ...config import Config
from ...llm_client import LLMClient

TASKS_DIR = Path(__file__).parent / "tasks"

# Factory that produces a fresh LLM per task. ``None`` -> build a real client
# from config; tests inject a factory that returns a scripted fake.
LLMFactory = Callable[[Config], Any]


@dataclass
class TaskResult:
    task: str
    passed: bool
    score: float
    status: str
    steps: int
    tokens: int
    wall_clock_s: float
    details: str = ""


@dataclass
class BenchmarkReport:
    results: list[TaskResult] = field(default_factory=list)
    success_rate: float = 0.0
    avg_steps: float = 0.0
    avg_tokens: float = 0.0
    avg_wall_clock: float = 0.0

    def finalize(self) -> "BenchmarkReport":
        n = len(self.results) or 1
        self.success_rate = round(sum(r.passed for r in self.results) / n, 4)
        self.avg_steps = round(sum(r.steps for r in self.results) / n, 2)
        self.avg_tokens = round(sum(r.tokens for r in self.results) / n, 1)
        self.avg_wall_clock = round(sum(r.wall_clock_s for r in self.results) / n, 2)
        return self

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def discover_tasks(tasks_dir: Path = TASKS_DIR) -> list[Path]:
    """Return task directories (those containing a ``task.md``), sorted by name."""
    return sorted(
        p for p in tasks_dir.iterdir() if p.is_dir() and (p / "task.md").is_file()
    )


def _grade(task_dir: Path, workspace: Path) -> dict[str, Any]:
    """Run the task's grader as an isolated subprocess; parse its JSON verdict."""
    grader = task_dir / "grade.py"
    if not grader.is_file():
        return {"passed": False, "score": 0.0, "details": "no grade.py"}
    proc = subprocess.run(
        [sys.executable, str(grader), str(workspace)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    out = (proc.stdout or "").strip().splitlines()
    for line in reversed(out):  # tolerate stray prints; take the last JSON line
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {
        "passed": False,
        "score": 0.0,
        "details": f"grader produced no JSON (rc={proc.returncode}): {proc.stderr[:200]}",
    }


def run_task(
    task_dir: Path,
    config: Config,
    *,
    llm_factory: LLMFactory | None = None,
    work_root: Path | None = None,
    policy: Any = None,
) -> TaskResult:
    """Run the harness on one task in an isolated workspace and grade the result.

    ``policy`` is an optional :class:`PromptPolicyVersion` to run under (the
    evolution gate passes a candidate here to A/B it against the active version).
    """
    task_text = (task_dir / "task.md").read_text(encoding="utf-8").strip()
    setup = task_dir / "setup"

    tmp_parent = Path(tempfile.mkdtemp(prefix=f"bench_{task_dir.name}_", dir=work_root))
    workspace = tmp_parent / "workspace"
    if setup.is_dir():
        shutil.copytree(setup, workspace)
    else:
        workspace.mkdir(parents=True)

    # Clone config so per-task overrides (workspace, trace) never leak between
    # tasks or back into the caller's config.
    cfg = copy.deepcopy(config)
    cfg.sandbox.workspace_root = str(workspace)
    cfg.logging.trace_file = str(tmp_parent / "trace.jsonl")
    cfg.logging.console = False

    llm = llm_factory(cfg) if llm_factory else LLMClient.from_config(cfg.llm)
    agent = Agent.create(cfg, llm=llm, policy=policy)
    try:
        result = agent.run(task_text)
        status, steps, tokens, wall = (
            result.status,
            result.steps,
            result.total_tokens,
            result.wall_clock_s,
        )
    except Exception as exc:  # noqa: BLE001 - a crashing run is a failed task, not a crashed suite
        status, steps, tokens, wall = (f"crash:{type(exc).__name__}", 0, 0, 0.0)

    verdict = _grade(task_dir, workspace)
    return TaskResult(
        task=task_dir.name,
        passed=bool(verdict.get("passed")),
        score=float(verdict.get("score", 0.0)),
        status=status,
        steps=steps,
        tokens=tokens,
        wall_clock_s=wall,
        details=str(verdict.get("details", "")),
    )


def run_benchmark(
    config: Config,
    *,
    tasks_dir: Path = TASKS_DIR,
    llm_factory: LLMFactory | None = None,
    work_root: Path | None = None,
    policy: Any = None,
) -> BenchmarkReport:
    """Run every task and return an aggregated :class:`BenchmarkReport`."""
    report = BenchmarkReport()
    for task_dir in discover_tasks(tasks_dir):
        report.results.append(
            run_task(
                task_dir,
                config,
                llm_factory=llm_factory,
                work_root=work_root,
                policy=policy,
            )
        )
    return report.finalize()


def _print_report(report: BenchmarkReport, label: str = "") -> None:
    head = f"benchmark report{' — ' + label if label else ''}"
    print("\n" + "=" * 60)
    print(head)
    print("=" * 60)
    for r in report.results:
        mark = "PASS" if r.passed else "FAIL"
        print(
            f"[{mark}] {r.task:<22} status={r.status:<10} "
            f"steps={r.steps:<3} tokens={r.tokens:<7} {r.wall_clock_s}s"
        )
        if not r.passed and r.details:
            print(f"        {r.details}")
    print("-" * 60)
    print(
        f"success_rate={report.success_rate}  avg_steps={report.avg_steps}  "
        f"avg_tokens={report.avg_tokens}  avg_wall_clock={report.avg_wall_clock}s"
    )
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="coding_harness.evolve.benchmark.runner",
        description="Run the harness against the benchmark suite and score it.",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML config.")
    parser.add_argument("--model", help="Override the model name.")
    parser.add_argument("--base-url", help="Override the API base URL.")
    parser.add_argument(
        "--compare-memory",
        action="store_true",
        help="Run the suite twice (memory OFF then ON) and report both — the L1-M4 A/B.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the report(s) as JSON.")
    args = parser.parse_args(argv)

    config = Config.load(args.config)
    if args.model:
        config.llm.model = args.model
    if args.base_url:
        config.llm.base_url = args.base_url

    if args.compare_memory:
        off = copy.deepcopy(config)
        off.memory.enabled = False
        on = copy.deepcopy(config)
        on.memory.enabled = True
        rep_off = run_benchmark(off)
        rep_on = run_benchmark(on)
        if args.json:
            print(json.dumps({"memory_off": rep_off.to_dict(), "memory_on": rep_on.to_dict()}))
        else:
            _print_report(rep_off, "memory OFF")
            _print_report(rep_on, "memory ON")
            print(
                f"\nΔ success_rate (on - off): "
                f"{round(rep_on.success_rate - rep_off.success_rate, 4)}"
            )
        return 0

    report = run_benchmark(config)
    if args.json:
        print(json.dumps(report.to_dict()))
    else:
        _print_report(report)
    return 0 if report.success_rate == 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
