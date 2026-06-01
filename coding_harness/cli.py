"""Command-line entry point.

Usage::

    python -m coding_harness.cli "fix the failing test in app.py"

Configuration is read from ``config.yaml`` by default (override with --config);
individual knobs can be overridden on the command line.
"""

from __future__ import annotations

import argparse
import sys

from .agent import Agent
from .config import Config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coding_harness",
        description="Run an autonomous coding agent on a task.",
    )
    parser.add_argument("task", help="The task for the agent to perform.")
    parser.add_argument(
        "--config", default="config.yaml", help="Path to the YAML config file."
    )
    parser.add_argument("--model", help="Override the model name.")
    parser.add_argument("--base-url", help="Override the API base URL.")
    parser.add_argument("--workspace", help="Override the workspace root directory.")
    parser.add_argument("--max-steps", type=int, help="Override the maximum loop steps.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not execute shell/python commands; echo them instead.",
    )
    parser.add_argument(
        "--no-console",
        action="store_true",
        help="Disable the console mirror (trace file is still written).",
    )
    return parser


def apply_overrides(config: Config, args: argparse.Namespace) -> None:
    if args.model:
        config.llm.model = args.model
    if args.base_url:
        config.llm.base_url = args.base_url
    if args.workspace:
        config.sandbox.workspace_root = args.workspace
    if args.max_steps is not None:
        config.loop.max_steps = args.max_steps
    if args.dry_run:
        config.sandbox.dry_run = True
    if args.no_console:
        config.logging.console = False


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = Config.load(args.config)
    apply_overrides(config, args)

    agent = Agent.create(config)
    result = agent.run(args.task)

    print("\n" + "=" * 60)
    print(f"status : {result.status}")
    print(f"steps  : {result.steps}")
    print(f"tokens : {result.total_tokens}")
    print(f"summary: {result.summary}")
    print("=" * 60)

    # Non-zero exit unless the agent finished cleanly (task_done or prose answer).
    return 0 if result.status in ("done", "answered") else 1


if __name__ == "__main__":
    sys.exit(main())
