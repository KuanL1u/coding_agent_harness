"""Layer 3's evolver — the offline meta-agent loop.

The pipeline, one stage per module:

    DIAGNOSE  (diagnose.py)  Layer-1 aggregates -> ranked weakness report
    PROPOSE   (propose.py)   weaknesses + active version -> validated patches
    EVALUATE  (evaluate.py)  Evaluation Gate: active vs each candidate, A/B
    DECIDE    (decide.py)    adoption rule + cost-regression budget
    cycle.py                 orchestrates the above + COMMIT, the entry point

It never runs in the live task path — live tasks always use the current active
version. Run one cycle explicitly with::

    python -m coding_harness.evolve.evolver.cycle
"""

from __future__ import annotations

from .decide import Decision, decide_one, pick_best
from .diagnose import Weakness, WeaknessReport, diagnose
from .evaluate import EvaluationResult, evaluate
from .propose import ProposeResult, heuristic_propose, llm_propose, propose

__all__ = [
    "diagnose",
    "Weakness",
    "WeaknessReport",
    "propose",
    "llm_propose",
    "heuristic_propose",
    "ProposeResult",
    "evaluate",
    "EvaluationResult",
    "decide_one",
    "pick_best",
    "Decision",
]
