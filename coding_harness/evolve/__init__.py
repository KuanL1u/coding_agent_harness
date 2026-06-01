"""Self-evolving harness capabilities built on top of the v1 ReAct agent.

This package adds capabilities that turn the harness from a stateless
task-runner into a system that learns from its own runs:

* ``memory`` — persist a structured record of every run and inject the most
  relevant past experience into future runs (retrieval-augmented context).
* ``benchmark`` — the shared Evaluation Gate: a fixed set of held-out tasks plus
  a runner that scores the harness, used to prove whether a change actually
  helped.

Prompt/policy self-tuning (``policy`` + ``evolver``) builds on these and lives
alongside them.
"""
