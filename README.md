# coding_harness

A from-scratch **autonomous coding agent** built on the `openai` SDK and the
Python standard library — no LangChain, no heavy frameworks.

**North star:** a coding agent harness that *self-evolves* — one that learns
from its own runs, measures whether a change actually helps on a held-out
benchmark, and then safely rewrites its own prompt, policy, and (eventually)
toolset to get better over time, without a human editing the agent by hand.

The repo builds toward that goal in layers, each safe and useful on its own:

1. **A solid base agent** — a budgeted ReAct loop over native tool calling,
   confined to a sandboxed workspace (shipped, see below).
2. **Experience / memory** — distil every run into an episode and inject the
   most relevant past experience into future runs (shipped).
3. **An evaluation gate** — a held-out benchmark that gives a *relative* signal
   for whether a change helped (shipped).
4. **A prompt/policy evolver** — propose, benchmark, and adopt prompt/policy
   edits that beat the active version, gated by structural safety rails
   (shipped; runs offline and human-reviewed).
5. **A pluggable, self-extending toolset** — tools discovered at runtime, with a
   roadmap to letting the agent author new tools behind a PR gate (registry
   shipped; the self-extension loop is not built yet).

The base agent runs a **ReAct loop** (reason → act → observe → repeat) using
**native OpenAI-compatible function/tool calling**, executes tools locally
inside a confined workspace, and writes a structured trace of everything it
does. Everything beyond the base agent is **off by default** and never sits in
the live task path unless explicitly enabled.

## How it works

```
system prompt + task
        │
        ▼
  ┌───────────────┐   tools=registry.schemas()
  │  LLMClient    │ ───────────────────────────►  Chat Completions API
  │  .complete()  │ ◄───────────────────────────  assistant message
  └───────────────┘        (+ tool_calls)
        │
        ▼
  for each tool_call:  registry.dispatch() → tool-role result message
        │
        ▼
  append results, repeat until task_done() / budget exhausted
```

- **Native tool calling**: the model emits structured `tool_calls`; the harness
  validates arguments against each tool's JSON schema and dispatches them. No
  brittle text parsing of `Thought:`/`Action:` blocks.
- **Parallel tool calls** are supported — every call in a turn gets its own
  `tool`-role result message keyed by `tool_call_id`.
- **Self-correcting**: tool errors are returned to the model as observations
  (prefixed `ERROR:`), so a bad call never crashes the loop.
- **Budgeted**: the loop stops on `task_done`, on `max_steps`, or on
  `max_total_tokens`.

## Tools (v1)

| Tool | Purpose |
|------|---------|
| `read_file` | Read a file, optionally a 1-based line range |
| `write_file` | Create/overwrite a file |
| `edit_file` | Exact-string replacement (unique match unless `replace_all`) |
| `list_dir` | List a directory |
| `grep_search` | Regex search over file contents |
| `glob_search` | Find files by glob pattern |
| `run_shell` | Run a shell command |
| `run_python` | Run inline code or a `.py` file |
| `run_tests` | Run `pytest` |
| `task_done` | Signal successful completion |

## Pluggable tool registry

Tools are **discovered at runtime from a directory**, not hardcoded at import.
Every tool — built-in or (later) agent-authored — lives as a self-describing
plugin folder under `coding_harness/tools/plugins/`:

```
plugins/<name>/
  tool.py        # exposes build(ctx) -> Tool | list[Tool]
  manifest.json  # metadata + lifecycle status ("staged" | "active" | "disabled")
  test_tool.py   # unit tests the validation gate runs
```

At startup `PluginLoader` scans that directory, loads only plugins whose
manifest `status == "active"`, calls each plugin's `build(ctx)` with a sandboxed
**`ToolContext`**, validates every produced tool's schema, and registers the
survivors. The built-ins (`files`, `search`, `exec`, `control`) are migrated
into this same layout and pin a `load_priority`, so the tool list keeps its v1
order and **runs are behaviourally identical** — the agent loop is unchanged
(`registry.schemas()` / `registry.dispatch()`).

- **`ToolContext` is the only capability surface.** A plugin's `build(ctx)`
  receives `ctx.jail` (the `WorkspaceJail`), `ctx.run_subprocess` (the
  deny-listed, timed, truncated, dry-run-aware runner), and `ctx.config` (a
  read-only view). A tool inherits all v1 safety automatically and has no
  privileged path to bypass it.
- **Discovery is failure-isolated.** A bad manifest, an import error, a build
  error, a malformed schema, or a duplicate tool name rejects only that one
  plugin (recorded and emitted as a `plugin_rejected` event) — it never aborts
  the load. A clean built-in load emits nothing, so a normal run's trace is
  unchanged.

> The registry is pluggable and reusable for manual tool additions today. The
> memory-driven loop that *proposes* new tools (gap detection → propose → stage →
> validate → open PR), the staging quarantine, and the forbidden-import scan are
> **not built yet**. The `tool_evolution.enabled` flag gates that loop and is off
> by default; new tools will only ever activate by merging a human-reviewed pull
> request.

## Safety

All execution is confined by `sandbox.py`:

- **Path jail** — every path is resolved and confined to `workspace_root`;
  `..` traversal, absolute paths outside the root, and escaping symlinks are
  rejected.
- **Command timeouts** — each subprocess runs in its own process group and is
  killed (whole group) on timeout.
- **Output truncation** — every tool result is truncated to `max_output_bytes`.
- **Deny-list** — commands matching configured regexes (e.g. `rm -rf /`) are
  refused.
- **Dry-run** — `--dry-run` echoes commands instead of executing them.

## Experience layer (memory)

The harness can **learn from its own runs**. When `memory.enabled` is set, every
run is distilled into a structured **episode** (outcome, cost, tools used, a diff
summary, the test result, and — on failure — a compact failure signature) and
appended to a local store. At the *start* of each run the incoming task is
embedded and the most similar past episodes and distilled **playbooks** are
injected as a short, token-budgeted "Relevant experience" block — so the agent
benefits from what worked (and what failed) on similar tasks before.

- **Safe by construction**: this layer only *adds context*; it never modifies
  code or safety config. It is fully failure-isolated — a memory error degrades
  to "no experience", never a crashed run — and is a no-op when disabled.
- **No embeddings endpoint required**: `embedding_model: local` (the default)
  uses a dependency-free local embedder, so memory works against any chat
  endpoint (including ones with no `/v1/embeddings`). Point it at a real
  embedding model name to use a remote endpoint instead.
- **Playbooks**: every `distill_every_n_episodes` runs, clusters of similar
  successful episodes are distilled into reusable step-by-step playbooks.

Each episode carries a `config_snapshot` (prompt version + budgets + temperature)
so outcomes can later be attributed to a specific prompt/policy version.

## Evaluation Gate (benchmark)

A small, fixed set of held-out coding tasks plus a runner that scores the harness
— the reliable *relative* signal used to tell whether a change actually helped.

```bash
# run the suite (exit 0 only if every task passes)
python -m coding_harness.evolve.benchmark.runner --config config.yaml

# A/B the experience layer: run the suite with memory OFF then ON
python -m coding_harness.evolve.benchmark.runner --compare-memory
```

Each task in `coding_harness/evolve/benchmark/tasks/<name>/` has a `task.md`
(the instruction), a `setup/` (initial workspace, copied into a fresh isolated
jail per run), and a self-contained `grade.py` that prints a JSON verdict. The
tasks, graders, and runner live **outside the agent's writable workspace** — the
agent can never edit what scores it. The runner reports `success_rate`,
`avg_steps`, `avg_tokens`, and `avg_wall_clock`.

## Prompt & policy self-tuning (evolver)

The harness can **rewrite its own prompt and policy** when the benchmark proves
the change helps. A versioned **prompt/policy registry** holds the tunable
surface — the system prompt, per-tool *descriptions*, loop budgets, temperature,
and the parallel-call flag — as addressable artifacts with an active pointer.
When `evolve.enabled` is set, the agent loads the **active version** instead of
the hardcoded prompt/static config, so the active version is the source of truth.

An offline **evolver** then runs this loop:

```
DIAGNOSE → PROPOSE → MATERIALIZE → EVALUATE → DECIDE → COMMIT
```

- **DIAGNOSE** reads the experience-store aggregates into a ranked weakness report
  (premature `task_done`, budget exhaustion, recurring failure signatures, …).
- **PROPOSE** (an LLM meta-agent, with a deterministic heuristic fallback) emits
  candidate edits.
- **EVALUATE** runs the Evaluation Gate: active vs each candidate, head-to-head.
- **DECIDE** adopts a candidate only if it beats the active version by
  `adopt_epsilon` **and** stays within the `max_cost_regression` budget.
- **COMMIT** activates the winner (a new version file + pointer) and, with
  `--commit`, makes a git commit carrying the benchmark delta; rejected
  candidates are archived with their reason.

```bash
python -m coding_harness.evolve.evolver.cycle --config config.yaml
python -m coding_harness.evolve.evolver.cycle --heuristic   # skip the LLM proposer
python -m coding_harness.evolve.evolver.cycle --commit      # also git-commit an adoption
```

**Safety is structural, not advisory.** The evolver's *only* output is a
whitelisted `PromptPolicyPatch`; there is no representable way for it to touch
the sandbox, deny-list, timeouts, workspace root, tool execution code, or the
benchmark. Out-of-range values are rejected too. A hard per-cycle token ceiling
bounds cost, the cycle runs offline (never in the live task path), and every
adoption is a revertible registry/git change.

> **Not built yet:** the evolution cycle currently runs only when invoked
> manually (the CLI above). **Scheduled cadence** — running the cycle
> automatically on a cron-style schedule or after every *K* live runs — and the
> **success-rate-over-versions dashboards** are not implemented. The `cadence` /
> `every_n_runs` config fields are accepted but are advisory metadata only; no
> scheduler reads them yet.

## Trusting the evolution signal (known limitations)

The scaffolding for self-evolution is safe by construction in what it *can't*
touch (sandbox, deny-list, timeouts, workspace, tool code — all unreachable from
a `PromptPolicyPatch`). The open work is in the *measurement* that drives
evolution: until the items below are addressed, the loop should be run with a
human reviewing each adoption rather than fully unattended, because the fitness
signal is not yet strong enough to prevent drift or overfitting.

### 1. The fitness function is statistically underpowered and overfittable

The evolver adopts a change only when the benchmark "proves" it helps — but the
benchmark can't prove much yet:

- **Only 3 tasks, run once each**, so `success_rate ∈ {0, 0.33, 0.67, 1.0}`.
- `adopt_epsilon = 0.03` is meant to guard against a single-task fluke, but the
  smallest possible improvement (one task flipping) is `0.333` — ~10× the
  threshold, so the margin guard is effectively inert.
- **No repeated trials / no variance estimate.** Active and candidate are each
  run once; even at `temperature=0.0` most endpoints are nondeterministic, so a
  truly-equal candidate can win on noise — and with `--commit` that noise is
  written to git history as "evolution."
- **No held-out split.** The prompt is tuned against the exact tasks it is
  scored on (Goodhart): the system can learn to special-case the gate rather
  than improve generally.

  *Direction:* many more tasks; `N` repeated trials per task with mean ± stderr;
  a DECIDE rule based on a significance test / confidence interval rather than a
  flat epsilon on a 3-point scale; and a **dev set** (drives PROPOSE) separate
  from a **held-out set** (gates ADOPT) so overfitting is detected, not rewarded.

### 2. The evaluation gate is contaminated by the live memory store

`run_task` (`benchmark/runner.py`) deep-copies the config but leaves
`memory.enabled` and `memory.store_path` untouched, so every benchmark run both
**reads (retrieve/inject) and writes** into the same shared `memory/` store.
This means:

- **Confounded A/B:** the active version runs first and writes episodes; the
  candidate then runs the identical tasks and retrieves what the active run just
  wrote — they are not evaluated under identical state.
- **Train/test leakage:** the held-out benchmark tasks land permanently in the
  experience store that live runs retrieve from and that `diagnose()` reads next
  cycle — the agent ends up "learning from" the tasks it is graded on.

  *Direction:* run the benchmark/evaluation against an **isolated, disabled, or
  frozen-snapshot** memory store (e.g. force `memory.enabled=False` or a per-eval
  temp `store_path`); if A/B-ing *with* memory, snapshot once and restore between
  active and candidate. Benchmark tasks must never enter the learning store.

### 3. Full system-prompt replacement is the least-guarded self-modification path

The structural whitelist makes sandbox/timeouts/tool-code unreachable, but
`prompt_text` can be **wholesale replaced**, validated only as a non-empty string
under 20k chars (`policy/patch.py`). The system prompt is where the agent's
behavioral and safety contract lives, yet a candidate that silently drops a
safety rule or rewrites the whole prompt is gated only by the weak, contaminated
signal above — and can then be auto-committed to git. "Prefer minimal edits" is
advisory proposer text, not enforced.

  *Direction:* treat prompt edits as privileged — bound the **diff size** (reject
  wholesale rewrites), assert a set of **pinned invariant clauses** (safety rules,
  the `task_done`/test-pass contract) survive every patch, require the stronger
  held-out + repeated-trial evidence before an *auto-commit*, and keep a
  human-in-the-loop checkpoint (or auto-revert on next-cycle regression) for
  prompt changes specifically — even while numeric-knob changes adopt
  automatically.

## Install

```bash
pip install -r requirements.txt
```

Requires Python 3.10+.

## Quickstart

1. Point the harness at an OpenAI-compatible endpoint via environment variables
   (the sample `config.yaml` reads these):

   ```bash
   # OpenAI
   export OPENAI_API_KEY=sk-...
   export OPENAI_BASE_URL=https://api.openai.com/v1

   # ...or a local vLLM / SGLang server
   # export OPENAI_BASE_URL=http://localhost:8000/v1
   # export OPENAI_API_KEY=dummy
   ```

2. Run a task. The agent operates inside the `workspace/` directory:

   ```bash
   python -m coding_harness.cli "create hello.py that prints 'hi' and run it"
   ```

3. Useful overrides:

   ```bash
   python -m coding_harness.cli "refactor utils.py" \
       --model gpt-4o \
       --base-url http://localhost:8000/v1 \
       --workspace ./workspace \
       --max-steps 40 \
       --dry-run
   ```

The full event trace is written as JSON Lines to `runs/trace.jsonl`, with a
compact mirror printed to the console. Every record carries a `run_id` and an
elapsed-time `ts`, so a single run's events can be isolated even though runs
share the append-only file (e.g. `jq 'select(.run_id=="abc12345")'`). The run
id is also printed in the final summary. Event types:

- `run_start` — task, model, base URL, budgets, workspace, and tool list.
- `llm_request` / `assistant_message` — one request/response pair per step.
- `llm_retry` / `llm_error` — each backoff attempt, and the terminal failure
  after retries are exhausted.
- `tool_call` / `tool_result` — each dispatched tool and its observation
  (`is_error` flags recoverable tool failures).
- `error` — an uncaught crash (e.g. exhausted LLM retries), with a traceback;
  distinct from hitting a step/token budget.
- `loop_end` — final `status` (`done` | `answered` | `max_steps` |
  `max_tokens` | `error`), step count, tokens, and elapsed time.

## Configuration

Edit `config.yaml` (string values support `${ENV_VAR}` expansion):

```yaml
llm:
  base_url: ${OPENAI_BASE_URL}
  api_key: ${OPENAI_API_KEY}
  model: llama-3.3-70b-versatile
  temperature: 0.0
  max_tokens: 2048
  parallel_tool_calls: true
  request_timeout: 120.0
loop:
  max_steps: 30
  max_total_tokens: 200000
sandbox:
  workspace_root: workspace
  command_timeout_s: 60.0
  max_output_bytes: 16000
  dry_run: false
  deny_patterns:               # also includes a fork-bomb pattern by default
    - '\brm\s+-rf\s+/'
    - '\bmkfs\b'
    - '\bshutdown\b'
    - '\breboot\b'
logging:
  trace_file: runs/trace.jsonl
  console: true
memory:                        # experience layer (safe to leave on)
  enabled: true
  store_path: memory
  embedding_model: local       # or a remote model name, e.g. text-embedding-3-small
  # embedding_base_url / embedding_api_key default to the llm.* values above
  retrieve_k_episodes: 3
  retrieve_k_playbooks: 2
  inject_token_budget: 800
  distill_every_n_episodes: 25
  # similarity_floor: 0.15     # episodes below this cosine similarity are never injected
evolve:                        # prompt/policy self-tuning (off by default)
  enabled: false
  active_policy_version: ""    # empty -> use the registry's active pointer
  policy_dir: policy
  max_candidates_per_cycle: 4
  adopt_epsilon: 0.03          # require +3% benchmark success to adopt
  max_cost_regression: 0.10    # reject if >10% more tokens/steps
  cycle_budget_tokens: 2000000 # hard per-cycle token ceiling
tool_evolution:                # pluggable tools + tool self-extension
  enabled: false               # gates the self-extension loop; registry is always pluggable
  require_approval: true       # a new tool only activates by merging a PR (v1 default)
  plugins_dir: ""              # empty -> packaged coding_harness/tools/plugins
  staging_dir: ""              # quarantine for proposed-but-unvalidated tools
  cadence: manual              # manual | nightly | every_n_runs (advisory; no scheduler yet)
  max_new_tools_per_cycle: 2
  gap_evidence_threshold: 0.25 # a gap must recur in >=25% of recent runs to mint a tool
  run_benchmark_ab: true
  github_repo: ""              # owner/repo a tool-proposal PR is opened against
```

## Project layout

```
coding_harness/
  cli.py          # entry point
  config.py       # dataclass config + YAML/env loading
  agent.py        # the ReAct loop
  llm_client.py   # OpenAI-compatible wrapper with retry/backoff + usage
  logging_.py     # JSONL event trace + console mirror
  prompts.py      # system prompt
  sandbox.py      # WorkspaceJail + safe subprocess execution
  tools/          # base registry, ToolContext, PluginLoader
    files.py search.py exec.py control.py   # built-in tool implementations
    plugins/      # discovery root: one self-describing folder per tool/group
      files/ search/ exec/ control/         # built-ins (author:"builtin")
  evolve/
    memory/       # experience store, episodes, embeddings,
                  #   retrieval/injection, distiller (playbooks)
    benchmark/    # Evaluation Gate: held-out tasks/ + runner.py
    policy/       # versioned prompt/policy registry + whitelisted patch
    evolver/      # diagnose / propose / evaluate / decide / cycle
tests/            # pytest unit tests for the harness
config.yaml       # sample config
```

## Tests

```bash
pytest
```

The suite covers path-jail confinement, `edit_file` unique-match behavior,
subprocess timeout/truncation/deny-list, registry schema generation and
dispatch validation, and a fully mocked agent loop (no network) that terminates
on `task_done` and enforces step/token budgets.
