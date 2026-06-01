# coding_harness

A small, from-scratch **autonomous coding agent** built on the `openai` SDK and
the Python standard library — no LangChain, no heavy frameworks.

It runs a **ReAct loop** (reason → act → observe → repeat) using **native
OpenAI-compatible function/tool calling**, executes tools locally inside a
confined workspace, and writes a structured trace of everything it does.

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
  model: gpt-4o-mini
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
  deny_patterns: ['\brm\s+-rf\s+/', '\bmkfs\b']
logging:
  trace_file: runs/trace.jsonl
  console: true
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
  tools/          # base registry + file/search/exec/control tools
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
