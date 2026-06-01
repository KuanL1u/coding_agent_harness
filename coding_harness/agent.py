"""The ReAct agent loop.

The :class:`Agent` wires together the LLM client, the tool registry, and the
event logger, then drives a reason -> act -> observe loop using native
OpenAI tool calling:

1. Send the running message list (system + task + history) to the model.
2. Append the assistant message. If it has no tool calls, the model answered in
   prose and the loop ends.
3. Otherwise dispatch every requested tool call, appending one ``tool``-role
   result message per call (keyed by ``tool_call_id``) before the next request.
4. Stop when the model calls ``task_done``, or when the step / token budget is
   exhausted.

Tool errors are returned to the model as observations (never raised), so the
agent can self-correct instead of crashing.
"""

from __future__ import annotations

import hashlib
import traceback
from dataclasses import dataclass
from typing import Any

from .config import Config
from .evolve.memory import Memory
from .evolve.policy import (
    PromptPolicyRegistry,
    PromptPolicyVersion,
    seed_version,
)
from .llm_client import LLMClient
from .logging_ import EventLogger
from .prompts import SYSTEM_PROMPT
from .sandbox import WorkspaceJail
from .tools import (
    PluginLoader,
    TaskState,
    ToolRegistry,
    default_plugins_dir,
    make_tool_context,
)


@dataclass
class RunResult:
    """Outcome of an agent run."""

    status: str  # "done" | "max_steps" | "max_tokens" | "answered" | "error"
    summary: str
    steps: int
    total_tokens: int
    messages: list[dict[str, Any]]
    run_id: str = ""
    wall_clock_s: float = 0.0


def _config_snapshot(
    config: Config, system_prompt: str, prompt_version: str | None
) -> dict[str, Any]:
    """The prompt/config identity stamped on each episode.

    ``prompt_version`` is the active policy version id when prompt/policy
    self-tuning is driving, otherwise a short content hash of the system prompt
    so outcomes can still be attributed to the exact prompt that produced them.
    """
    if not prompt_version:
        prompt_hash = hashlib.blake2b(
            system_prompt.encode("utf-8"), digest_size=4
        ).hexdigest()
        prompt_version = f"p_{prompt_hash}"
    return {
        "prompt_version": prompt_version,
        "max_steps": config.loop.max_steps,
        "max_total_tokens": config.loop.max_total_tokens,
        "temperature": config.llm.temperature,
        "model": config.llm.model,
    }


def _resolve_policy(config: Config) -> PromptPolicyVersion | None:
    """Load the active prompt/policy version when self-tuning is enabled.

    Bootstraps an empty registry with a ``p1`` seed built from the current code
    defaults (so the first enabled run reproduces v1 behaviour exactly). Returns
    ``None`` when evolution is disabled, leaving the v1 hardcoded path intact.
    """
    evolve = getattr(config, "evolve", None)
    if evolve is None or not evolve.enabled:
        return None
    registry = PromptPolicyRegistry(evolve.policy_dir)
    active = registry.get_active(evolve.active_policy_version)
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


def _apply_policy(config: Config, policy: PromptPolicyVersion) -> None:
    """Fold a policy version's tunable knobs into ``config`` (mutates in place).

    Only the whitelisted, non-safety knobs are touched; the prompt and tool
    descriptions are applied separately by the caller.
    """
    config.loop.max_steps = policy.max_steps
    config.loop.max_total_tokens = policy.max_total_tokens
    config.llm.temperature = policy.temperature
    config.llm.parallel_tool_calls = policy.parallel_tool_calls


def _apply_tool_descriptions(registry: ToolRegistry, overrides: dict[str, str]) -> None:
    """Override registered tools' ``description`` fields (never their code)."""
    for name, description in (overrides or {}).items():
        tool = registry.get(name)
        if tool is not None:
            tool.description = description


def _plugins_dir(config: Config) -> str:
    """The plugins directory to discover tools from.

    A configured ``tool_evolution.plugins_dir`` wins; an empty value falls back
    to the packaged built-in plugins so the harness works from any cwd.
    """
    configured = getattr(getattr(config, "tool_evolution", None), "plugins_dir", "")
    return configured or str(default_plugins_dir())


def build_registry(
    config: Config,
    jail: WorkspaceJail,
    state: TaskState,
    *,
    on_event: Any = None,
) -> ToolRegistry:
    """Discover and register the active tool plugins into a registry.

    Tools are loaded from ``tool_evolution.plugins_dir`` (defaulting to the
    packaged built-ins) via :class:`PluginLoader`. The built-in plugins pin a
    load priority so the tool list keeps its v1 order, so this is behaviourally
    identical to the old hardcoded assembly — just directory-discovered and
    status-filtered. ``on_event`` (optional) routes plugin rejections/skips into
    the run trace; a clean built-in load emits nothing.
    """
    ctx = make_tool_context(config, jail, state)
    loader = PluginLoader(_plugins_dir(config), ctx, on_event=on_event)
    return loader.load()


class Agent:
    """Drives the ReAct loop for a single task."""

    def __init__(
        self,
        config: Config,
        llm: LLMClient,
        logger: EventLogger,
        registry: ToolRegistry,
        state: TaskState,
        memory: Memory | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        policy_version: str | None = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.logger = logger
        self.registry = registry
        self.state = state
        self.memory = memory
        self.system_prompt = system_prompt
        self.policy_version = policy_version

    @classmethod
    def create(
        cls,
        config: Config,
        llm: LLMClient | None = None,
        policy: PromptPolicyVersion | None = None,
    ) -> "Agent":
        """Build an agent and all its collaborators from config.

        ``policy`` is the active prompt/policy version. When omitted it
        is loaded from the registry if ``evolve.enabled``; otherwise the harness
        runs on the hardcoded v1 prompt and static config. An explicit ``policy``
        always wins — the evolver's gate uses it to A/B a candidate version.
        """
        if policy is None:
            policy = _resolve_policy(config)

        system_prompt = SYSTEM_PROMPT
        if policy is not None:
            system_prompt = policy.prompt_text or SYSTEM_PROMPT
            _apply_policy(config, policy)

        jail = WorkspaceJail(config.sandbox.workspace_root)
        state = TaskState()

        # The logger is created before the registry so plugin discovery can route
        # any rejection/skip into the same run trace as everything else.
        logger = EventLogger(config.logging.trace_file, console=config.logging.console)
        registry = build_registry(
            config, jail, state, on_event=lambda event, fields: logger.log(event, **fields)
        )
        if policy is not None and policy.tool_descriptions:
            _apply_tool_descriptions(registry, policy.tool_descriptions)

        llm = llm or LLMClient.from_config(config.llm)
        # Route the client's retry/error events into the same trace as the loop.
        if isinstance(llm, LLMClient):
            llm.event_hook = lambda event, fields: logger.log(event, **fields)
        # The memory layer is config-gated; ``None`` when disabled.
        # Its events flow into the same trace as everything else.
        memory = Memory.from_config(
            config, on_event=lambda event, fields: logger.log(event, **fields)
        )
        return cls(
            config,
            llm,
            logger,
            registry,
            state,
            memory,
            system_prompt=system_prompt,
            policy_version=policy.version if policy is not None else None,
        )

    def run(self, task: str) -> RunResult:
        """Execute ``task`` to completion or until a budget is hit."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]

        # Memory, pre-loop: inject the most relevant past experience as a
        # leading system message. ``task_embedding`` is reused for the episode
        # record so the task is only embedded once.
        task_embedding: list[float] = []
        if self.memory is not None:
            experience, task_embedding = self.memory.retrieve(task)
            if experience:
                messages.insert(1, {"role": "system", "content": experience})

        tools = self.registry.schemas()

        self.logger.log(
            "run_start",
            run_id=self.logger.run_id,
            task=task,
            model=self.config.llm.model,
            base_url=self.config.llm.base_url,
            max_steps=self.config.loop.max_steps,
            max_total_tokens=self.config.loop.max_total_tokens,
            workspace=self.config.sandbox.workspace_root,
            dry_run=self.config.sandbox.dry_run,
            tools=self.registry.names(),
            policy_version=self.policy_version,
        )

        status = "max_steps"
        try:
            for _ in range(self.config.loop.max_steps):
                step = self.logger.record_step()

                self.logger.log("llm_request", step=step, num_messages=len(messages))
                assistant = self.llm.complete(messages, tools=tools, tool_choice="auto")
                messages.append(_assistant_to_dict(assistant))

                self.logger.set_total_tokens(self.llm.usage.total_tokens)
                tool_calls = getattr(assistant, "tool_calls", None) or []
                self.logger.log(
                    "assistant_message",
                    step=step,
                    content=getattr(assistant, "content", None),
                    tool_calls=[_call_summary(c) for c in tool_calls],
                )

                if not tool_calls:
                    # Model responded in prose with no action: treat as a final answer.
                    status = "answered"
                    self.state.summary = getattr(assistant, "content", "") or ""
                    break

                for call in tool_calls:
                    self.logger.log("tool_call", step=step, **_call_summary(call))
                    result_msg = self.registry.dispatch(call)
                    messages.append(result_msg)
                    self.logger.log(
                        "tool_result",
                        step=step,
                        tool_call_id=result_msg["tool_call_id"],
                        is_error=result_msg["content"].startswith("ERROR: "),
                        content=result_msg["content"],
                    )

                if self.state.done:
                    status = "done"
                    break

                if self.llm.usage.total_tokens >= self.config.loop.max_total_tokens:
                    status = "max_tokens"
                    break
        except Exception as exc:
            # Anything that escapes the loop (e.g. exhausted LLM retries) is a
            # crash, not a budget exhaustion. Record it distinctly with a
            # traceback so the trace never mislabels it as ``max_steps``.
            status = "error"
            self.logger.log(
                "error",
                error_type=type(exc).__name__,
                message=str(exc),
                traceback=traceback.format_exc(),
            )
            raise
        finally:
            wall_clock_s = self.logger.elapsed()
            self.logger.log(
                "loop_end",
                status=status,
                steps=self.logger.steps,
                total_tokens=self.llm.usage.total_tokens,
                elapsed_s=wall_clock_s,
            )
            # Memory, post-loop: persist the episode (and distill on cadence).
            # Runs even on a crashing/budget-stopped path so the failure is
            # recorded; the call is internally failure-isolated.
            if self.memory is not None:
                self.memory.record(
                    task,
                    messages,
                    status=status,
                    steps=self.logger.steps,
                    tokens=self.llm.usage.total_tokens,
                    wall_clock_s=wall_clock_s,
                    trace_path=str(self.logger.trace_path),
                    config_snapshot=_config_snapshot(
                        self.config, self.system_prompt, self.policy_version
                    ),
                    task_embedding=task_embedding,
                )
            self.logger.close()

        return RunResult(
            status=status,
            summary=self.state.summary,
            steps=self.logger.steps,
            total_tokens=self.llm.usage.total_tokens,
            messages=messages,
            run_id=self.logger.run_id,
            wall_clock_s=wall_clock_s,
        )


def _assistant_to_dict(assistant: Any) -> dict[str, Any]:
    """Convert an assistant message object into the dict form for the next request.

    Preserves ``tool_calls`` exactly so the follow-up ``tool`` messages can be
    matched by id.
    """
    msg: dict[str, Any] = {
        "role": "assistant",
        "content": getattr(assistant, "content", None) or "",
    }
    tool_calls = getattr(assistant, "tool_calls", None)
    if tool_calls:
        msg["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {
                    "name": c.function.name,
                    "arguments": c.function.arguments,
                },
            }
            for c in tool_calls
        ]
    return msg


def _call_summary(call: Any) -> dict[str, str]:
    """Compact (name, arguments) view of a tool call for logging."""
    fn = getattr(call, "function", None) or {}
    name = getattr(fn, "name", "") if not isinstance(fn, dict) else fn.get("name", "")
    args = (
        getattr(fn, "arguments", "")
        if not isinstance(fn, dict)
        else fn.get("arguments", "")
    )
    return {"name": name, "arguments": args or ""}
