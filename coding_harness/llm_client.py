"""Thin wrapper over an OpenAI-compatible Chat Completions API.

Works against OpenAI itself or any compatible server (vLLM, SGLang, ...) via a
configurable ``base_url``. Adds retry-with-exponential-backoff on rate-limit and
server errors, and accumulates token usage across calls.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from openai import APIConnectionError, APIStatusError, RateLimitError
from openai import OpenAI


class EmptyResponseError(RuntimeError):
    """Raised when the server returns a 200 with no usable choices.

    Treated as a transient/retryable condition: some OpenAI-compatible servers
    occasionally return an empty ``choices`` list, and a retry usually succeeds.
    """


@dataclass
class Usage:
    """Cumulative token accounting across all requests."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    requests: int = 0

    def add(self, prompt: int, completion: int, total: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += total
        self.requests += 1


@dataclass
class LLMClient:
    """Configured client for one model/endpoint.

    Construct via :meth:`from_config` in normal use; the explicit fields exist so
    tests can build one (or substitute a fake) directly.
    """

    model: str
    temperature: float = 0.0
    max_tokens: int = 2048
    parallel_tool_calls: bool = True
    request_timeout: float = 120.0
    max_retries: int = 5
    base_backoff: float = 1.0
    _client: Any = None
    usage: Usage = field(default_factory=Usage)
    # Optional sink for observability events (e.g. EventLogger.log). Called as
    # ``event_hook(event_type, fields_dict)``; left unset in tests.
    event_hook: Callable[[str, dict[str, Any]], None] | None = None

    def _emit(self, event: str, **fields: Any) -> None:
        if self.event_hook is not None:
            self.event_hook(event, fields)

    @classmethod
    def from_config(cls, cfg: Any) -> "LLMClient":
        client = OpenAI(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            timeout=cfg.request_timeout,
        )
        return cls(
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            parallel_tool_calls=cfg.parallel_tool_calls,
            request_timeout=cfg.request_timeout,
            _client=client,
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
    ) -> Any:
        """Request one assistant message, retrying transient failures.

        Returns the assistant message object from the first choice. Usage is
        accumulated on ``self.usage``.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
            # Only forward parallel_tool_calls when explicitly enabled. Many
            # OpenAI-compatible servers (Ollama, vLLM, SGLang) reject the param
            # outright — even with value false — so when it is disabled we omit
            # it entirely and let the server fall back to its own default.
            if self.parallel_tool_calls:
                kwargs["parallel_tool_calls"] = True

        response = self._request_with_retry(kwargs)

        usage = getattr(response, "usage", None)
        if usage is not None:
            self.usage.add(
                getattr(usage, "prompt_tokens", 0) or 0,
                getattr(usage, "completion_tokens", 0) or 0,
                getattr(usage, "total_tokens", 0) or 0,
            )
        return response.choices[0].message

    def _request_with_retry(self, kwargs: dict[str, Any]) -> Any:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(**kwargs)
                if not getattr(response, "choices", None):
                    # 200 with an empty choices list: retry rather than letting
                    # a downstream ``choices[0]`` raise an opaque IndexError.
                    raise EmptyResponseError("LLM response contained no choices")
                return response
            except (RateLimitError, APIConnectionError, EmptyResponseError) as exc:
                last_exc = exc
            except APIStatusError as exc:
                # Retry only on server-side (5xx) errors; client errors are fatal.
                if exc.status_code < 500:
                    raise
                last_exc = exc

            if attempt == self.max_retries:
                break
            # Exponential backoff with full jitter.
            sleep = self.base_backoff * (2 ** attempt)
            sleep = random.uniform(0, sleep)
            self._emit(
                "llm_retry",
                attempt=attempt + 1,
                max_retries=self.max_retries,
                error_type=type(last_exc).__name__,
                error=str(last_exc),
                sleep_s=round(sleep, 3),
            )
            time.sleep(sleep)

        assert last_exc is not None
        # Retries exhausted: record the terminal failure before propagating so
        # the trace shows why the run died rather than only a raw traceback.
        self._emit(
            "llm_error",
            error_type=type(last_exc).__name__,
            error=str(last_exc),
            attempts=self.max_retries + 1,
        )
        raise last_exc
