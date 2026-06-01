"""Typed configuration with YAML + environment-variable loading.

Configuration is grouped into four sections (llm, loop, sandbox, logging), each
a dataclass with sensible defaults. ``${ENV_VAR}`` references in string values
are expanded from the environment, so secrets such as the API key never need to
live in the YAML file.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

_ENV_REF = re.compile(r"\$\{([^}]+)\}")


def _expand(value: Any) -> Any:
    """Recursively expand ${ENV_VAR} references in strings."""
    if isinstance(value, str):
        return _ENV_REF.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


@dataclass
class LLMConfig:
    base_url: str = "https://api.openai.com/v1"
    api_key: str = "${OPENAI_API_KEY}"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 2048
    parallel_tool_calls: bool = True
    request_timeout: float = 120.0


@dataclass
class LoopConfig:
    max_steps: int = 30
    max_total_tokens: int = 200_000


@dataclass
class SandboxConfig:
    workspace_root: str = "workspace"
    command_timeout_s: float = 60.0
    max_output_bytes: int = 16_000
    dry_run: bool = False
    deny_patterns: list[str] = field(
        default_factory=lambda: [
            r"\brm\s+-rf\s+/",      # recursive delete of root-ish paths
            r":\(\)\s*\{.*\};:",    # classic fork bomb
            r"\bmkfs\b",            # filesystem format
            r"\bshutdown\b",
            r"\breboot\b",
        ]
    )


@dataclass
class LoggingConfig:
    trace_file: str = "runs/trace.jsonl"
    console: bool = True


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        """Load config from a YAML file, applying env expansion and defaults.

        A missing file yields all-default config so the harness can run with
        nothing but environment variables set.
        """
        data: dict[str, Any] = {}
        if path is not None:
            p = Path(path)
            if p.is_file():
                raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                data = _expand(raw)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        return cls(
            llm=_build(LLMConfig, data.get("llm", {})),
            loop=_build(LoopConfig, data.get("loop", {})),
            sandbox=_build(SandboxConfig, data.get("sandbox", {})),
            logging=_build(LoggingConfig, data.get("logging", {})),
        )


def _build(dc_type: type, data: dict[str, Any]) -> Any:
    """Instantiate dataclass ``dc_type`` from ``data``, ignoring unknown keys.

    Keys whose value is an empty string (a common result of expanding an unset
    ``${ENV_VAR}``) are dropped so the dataclass default is used instead.
    """
    assert is_dataclass(dc_type)
    known = {f.name for f in fields(dc_type)}
    kwargs = {
        k: v
        for k, v in (data or {}).items()
        if k in known and not (isinstance(v, str) and v == "")
    }
    return dc_type(**kwargs)
