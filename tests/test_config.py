"""Tests for config loading and env-var expansion."""

from __future__ import annotations

from coding_harness.config import Config


def test_defaults_when_no_file(tmp_path):
    cfg = Config.load(tmp_path / "does_not_exist.yaml")
    assert cfg.llm.model  # default present
    assert cfg.loop.max_steps > 0
    assert cfg.sandbox.workspace_root


def test_env_expansion(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_KEY", "secret-123")
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("llm:\n  api_key: ${MY_KEY}\n  model: m1\n")
    cfg = Config.load(cfg_file)
    assert cfg.llm.api_key == "secret-123"
    assert cfg.llm.model == "m1"


def test_empty_env_falls_back_to_default(tmp_path):
    # ${UNSET_VAR} expands to "" and should not override the default base_url.
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("llm:\n  base_url: ${DEFINITELY_UNSET_VAR_XYZ}\n")
    cfg = Config.load(cfg_file)
    assert cfg.llm.base_url == "https://api.openai.com/v1"


def test_unknown_keys_ignored(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("llm:\n  model: m1\n  bogus_key: 99\n")
    cfg = Config.load(cfg_file)
    assert cfg.llm.model == "m1"
