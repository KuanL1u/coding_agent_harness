"""Tests for the policy registry + the whitelisted patch boundary."""

from __future__ import annotations

import pytest

from coding_harness.evolve.policy import (
    PatchValidationError,
    PromptPolicyPatch,
    PromptPolicyRegistry,
    materialize,
    seed_version,
)


def _seed():
    return seed_version(
        prompt_text="base prompt",
        max_steps=30,
        max_total_tokens=200_000,
        temperature=0.0,
        parallel_tool_calls=True,
    )


# -- patch validation (the safety boundary) ----------------------------------


def test_patch_rejects_non_whitelisted_field():
    with pytest.raises(PatchValidationError, match="non-whitelisted"):
        PromptPolicyPatch.parse(
            {
                "base_version": "p1",
                "rationale": "sneaky",
                "deny_patterns": [],  # safety surface — must be rejected
            }
        )


def test_patch_rejects_workspace_and_timeout_fields():
    for bad in ("workspace_root", "command_timeout_s", "dry_run"):
        with pytest.raises(PatchValidationError):
            PromptPolicyPatch.parse(
                {"base_version": "p1", "rationale": "x", bad: "anything"}
            )


def test_patch_requires_base_version_and_rationale():
    with pytest.raises(PatchValidationError):
        PromptPolicyPatch.parse({"rationale": "x", "max_steps": 40})
    with pytest.raises(PatchValidationError):
        PromptPolicyPatch.parse({"base_version": "p1", "max_steps": 40})


def test_patch_rejects_out_of_range_values():
    with pytest.raises(PatchValidationError, match="temperature"):
        PromptPolicyPatch.parse(
            {"base_version": "p1", "rationale": "x", "temperature": 9.0}
        )
    with pytest.raises(PatchValidationError, match="max_steps"):
        PromptPolicyPatch.parse(
            {"base_version": "p1", "rationale": "x", "max_steps": 0}
        )


def test_patch_rejects_no_op():
    with pytest.raises(PatchValidationError, match="no change"):
        PromptPolicyPatch.parse({"base_version": "p1", "rationale": "nothing changes"})


def test_patch_rejects_unknown_tool_override():
    with pytest.raises(PatchValidationError, match="unknown tool"):
        PromptPolicyPatch.parse(
            {
                "base_version": "p1",
                "rationale": "x",
                "tool_description_overrides": {"not_a_tool": "desc"},
            },
            known_tools={"run_tests", "task_done"},
        )


def test_patch_accepts_valid_and_materializes():
    patch = PromptPolicyPatch.parse(
        {
            "base_version": "p1",
            "rationale": "tighten task_done",
            "prompt_text": "new prompt",
            "tool_description_overrides": {"task_done": "stricter desc"},
            "max_steps": 40,
        },
        known_tools={"task_done"},
    )
    base = _seed()
    cand = materialize(patch, base, new_version_id="cand1")
    assert cand.version == "cand1"
    assert cand.parent == "p1"
    assert cand.prompt_text == "new prompt"
    assert cand.tool_descriptions["task_done"] == "stricter desc"
    assert cand.max_steps == 40
    # Untouched fields inherit from the base.
    assert cand.temperature == base.temperature
    assert cand.max_total_tokens == base.max_total_tokens


# -- registry -----------------------------------------------------------------


def test_registry_seed_and_active(tmp_path):
    reg = PromptPolicyRegistry(tmp_path / "policy")
    assert reg.get_active() is None
    seeded = reg.seed(_seed())
    assert reg.get_active().version == seeded.version == "p1"
    assert reg.list_versions() == ["p1"]
    assert reg.next_version_id() == "p2"


def test_registry_set_active_and_pin(tmp_path):
    reg = PromptPolicyRegistry(tmp_path / "policy")
    reg.seed(_seed())
    v2 = materialize(
        PromptPolicyPatch.parse({"base_version": "p1", "rationale": "x", "max_steps": 50}),
        reg.get("p1"),
        new_version_id="p2",
    )
    reg.save(v2)
    reg.set_active("p2")
    assert reg.get_active().version == "p2"
    # An explicit pin overrides the stored pointer.
    assert reg.get_active(pin="p1").version == "p1"


def test_registry_rejects_activating_unknown(tmp_path):
    reg = PromptPolicyRegistry(tmp_path / "policy")
    reg.seed(_seed())
    with pytest.raises(ValueError, match="unknown version"):
        reg.set_active("p99")


def test_registry_archive(tmp_path):
    reg = PromptPolicyRegistry(tmp_path / "policy")
    cand = materialize(
        PromptPolicyPatch.parse({"base_version": "p1", "rationale": "x", "max_steps": 50}),
        _seed(),
        new_version_id="cand1",
    )
    path = reg.archive(cand, "did not beat active")
    assert path.is_file()
    import json

    payload = json.loads(path.read_text())
    assert payload["archived_reason"] == "did not beat active"
