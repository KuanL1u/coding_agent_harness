"""Tests for the memory components (no network; local embedder)."""

from __future__ import annotations

import json

from coding_harness.evolve.memory.embeddings import (
    LocalHashEmbeddings,
    build_embedding_client,
    cosine,
)
from coding_harness.evolve.memory.episode import build_episode
from coding_harness.evolve.memory.store import LocalFileStore, Playbook
from coding_harness.evolve.memory.retrieval import render_experience_block
from coding_harness.evolve.memory.distiller import distill
from coding_harness.config import MemoryConfig, LLMConfig


# -- embeddings ---------------------------------------------------------------


def test_local_embeddings_deterministic_and_normalized():
    emb = LocalHashEmbeddings()
    a1 = emb.embed(["fix the failing pytest in calc.py"])[0]
    a2 = emb.embed(["fix the failing pytest in calc.py"])[0]
    assert a1 == a2  # deterministic
    assert abs(sum(x * x for x in a1) ** 0.5 - 1.0) < 1e-9  # unit norm


def test_local_embeddings_similarity_orders_by_overlap():
    emb = LocalHashEmbeddings()
    base = emb.embed(["fix the failing test in calc.py"])[0]
    near = emb.embed(["fix a failing test in calculator.py"])[0]
    far = emb.embed(["write a haiku about the ocean"])[0]
    assert cosine(base, near) > cosine(base, far)


def test_build_embedding_client_defaults_to_local():
    client = build_embedding_client(MemoryConfig(embedding_model="local"), LLMConfig())
    assert isinstance(client, LocalHashEmbeddings)
    client2 = build_embedding_client(MemoryConfig(embedding_model=""), LLMConfig())
    assert isinstance(client2, LocalHashEmbeddings)


# -- episode builder ----------------------------------------------------------


def _assistant(tool_calls):
    return {"role": "assistant", "content": "", "tool_calls": tool_calls}


def _call(cid, name, **args):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def test_build_episode_derives_fields():
    messages = [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "fix it"},
        _assistant([_call("c1", "read_file", path="calc.py")]),
        {"role": "tool", "tool_call_id": "c1", "content": "calc.py ..."},
        _assistant([_call("c2", "edit_file", path="calc.py", old_string="a", new_string="b")]),
        {"role": "tool", "tool_call_id": "c2", "content": "replaced 1 occurrence(s)"},
        _assistant([_call("c3", "run_tests")]),
        {"role": "tool", "tool_call_id": "c3", "content": "exit_code: 0"},
        _assistant([_call("c4", "task_done", summary="fixed")]),
        {"role": "tool", "tool_call_id": "c4", "content": "Task marked complete"},
    ]
    ep = build_episode(
        "fix it", messages, status="done", steps=4, tokens=123, wall_clock_s=1.5,
        trace_path="runs/t.jsonl", config_snapshot={"prompt_version": "p_x"},
        task_embedding=[0.1, 0.2],
    )
    assert ep.outcome == "success"
    assert ep.tools_used == {"read_file": 1, "edit_file": 1, "run_tests": 1, "task_done": 1}
    assert "edited calc.py" in ep.final_diff_summary
    assert ep.test_result == "passed"
    assert ep.failure_signature is None
    assert ep.task_embedding == [0.1, 0.2]


def test_build_episode_failure_signature_from_test_error():
    messages = [
        _assistant([_call("c1", "run_tests")]),
        {"role": "tool", "tool_call_id": "c1",
         "content": "ERROR: exit_code: 1\nE   AssertionError: assert 1 == 2"},
    ]
    ep = build_episode(
        "x", messages, status="max_steps", steps=1, tokens=1, wall_clock_s=0.1,
        trace_path="t", config_snapshot={},
    )
    assert ep.outcome == "stopped_budget"
    assert ep.test_result == "failed"
    assert ep.failure_signature and "AssertionError" in ep.failure_signature


# -- store --------------------------------------------------------------------


def _episode(task, emb, outcome="success", sig=None):
    return build_episode(
        task, [], status="done" if outcome == "success" else "error",
        steps=1, tokens=1, wall_clock_s=0.1, trace_path="t",
        config_snapshot={}, task_embedding=emb,
    )


def test_store_roundtrip_and_query_similar(tmp_path):
    store = LocalFileStore(tmp_path / "mem")
    emb = LocalHashEmbeddings()
    e_calc = build_episode(
        "fix failing test in calc.py", [], status="done", steps=3, tokens=10,
        wall_clock_s=1.0, trace_path="t", config_snapshot={},
        task_embedding=emb.embed(["fix failing test in calc.py"])[0],
    )
    e_haiku = build_episode(
        "write a poem", [], status="done", steps=2, tokens=5, wall_clock_s=0.5,
        trace_path="t", config_snapshot={},
        task_embedding=emb.embed(["write a poem"])[0],
    )
    store.add_episode(e_calc)
    store.add_episode(e_haiku)
    assert store.count_episodes() == 2

    q = emb.embed(["fix the broken test in calculator"])[0]
    results = store.query_similar(q, k=1)
    assert len(results) == 1
    assert results[0].item.task == "fix failing test in calc.py"


def test_store_failure_stats(tmp_path):
    store = LocalFileStore(tmp_path / "mem")
    emb = LocalHashEmbeddings()
    for _ in range(3):
        ep = build_episode(
            "t", [
                {"role": "tool", "tool_call_id": "x",
                 "content": "ERROR: E   AssertionError: nope"},
            ],
            status="max_steps", steps=1, tokens=1, wall_clock_s=0.1,
            trace_path="t", config_snapshot={}, task_embedding=emb.embed(["t"])[0],
        )
        # run_tests not in tools_used here, so test_result is n/a but signature
        # still derives from the error observation.
        store.add_episode(ep)
    ok = build_episode("ok", [], status="done", steps=1, tokens=1, wall_clock_s=0.1,
                       trace_path="t", config_snapshot={}, task_embedding=emb.embed(["ok"])[0])
    store.add_episode(ok)

    stats = store.failure_stats()
    assert stats["total_episodes"] == 4
    assert stats["total_failures"] == 3
    assert any("AssertionError" in sig for sig in stats["signatures"])


# -- retrieval ----------------------------------------------------------------


def test_render_experience_block_empty_when_no_match(tmp_path):
    store = LocalFileStore(tmp_path / "mem")
    cfg = MemoryConfig()
    block = render_experience_block(store, [0.0] * 256, cfg)
    assert block == ""


def test_render_experience_block_includes_similar_and_respects_budget(tmp_path):
    store = LocalFileStore(tmp_path / "mem")
    emb = LocalHashEmbeddings()
    store.add_episode(build_episode(
        "fix the failing test in calc.py", [], status="done", steps=3, tokens=10,
        wall_clock_s=1.0, trace_path="t", config_snapshot={},
        task_embedding=emb.embed(["fix the failing test in calc.py"])[0],
    ))
    cfg = MemoryConfig(inject_token_budget=800, similarity_floor=0.0)
    q = emb.embed(["fix failing test in calc.py"])[0]
    block = render_experience_block(store, q, cfg)
    assert "Relevant experience" in block
    assert "calc.py" in block

    tiny = MemoryConfig(inject_token_budget=10, similarity_floor=0.0)
    small_block = render_experience_block(store, q, tiny)
    assert len(small_block) <= len(block)


# -- distiller ----------------------------------------------------------------


def test_distill_creates_playbook_from_success_cluster(tmp_path):
    store = LocalFileStore(tmp_path / "mem")
    emb = LocalHashEmbeddings()
    for i in range(3):
        store.add_episode(build_episode(
            f"fix the failing pytest in module_{i}.py",
            [
                _assistant([_call("c1", "read_file", path="m.py")]),
                _assistant([_call("c2", "edit_file", path="m.py", old_string="a", new_string="b")]),
                _assistant([_call("c3", "run_tests")]),
                {"role": "tool", "tool_call_id": "c3", "content": "exit_code: 0"},
            ],
            status="done", steps=3, tokens=10, wall_clock_s=1.0, trace_path="t",
            config_snapshot={},
            task_embedding=emb.embed([f"fix the failing pytest in module_{i}.py"])[0],
        ))
    written = distill(store, min_cluster_size=2)
    assert written, "expected at least one playbook from the success cluster"
    pb = written[0]
    assert pb.evidence["success_rate"] == 1.0
    assert any("run_tests" in s for s in pb.steps)
    # Re-distilling bumps the version in place rather than duplicating.
    again = distill(store, min_cluster_size=2)
    assert again[0].version == pb.version + 1
