"""Caller-only CPU fixtures and solver stubs; no actual recording or beam/exact solve."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import run_sparse_instance_five_minute_identity_beam as run
from test_compare_fast_tail_two_minutes import candidate, refresh_derived


@pytest.fixture(autouse=True)
def forbid_solver_and_gpu(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("only synthetic caller contracts; actual solver/GPU/video prohibited")

    monkeypatch.setattr(run.beam, "solve_identity_node_memory_beam", forbidden)
    torch = run.sys.modules.get("torch")
    if torch is not None:
        assert not torch.cuda.is_initialized()
        monkeypatch.setattr(torch.cuda, "is_available", forbidden)


@pytest.fixture
def scope(monkeypatch, tmp_path):
    for key, value in dict(
        FIRST=8, START=10, STOP=13, WORK=tmp_path, COMPARISON_ROOT=tmp_path / run.comparison.STAGE
    ).items():
        monkeypatch.setattr(run, key, value)
    return tmp_path


def test_disabled_before_creating_outputs_or_reading_any_input(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("disabled stage must not create output")

    monkeypatch.setattr(run.proof, "Output", forbidden)
    with pytest.raises(ValueError, match="not enabled"):
        run.run("0" * 64, tmp_path / "never")
    assert not (tmp_path / "never").exists() and run.EXECUTION_READY is False


@pytest.fixture
def saved_comparison(scope):
    raw, _ = candidate(run.FIRST, run.STOP)
    raw["identity_review"][:, 1] = True
    refresh_derived(raw)
    references, tracking, inputs = (
        {},
        {},
        {str(Path(run.comparison.__file__).resolve()): run.COMPARISON_SHA},
    )
    for arm in run.comparison.ARMS:
        spec = run.comparison.SPECS[arm]
        root = scope / spec["stage"]
        root.mkdir()
        final = str((root / "candidate.npz").resolve())
        milestone = str((root / f"milestones/through_{run.STOP}/candidate.npz").resolve())
        # Distinct archive digests, same numerical21 signature. No producer payload is written.
        references[arm] = dict(
            path=final,
            sha256="a" * 64,
            milestone_path=milestone,
            milestone_sha256="b" * 64,
            terminal_horizon=run.STOP,
            arrays=run.comparison.common.signature(raw),
        )
        producer = dict(
            stage=root.name,
            status="ok",
            inputs={str((scope / spec["runner"]).resolve()): spec["source_sha"]},
            outputs={
                "candidate.npz": dict(path=final, sha256="a" * 64, size_bytes=123),
                f"milestones/through_{run.STOP}/candidate.npz": dict(
                    path=milestone, sha256="b" * 64, size_bytes=456
                ),
            },
        )
        path = root / "tracking.provenance.json"
        path.write_bytes(run.proof.json_bytes(producer))
        digest = run.proof.sha(path)
        tracking[arm] = digest
        inputs.update({str(path.resolve()): digest, final: "a" * 64, milestone: "b" * 64})
    output = run.proof.Output(run.COMPARISON_ROOT)
    common = dict(
        schema=run.comparison.STAGE,
        native_interval=[run.START, run.STOP],
        native_frames=run.STOP - run.START,
        terminal_horizon_by_arm=dict.fromkeys(run.comparison.ARMS, run.STOP),
        candidate_references=references,
    )
    output.write("plan.json", dict(**common, tracking_seals=tracking))
    output.write(
        "summary.json",
        dict(
            **common,
            status="ok",
            review_modified=False,
            candidate_modified=False,
            human_truth_created=False,
            all_three_physical_ID_truth=False,
        ),
    )
    output.write(f"candidates/{run.ARM}.npz", raw)
    seal = dict(stage=run.comparison.STAGE, status="ok", inputs=inputs, outputs=output.records)
    path = run.COMPARISON_ROOT / "comparison.provenance.json"
    path.write_bytes(run.proof.json_bytes(seal))
    return raw, seal, path


def test_saved_copy_bound_by_all21_not_identical_archive_digest(saved_comparison):
    raw, _, path = saved_comparison
    evidence = run.proof.Evidence()
    seal, reference = run.comparison_headers(run.proof.sha(path), evidence)
    loaded, signatures, copy_ref = run.load_candidate(seal, reference, evidence)
    assert signatures == run.comparison.common.signature(raw) == reference["arrays"]
    assert copy_ref["sha256"] not in (reference["sha256"], reference["milestone_sha256"])
    assert len(loaded) == 21 and all(not a.flags.writeable for a in loaded.values())
    assert loaded["identity_review"][:, 1].all()
    part = run.scientific_geometry(loaded)
    assert part["frame_idx"].tolist() == [10, 11, 12]
    assert len(evidence.paths) == 6  # Comparison seal/2 docs/2 producer seals/copy only.
    evidence.final_guard()


@pytest.mark.parametrize(
    "damage",
    [
        "failed_comparison",
        "milestone_edge",
        "final_edge",
        "source_sha",
        "candidate_signature",
        "horizon",
    ],
)
def test_header_tampering_rejected_before_payload_decode(saved_comparison, damage, monkeypatch):
    _, seal, path = saved_comparison
    if damage == "failed_comparison":
        seal["status"] = "failed"
    elif damage in ("milestone_edge", "final_edge"):
        member = "milestone_path" if damage == "milestone_edge" else "path"
        summary = run.proof.read_json(run.COMPARISON_ROOT / "summary.json")
        seal["inputs"].pop(summary["candidate_references"][run.ARM][member])
    elif damage == "source_sha":
        seal["inputs"][str(Path(run.comparison.__file__).resolve())] = "0" * 64
    else:
        summary_path = run.COMPARISON_ROOT / "summary.json"
        summary = run.proof.read_json(summary_path)
        if damage == "horizon":
            summary["terminal_horizon_by_arm"][run.ARM] -= 1
        else:
            summary["candidate_references"][run.ARM]["arrays"]["pose_xyz"]["sha256"] = "0" * 64
        summary_path.write_bytes(run.proof.json_bytes(summary))
        seal["outputs"]["summary.json"].update(
            sha256=run.proof.sha(summary_path), size_bytes=summary_path.stat().st_size
        )
    path.write_bytes(run.proof.json_bytes(seal))
    monkeypatch.setattr(
        run.np, "load", lambda *a, **k: pytest.fail("payload opened before valid headers")
    )
    with pytest.raises(ValueError):
        run.comparison_headers(run.proof.sha(path), run.proof.Evidence())


def test_failed_final_producer_is_rejected_even_if_comparison_claims_success(saved_comparison):
    _, seal, path = saved_comparison
    producer_path = run.WORK / run.comparison.SPECS[run.ARM]["stage"] / "tracking.provenance.json"
    producer = run.proof.read_json(producer_path)
    producer["status"] = "failed"
    producer_path.write_bytes(run.proof.json_bytes(producer))
    new_sha = run.proof.sha(producer_path)
    plan_path = run.COMPARISON_ROOT / "plan.json"
    plan = run.proof.read_json(plan_path)
    plan["tracking_seals"][run.ARM] = new_sha
    plan_path.write_bytes(run.proof.json_bytes(plan))
    seal["inputs"][str(producer_path.resolve())] = new_sha
    seal["outputs"]["plan.json"].update(
        sha256=run.proof.sha(plan_path), size_bytes=plan_path.stat().st_size
    )
    path.write_bytes(run.proof.json_bytes(seal))
    with pytest.raises(ValueError, match="both final"):
        run.comparison_headers(run.proof.sha(path), run.proof.Evidence())


def test_first_direct_missing_never_shifts_initialization(scope):
    raw, _ = candidate(run.FIRST, run.STOP)
    raw["observed"][run.START - run.FIRST, 1, 0] = False
    with pytest.raises(ValueError, match="no shifted initialization"):
        run.scientific_geometry(raw)


def test_one_call_fixed_objective_full_interval_and_raw_arrays_unchanged(scope, monkeypatch):
    raw, _ = candidate(run.FIRST, run.STOP)
    before = run.comparison.common.signature(raw)
    part = run.scientific_geometry(raw)
    calls = []
    marker = object()

    def stub(frames, unary, xyz, direct, **kwargs):
        calls.append(kwargs)
        np.testing.assert_array_equal(frames, [10, 11, 12])
        assert unary.shape == (3, 6) and unary.dtype == np.float64 and not unary.any()
        np.testing.assert_array_equal(xyz, part["pose_xyz"][:, :, run.NODES, :])
        np.testing.assert_array_equal(direct, part["observed"][:, :, run.NODES])
        assert kwargs == dict(beam_width=256, node_indices=(1, 0), **run.CONFIG)
        return marker

    monkeypatch.setattr(run.beam, "solve_identity_node_memory_beam", stub)
    assert run.solve_once(part) is marker and len(calls) == 1
    assert run.comparison.common.signature(raw) == before


def synthetic_result(part):
    n = len(part["frame_idx"])
    assert n == 3
    mapping = np.tile([0, 1, 2], (n, 1))
    ledger = run.beam.frozen.node_memory_path_links(
        part["frame_idx"],
        part["pose_xyz"][:, :, run.NODES, :],
        part["observed"][:, :, run.NODES],
        mapping,
        node_indices=run.NODES,
        scale_mm=0.8,
        weight=0.2,
    )
    values = dict(
        schema=run.beam.SCHEMA,
        status="optimal_on_retained_graph",
        frame_idx=part["frame_idx"].copy(),
        permutations=run.beam.frozen.PERMUTATIONS.copy(),
        path_state=np.zeros(n, np.int64),
        permutation_by_id=mapping,
        path_node=np.array([0, 1, 7]),
        path_cost=ledger["motion_cost"],
        layer_offsets=np.array([0, 1, 7, 13]),
        state_permutation=np.r_[0, np.tile(np.arange(6), 2)],
        state_last_event=np.zeros((13, 3, 2), np.int64),
        edge_offsets=np.array([0, 6, 42]),
        edge_source=np.r_[np.zeros(6, np.int64), np.repeat(np.arange(1, 7), 6)],
        edge_target=np.r_[np.arange(1, 7), np.tile(np.arange(7, 13), 6)],
        edge_cost=np.zeros(42),
        forward_cost=np.zeros(13),
        backward_cost=np.zeros(13),
        conditional_min_marginal_cost=np.zeros((n, 6)),
        conditional_min_marginal_delta=np.zeros((n, 6)),
        conditional_optimal_state_mask=np.ones((n, 6), bool),
        conditional_id_slot_min_marginal_delta=np.zeros((n, 3, 3)),
        conditional_state_alternative_gap=np.zeros(n),
        conditional_identity_alternative_gap=np.zeros((n, 3)),
        conditional_state_alternative_status=np.full(n, "finite_gap", "U32"),
        conditional_identity_alternative_status=np.full((n, 3), "not_retained_unknown", "U32"),
        conditional_global_alternative_gap=0.0,
        conditional_global_alternative_status="finite_gap",
        **{key: ledger[key] for key in run.TERMINAL},
        expanded_state_counts=np.array([1, 6, 6]),
        retained_state_counts=np.array([1, 6, 6]),
        discarded_state_counts=np.zeros(n, np.int64),
        discarded_finite_state_counts=np.zeros(n, np.int64),
        expanded_edge_counts=np.array([0, 6, 36]),
        retained_edge_counts=np.array([0, 6, 36]),
        discarded_edge_counts=np.zeros(n, np.int64),
        numerical_tie_tolerance=run.CONFIG["tie_atol"],
        roundoff_clamped_deltas=0,
        common_cost_offset=0.0,
        numerical_cost_span=0.0,
        beam_width=256,
        max_layer_states=100000,
        max_total_states=1000000,
        max_total_edges=6000000,
        node_indices=run.NODES,
        scale_mm=0.8,
        weight=0.2,
        initial_state=0,
        pruning_occurred=True,
        finite_prefixes_discarded=True,
        approximation_used=True,
    )
    result = run.beam.IdentityNodeMemoryBeamResult(**values)
    for field in dataclasses.fields(result):
        value = getattr(result, field.name)
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    return result


def test_independent_ledger_and_result_codec_preserve_conditional_unknowns(scope):
    raw, _ = candidate(run.FIRST, run.STOP)
    part = run.scientific_geometry(raw)
    result = synthetic_result(part)
    ledger, report = run.checked_ledger(part, result)
    assert report["signed_difference"] == 0 and all(report["terminal_arrays_exact"].values())
    resources = run.resource_report(result)
    assert resources["retained_states"] == 13 and resources["generated_edges"] == 42
    out = run.proof.Output(scope / "synthetic_result_codec")
    run.write_result(out, result, ledger)
    with np.load(out.root / "identity_mapping_and_ambiguity.npz", allow_pickle=False) as archive:
        assert set(archive.files) == set(run.MAPPING_FIELDS)
        assert (archive["conditional_identity_alternative_status"] == "not_retained_unknown").all()
    assert not (out.root / "raw_candidate21.npz").exists()
    assert run.explicit_json(dict(gap=np.inf, unknown=np.nan)) == dict(
        gap="+Infinity", unknown="NaN"
    )


@pytest.mark.parametrize("damage", ["cost", "terminal", "mapping", "config", "global_certificate"])
def test_invalid_result_never_passes_path_qa(scope, damage):
    raw, _ = candidate(run.FIRST, run.STOP)
    part = run.scientific_geometry(raw)
    result = synthetic_result(part)
    if damage == "cost":
        result = dataclasses.replace(result, path_cost=1.0)
    elif damage == "terminal":
        result = dataclasses.replace(result, terminal_age_frames=result.terminal_age_frames + 1)
    elif damage == "mapping":
        result = dataclasses.replace(result, permutation_by_id=np.zeros((3, 3), np.int64))
    elif damage == "config":
        result = dataclasses.replace(result, weight=0.3)
    else:
        result = dataclasses.replace(result, global_optimality_certified=True)
    with pytest.raises(ValueError):
        run.checked_ledger(part, result)


def test_capacity_failure_writes_native_reason_once_without_partial_mapping(scope, monkeypatch):
    raw, _ = candidate(run.FIRST, run.STOP)
    signatures = run.comparison.common.signature(raw)
    monkeypatch.setattr(run, "EXECUTION_READY", True)  # Synthetic dependency stubs only.
    monkeypatch.setattr(run, "source_code", lambda: {})
    monkeypatch.setattr(run, "qualification", lambda evidence: {"synthetic": True})
    monkeypatch.setattr(run, "comparison_headers", lambda sha, evidence: ({}, {"synthetic": True}))
    monkeypatch.setattr(run, "load_candidate", lambda *args: (raw, signatures, {"synthetic": True}))
    calls = []

    def capacity(*args, **kwargs):
        calls.append(True)
        raise run.beam.IdentityBeamCapacityError(2, "total_states", 1000001, 1000000)

    monkeypatch.setattr(run.beam, "solve_identity_node_memory_beam", capacity)
    output = scope / "synthetic_capacity_failure"
    with pytest.raises(run.beam.IdentityBeamCapacityError):
        run.run("0" * 64, output)
    failure = json.loads((output / "failure.json").read_text())
    assert len(calls) == failure["solve_calls"] == 1
    assert (
        failure["native_frame"] == 12
        and failure["attempted"] == 1000001
        and failure["limit"] == 1000000
    )
    assert failure["raw21_unchanged_at_failure"]
    assert not failure["automatic_retry"] and not failure["partial_identity_solution_published"]
    seal = json.loads((output / "identity.provenance.json").read_text())
    assert seal["status"] == "failed" and "failure.json" in seal["outputs"]
    assert not (output / "identity_mapping_and_ambiguity.npz").exists()
    assert run.comparison.common.signature(raw) == signatures
