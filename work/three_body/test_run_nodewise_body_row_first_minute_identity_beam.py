"""Isolated wrapper checks: bound three-field export, one call and no retry."""

from pathlib import Path

import numpy as np
import pytest
import run_nodewise_body_row_first_minute_identity_beam as m

NP_LOAD = np.load


@pytest.fixture(autouse=True)
def isolation(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "WORK", tmp_path)
    monkeypatch.setattr(m, "REPO", tmp_path)
    monkeypatch.setattr(m, "STOP", m.START + 4)
    monkeypatch.setattr(m.common, "WORK", tmp_path)
    monkeypatch.setattr(m.common, "REPO", tmp_path)
    monkeypatch.setattr(m.proof, "REPO", tmp_path)
    monkeypatch.setattr(m, "source_code", lambda evidence: {"synthetic_source_boundary": True})
    monkeypatch.setattr(m, "qualification", lambda evidence: {"existing_result_reused": True})
    monkeypatch.setattr(m.beam, "verify_sources", lambda: None)

    def forbidden(*args, **kwargs):
        pytest.fail("old workflow or actual data reached")

    for name in (
        "run",
        "source_code",
        "qualification",
        "comparison_headers",
        "load_candidate",
        "solve_once",
        "checked_ledger",
        "resource_report",
    ):
        monkeypatch.setattr(m.frozen, name, forbidden)
    for name in ("run", "load_control", "compare_result"):
        monkeypatch.setattr(m.minute, name, forbidden)
    real_open, real_bytes, real_text = Path.open, Path.read_bytes, Path.read_text

    def guard(path):
        assert Path(path).resolve().is_relative_to(tmp_path), "non-synthetic file access"

    def open_local(path, *args, **kwargs):
        guard(path)
        return real_open(path, *args, **kwargs)

    def bytes_local(path):
        guard(path)
        return real_bytes(path)

    def text_local(path, *args, **kwargs):
        guard(path)
        return real_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_local)
    monkeypatch.setattr(Path, "read_bytes", bytes_local)
    monkeypatch.setattr(Path, "read_text", text_local)
    reads = []

    class Archive:
        def __init__(self, path, *args, **kwargs):
            guard(path)
            self.inner = NP_LOAD(path, *args, **kwargs)
            self.files = self.inner.files
            reads.append("OPEN")

        def __getitem__(self, key):
            reads.append(key)
            return self.inner[key]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.inner.close()

    monkeypatch.setattr(np, "load", Archive)
    return reads


def fixture_comparison(tmp_path, monkeypatch, damage=None):
    source = tmp_path / "compare_nodewise_body_row_first_minute.py"
    source.write_text("# synthetic reviewed comparison\n")
    monkeypatch.setattr(m, "COMPARISON_SHA", m.proof.sha(source))
    geometry = dict(
        frame_idx=np.arange(m.START, m.STOP, dtype=np.int64),
        pose_xyz=np.zeros((4, 3, 3, 3), np.float64),
        observed=np.ones((4, 3, 3), bool),
    )
    geometry["pose_xyz"][..., 0] = np.arange(3)[None, :, None] * 20
    geometry["pose_xyz"][..., 1] = np.arange(3)[None, None, :] * 2
    geometry["pose_xyz"] += np.arange(4)[:, None, None, None] * 0.1
    geometry["observed"][1, 1, 0] = False
    if damage == "first_support":
        geometry["observed"][0, 2, 0] = False
    signatures = m.common.signature(geometry)
    if damage == "science_signature":
        signatures["pose_xyz"]["sha256"] = "0" * 64
    if damage == "membership":
        geometry["unrelated"] = np.zeros(1)
    root = tmp_path / m.COMPARISON_STAGE
    out = m.proof.Output(root)
    out.write("science/nodewise.npz", geometry)
    signature_map = dict(tail=signatures.copy(), nodewise=signatures)
    out.write("science_signatures.json", signature_map)
    milestone_root = tmp_path / m.PRODUCER_STAGE / f"milestones/through_{m.STOP:06d}"
    milestone_root.mkdir(parents=True)
    candidate = dict(path=str(milestone_root / "candidate.npz"), sha256="a" * 64, size_bytes=123)
    # Original candidate and checkpoint payload files deliberately do not exist.
    declared = {str(i): {"synthetic_declaration_only": True} for i in range(21)}
    milestone = dict(
        schema=m.PRODUCER_STAGE + "_milestone",
        status="ok",
        horizon_stop=m.STOP,
        scientific_interval=[m.START, m.STOP],
        native_frames=4,
        initialize_count=1,
        restore_count=4,
        finish_calls_so_far=0,
        new_begin_calls=0,
        external_pose_resets=0,
        all39_roundtrip=True,
        prior_history_prefix_unchanged=True,
        whole_process_completion_pending=True,
        physical_ID_truth=False,
        candidate=candidate,
        candidate_arrays=declared,
        checkpoint={"manifest_sha256": m.CHECKPOINT_SHA},
    )
    milestone_path = milestone_root / "summary.json"
    milestone_path.write_bytes(m.proof.json_bytes(milestone))
    monkeypatch.setattr(m, "MILESTONE_SHA", m.proof.sha(milestone_path))
    scope = dict(
        schema=m.COMPARISON_STAGE,
        native_interval=[m.START, m.STOP],
        retained_interval=[m.FIRST, m.STOP],
        terminal_horizon=m.STOP,
        original_tracking_sha256=m.ORIGINAL_SHA,
        original_checkpoint_sha256=m.ORIGINAL_CHECKPOINT_SHA,
        same_original_metadata_and39_signatures=True,
        common_source_prefix_chunks=16,
        nodewise_source_sha256=m.PRODUCER_SHA,
        nodewise_milestone_summary_sha256=m.MILESTONE_SHA,
        nodewise_checkpoint_manifest_sha256=m.CHECKPOINT_SHA,
        whole_five_minute_completion_claimed=False,
        support_recomputed=False,
        review_cleared=False,
        all21_membership_verified=True,
        undecoded15_per_arm_numerically_verified=False,
        source_candidate_npz_decodes_per_arm=1,
        science_signatures=signature_map,
        nodewise_milestone_candidate=candidate,
        nodewise_all21_declared_signatures=declared,
    )
    if damage == "horizon":
        scope["terminal_horizon"] += 1
    if damage == "milestone_binding":
        scope["nodewise_milestone_summary_sha256"] = "b" * 64
    out.write("plan.json", scope)
    out.write("summary.json", dict(**scope, status="ok"))
    inputs = {
        str(source): m.COMPARISON_SHA,
        str(milestone_path): m.MILESTONE_SHA,
        candidate["path"]: candidate["sha256"],
        str(tmp_path / "run_nodewise_body_row_five_minutes.py"): m.PRODUCER_SHA,
    }
    if damage == "source_edge":
        inputs[str(source)] = "c" * 64
    seal_path = root / "comparison.provenance.json"
    seal_path.write_bytes(
        m.proof.json_bytes(
            dict(
                stage=m.COMPARISON_STAGE,
                status="ok",
                inputs=inputs,
                outputs=out.records,
            )
        )
    )
    return m.proof.sha(seal_path)


def test_one_call_three_fields_real_small_solver_ledger_and_writer(
    tmp_path, monkeypatch, isolation
):
    comparison_sha = fixture_comparison(tmp_path, monkeypatch)
    real = m.beam.solve_identity_node_memory_beam
    calls = []

    def counted(frames, unary, xyz, observed, **kwargs):
        calls.append(kwargs)
        assert frames.tolist() == list(range(m.START, m.STOP))
        assert unary.dtype == np.float64 and unary.shape == (4, 6) and not unary.any()
        assert xyz.shape == (4, 3, 2, 3) and observed.shape == (4, 3, 2)
        assert kwargs == dict(beam_width=256, node_indices=(1, 0), **m.CONFIG)
        return real(frames, unary, xyz, observed, **kwargs)

    monkeypatch.setattr(m.beam, "solve_identity_node_memory_beam", counted)
    root = tmp_path / "result"
    assert m.run(comparison_sha, root)["status"] == "ok"
    assert len(calls) == 1 and isolation == ["OPEN", *m.FIELDS]
    summary = m.proof.read_json(root / "summary.json")
    assert summary["solve_calls"] == 1 and summary["science_archive_decodes"] == 1
    assert summary["native_interval"] == [m.START, m.STOP]
    assert summary["initial_P"] == [0, 1, 2] and not summary["physical_ID_truth"]
    assert summary["whole_five_minute_completion_claimed"] is False
    checked = m.proof.read_json(root / "independent_path_ledger_check.json")
    assert abs(checked["signed_difference"]) <= checked["tolerance"]
    assert all(checked["terminal_arrays_exact"].values())
    seal = m.proof.read_json(root / "identity.provenance.json")
    assert seal["status"] == "ok"
    assert "identity_mapping_and_ambiguity.npz" in seal["outputs"]


@pytest.mark.parametrize(
    "damage",
    [
        "source_edge",
        "horizon",
        "milestone_binding",
        "first_support",
        "science_signature",
        "membership",
    ],
)
def test_bad_binding_or_science_refused_before_solve(tmp_path, monkeypatch, isolation, damage):
    digest = fixture_comparison(tmp_path, monkeypatch, damage)
    monkeypatch.setattr(m, "solve_once", lambda part: pytest.fail("invalid data reached solver"))
    root = tmp_path / "failed"
    with pytest.raises(ValueError):
        m.run(digest, root)
    seal = m.proof.read_json(root / "identity.provenance.json")
    assert seal["status"] == "failed" and seal["solve_calls"] == 0
    assert not (root / "identity_mapping_and_ambiguity.npz").exists()
    if damage in {"source_edge", "horizon", "milestone_binding"}:
        assert isolation == []


def test_capacity_failure_is_one_call_no_retry_no_partial_solution(tmp_path, monkeypatch):
    digest = fixture_comparison(tmp_path, monkeypatch)
    calls = []

    def capacity(*args, **kwargs):
        calls.append(kwargs)
        raise m.beam.IdentityBeamCapacityError(2, "total_states", 1_000_001, 1_000_000)

    monkeypatch.setattr(m.beam, "solve_identity_node_memory_beam", capacity)
    root = tmp_path / "capacity"
    with pytest.raises(m.beam.IdentityBeamCapacityError):
        m.run(digest, root)
    failure = m.proof.read_json(root / "failure.json")
    assert len(calls) == 1 and failure["solve_calls"] == 1
    assert failure["native_frame"] == m.START + 2 and not failure["automatic_retry"]
    assert not failure["cap_increase"] and failure["decoded_input_fields_unchanged"]
    assert not (root / "identity_mapping_and_ambiguity.npz").exists()
