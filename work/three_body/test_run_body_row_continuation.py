"""Synthetic caller contracts only; no real source inputs, checkpoint, GPU or fitting."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import run_body_row_continuation as m


def forbidden(*args, **kwargs):
    pytest.fail("real tracking/input execution is forbidden in caller tests")


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    work = tmp_path / "repo/work/three_body"
    work.mkdir(parents=True)
    monkeypatch.setattr(m, "WORK", work)
    monkeypatch.setattr(m, "REPO", work.parents[1])
    monkeypatch.setattr(m, "PARENT", work / "synthetic_sparse_parent")
    monkeypatch.setattr(m.original, "verified_inputs", forbidden)
    monkeypatch.setattr(m.original, "SourceSequenceReader", forbidden)
    monkeypatch.setattr(m.prior, "read_checkpoint", forbidden)
    monkeypatch.setattr(m.prior, "runtime", forbidden)
    monkeypatch.setattr(m.prior, "local_code", forbidden)
    monkeypatch.setattr(m.prior, "smi_sample", forbidden)
    monkeypatch.setattr(m.sparse, "SparseInstanceRescoreContinuousTracker", forbidden)
    monkeypatch.setattr(m.variant, "BodyRowContinuousTracker", forbidden)
    monkeypatch.setattr(m.torch.cuda, "synchronize", forbidden)
    monkeypatch.setattr(m.torch.cuda, "set_per_process_memory_fraction", forbidden)
    real_load = np.load

    def confined_load(path, *args, **kwargs):
        assert Path(path).resolve().is_relative_to(tmp_path.resolve())
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", confined_load)
    return work


def body(frame, incoming=18, admitted=3, generated=8):
    return dict(
        schema="body_row_frame_diagnostics_v1",
        frame_idx=frame,
        frame_completed=True,
        incoming_proposal_count=incoming,
        outgoing_proposal_count=incoming + admitted,
        decoder_candidate_cap=24,
        headroom=24 - incoming,
        zero_headroom=incoming == 24,
        generated_scene_count=generated,
        added_proposal_count=admitted,
        excluded_by_capacity=generated - admitted,
        new_proposal_indices=np.arange(incoming, incoming + admitted, dtype=np.int64),
        new_proposal_kind="body_row",
        test_latent_values=np.array([1.0, np.nan, np.inf]),
    )


class Tracker:
    def __init__(self):
        self.initialize_count = 1
        self.restore_count = 7
        self.previous_frame = m.START - 1
        self.processed_frames = m.START - m.FIRST
        self.visited = []

    def advance(self, observation):
        self.visited.append(observation.frame_idx)
        self.previous_frame = observation.frame_idx
        self.processed_frames += 1


def test_serial_native_driver_keeps_one_tracker_and_restore_count(monkeypatch):
    tracker = Tracker()
    saved = []
    monkeypatch.setattr(
        m, "save_frame_diagnostics", lambda out, t, f: saved.append(f) or dict(frame_idx=f)
    )
    stop = m.START + 4
    rows, _ = m.drive(
        tracker,
        iter(SimpleNamespace(frame_idx=f) for f in range(m.START, stop)),
        None,
        synchronize=lambda: None,
        guard=lambda: None,
        stop=stop,
    )
    assert saved == tracker.visited == list(range(m.START, stop))
    assert [r["frame_idx"] for r in rows] == saved
    assert (tracker.initialize_count, tracker.restore_count) == (1, 7)
    m.lifecycle(tracker, stop)


def test_wrong_native_rejected_before_advance():
    tracker = Tracker()
    with pytest.raises(ValueError, match="skipped or reordered"):
        m.drive(
            tracker,
            iter([SimpleNamespace(frame_idx=m.START + 1)]),
            None,
            synchronize=forbidden,
            guard=lambda: None,
            stop=m.START + 1,
        )
    assert tracker.visited == []


@pytest.mark.parametrize(
    "field,value", [("restore_count", 8), ("initialize_count", 2), ("processed_frames", 1)]
)
def test_lifecycle_rejects_reset_or_extent_change(field, value):
    tracker = Tracker()
    setattr(tracker, field, value)
    with pytest.raises(ValueError, match="lifecycle"):
        m.lifecycle(tracker, m.START)


@pytest.mark.parametrize("incoming,admitted", [(18, 6), (24, 0), (20, 2)])
def test_admission_counts_and_zero_headroom(incoming, admitted):
    result = m.body_row_counts(body(m.START, incoming, admitted))
    assert result["original_count"] == incoming
    assert result["admitted"] == admitted
    assert result["headroom"] == 24 - incoming


def test_candidate_index_is_the_appended_subset():
    value = body(m.START)
    value["new_proposal_indices"] = np.array([0, 1, 2])
    with pytest.raises(ValueError, match="admission/cap/index"):
        m.body_row_counts(value)


def test_both_diagnostic_families_saved_with_exact_nonfinite_codec(isolate, monkeypatch):
    value = body(m.START)
    tracker = SimpleNamespace(export_body_row_diagnostics=lambda: copy.deepcopy(value))
    saved = []
    monkeypatch.setattr(
        m.diagnostic,
        "save_diagnostic",
        lambda out, t, f: saved.append(f) or dict(frame_idx=f, candidate_count=21),
    )
    out = m.proof.Output(isolate / "diagnostic_output")
    row = m.save_frame_diagnostics(out, tracker, m.START)
    doc = m.proof.read_json(out.root / f"body_row_diagnostics/f{m.START}.json")
    with np.load(doc["archive"]["path"], allow_pickle=False) as archive:
        restored = m.diagnostic.restore_diagnostic(
            doc["tree"], {k: archive[k] for k in archive.files}
        )
    assert restored["test_latent_values"].tobytes() == value["test_latent_values"].tobytes()
    assert row["body_row"]["admitted"] == 3 and saved == [m.START]
    assert value["frame_completed"] is True


@pytest.mark.parametrize(
    "mutate,match",
    [("incomplete", "completed native"), ("wrong_pool", "different candidate pools")],
)
def test_diagnostic_completion_and_same_pool_required(isolate, monkeypatch, mutate, match):
    value = body(m.START)
    if mutate == "incomplete":
        value["frame_completed"] = False
    tracker = SimpleNamespace(export_body_row_diagnostics=lambda: value)
    monkeypatch.setattr(
        m.diagnostic,
        "save_diagnostic",
        lambda *args: dict(frame_idx=m.START, candidate_count=20 if mutate == "wrong_pool" else 21),
    )
    with pytest.raises(ValueError, match=match):
        m.save_frame_diagnostics(m.proof.Output(isolate / "failed_diagnostic"), tracker, m.START)


def test_final_selection_count_excludes_inherited_history(monkeypatch):
    monkeypatch.setattr(m, "STOP", m.START + 3)
    rows = [
        dict(frame_idx=m.START + i, body_row=dict(admitted=n, headroom=h))
        for i, (n, h) in enumerate(((0, 0), (2, 4), (3, 6)))
    ]
    candidate = dict(
        frame_idx=np.arange(m.START - 1, m.STOP),
        proposal_kind=np.array(["body_row", "predict", "body_row", "body_row"]),
    )
    result = m.proposal_summary(rows, candidate)
    assert result["final_selected_body_row_frame_count"] == 2
    assert result["total_admitted"] == 5 and result["zero_headroom_frames"] == 1
    assert (result["admitted_min"], result["admitted_max"]) == (0, 3)


def write_record(root, relative, document):
    p = root / relative
    p.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(document).encode()
    p.write_bytes(raw)
    return dict(path=str(p.resolve()), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def parent_fixture(monkeypatch):
    root = m.PARENT
    root.mkdir()
    source = m.WORK / "continue_sparse_instance_five_minutes.py"
    source.write_text("# synthetic source only\n")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(m, "PARENT_SOURCE_SHA", source_sha)
    outputs = {}
    binding = dict(synthetic=True)
    checkpoint_binding = dict(original_binding=binding)
    metadata = dict(
        schema="persistent_tracker_full_state_v1",
        initialize_count=1,
        restore_count=6,
        previous_frame=m.START - 1,
        processed_frames=m.START - m.FIRST,
        context=dict(
            implementation="SparseInstanceRescoreContinuousTracker",
            sparse_distance_backend=dict(effective_from_native=78020),
        ),
    )
    metaname = m.CHECKPOINT_RELATIVE + "/metadata.json"
    outputs[metaname] = write_record(root, metaname, metadata)
    payloadname = m.CHECKPOINT_RELATIVE + "/arrays.npz"
    outputs[payloadname] = dict(
        path=str((root / payloadname).resolve()), sha256="a" * 64, size_bytes=123
    )
    manifest = dict(
        schema="medaka_tracker_checkpoint_v1",
        status="ok",
        bindings=checkpoint_binding,
        metadata=dict(
            file="metadata.json",
            sha256=outputs[metaname]["sha256"],
            bytes=outputs[metaname]["size_bytes"],
        ),
        payload=dict(file="arrays.npz", sha256="a" * 64, bytes=123),
    )
    manifestname = m.CHECKPOINT_RELATIVE + "/manifest.json"
    outputs[manifestname] = write_record(root, manifestname, manifest)
    monkeypatch.setattr(m, "CHECKPOINT_SHA", outputs[manifestname]["sha256"])
    plan = dict(
        original_binding=binding,
        checkpoint_binding=checkpoint_binding,
        source_sha256={str(source.resolve()): source_sha},
    )
    summary = dict(
        schema=root.name,
        status="ok",
        process_interval=[78020, 116200],
        retained_interval=[m.FIRST, 116200],
        scientific_interval=[m.SCIENCE_START, 116200],
        initialize_count=1,
        restore_count=6,
        new_begin_calls=0,
        external_pose_resets=0,
        finish_calls=1,
        all39_roundtrip=True,
        prior_history_prefix_unchanged=True,
        finish_matches_checkpoint_backtrace=True,
        checkpoints=[
            dict(
                previous_frame=m.START - 1,
                path=str(root / m.CHECKPOINT_RELATIVE),
                manifest_sha256=m.CHECKPOINT_SHA,
            )
        ],
    )
    for name, document in (
        ("plan", plan),
        ("summary", summary),
        ("input_index", dict(original_binding=binding, prefix_chunks=list(range(40)))),
    ):
        outputs[name + ".json"] = write_record(root, name + ".json", document)
    seal = dict(
        stage=root.name, status="ok", outputs=outputs, inputs={str(source.resolve()): source_sha}
    )
    record = write_record(root, "tracking.provenance.json", seal)
    monkeypatch.setattr(m, "PARENT_SHA", record["sha256"])
    return seal


def test_parent_headers_bind_native_checkpoint_without_opening_missing_payload(monkeypatch):
    parent_fixture(monkeypatch)
    result = m.load_parent_headers(m.original.phase.Digests())
    assert result["metadata"]["previous_frame"] == 81339
    assert result["metadata"]["restore_count"] == 6
    assert not (m.PARENT / m.CHECKPOINT_RELATIVE / "arrays.npz").exists()


def test_parent_requires_completed_success_before_any_payload(monkeypatch):
    seal = parent_fixture(monkeypatch)
    seal["status"] = "failed"
    record = write_record(m.PARENT, "tracking.provenance.json", seal)
    monkeypatch.setattr(m, "PARENT_SHA", record["sha256"])
    with pytest.raises(ValueError, match="successful Sparse"):
        m.load_parent_headers(m.original.phase.Digests())


def test_main_preserves_failure_and_never_implicitly_retries(isolate, monkeypatch):
    calls = []

    def fail(evidence):
        calls.append(True)
        raise ValueError("synthetic source mismatch")

    monkeypatch.setattr(m, "source_code", fail)
    output = isolate / "failure_run"
    with pytest.raises(ValueError, match="synthetic source mismatch"):
        m.main(SimpleNamespace(output=output, receipt_root=isolate / "receipts"))
    seal = m.proof.read_json(output / "tracking.provenance.json")
    assert seal["stage"] == m.STAGE and seal["status"] == "failed"
    assert calls == [True]
    with pytest.raises(ValueError, match="fresh workspace"):
        m.main(SimpleNamespace(output=output, receipt_root=isolate / "receipts"))
    assert calls == [True]
