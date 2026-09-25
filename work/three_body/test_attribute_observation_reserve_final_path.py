"""Synthetic final-path attribution; all paths and every ZIP read are temporary."""

import copy
import io
import json
import zipfile
from pathlib import Path

import attribute_observation_reserve_final_path as m
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    for module in (m, m.proof):
        monkeypatch.setattr(module, "REPO", tmp_path)
        monkeypatch.setattr(module, "WORK", tmp_path)
    monkeypatch.setattr(m, "TRACKING", tmp_path / m.TRACKING_STAGE)
    capture = m.proof.Evidence.capture

    def only_temp(self, path, expected=None):
        assert Path(path).resolve().is_relative_to(tmp_path)
        return capture(self, path, expected)

    monkeypatch.setattr(m.proof.Evidence, "capture", only_temp)
    monkeypatch.setattr(
        np, "load", lambda *a, **kw: pytest.fail("full checkpoint loader forbidden")
    )
    monkeypatch.setattr(
        m.codec, "read_checkpoint", lambda *a, **kw: pytest.fail("full39 decoder forbidden")
    )
    monkeypatch.setattr(
        m.proof.cv2, "VideoCapture", lambda *a, **kw: pytest.fail("video forbidden")
    )


def arrays_and_index():
    count = m.STOP - m.FIRST
    offsets = 6 * np.arange(count + 1, dtype=np.int64)
    parents = np.zeros(count * 6, np.int64)
    # A later winning descendant points to the reserved sixth slot at75053.
    parents[offsets[75054 - m.FIRST]] = 5
    arrays = dict(
        history_offsets=offsets,
        history_parent=parents,
        history_frame_idx=np.repeat(np.arange(m.FIRST, m.STOP, dtype=np.int64), 6),
        state_cost=np.arange(6, dtype=np.float64),
    )
    rows = []
    for frame in range(m.START, m.STOP):
        changed = frame == 75053
        rows.append(
            dict(
                frame_idx=frame,
                candidate_count=8,
                observation_reserve=dict(
                    baseline_indices=list(range(6)),
                    selected_indices=[0, 1, 2, 3, 4, 7] if changed else list(range(6)),
                    baseline_best_index=0,
                    changed=changed,
                    reservation_count=int(changed),
                    support_evaluation_count=8,
                    reason="reserved" if changed else "no_eligible",
                    replacement=dict(removed=5, admitted=7, extra_path_cost=2.0)
                    if changed
                    else None,
                ),
            )
        )
    metadata = dict(processed_frames=count, previous_frame=m.STOP - 1)
    return arrays, metadata, dict(rows=rows)


@pytest.fixture
def completed(tmp_path, monkeypatch):
    root = m.TRACKING
    root.mkdir()
    outputs = {}

    def artifact(name, value):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        data = value if isinstance(value, bytes) else json.dumps(value).encode()
        path.write_bytes(data)
        ref = dict(path=str(path), sha256=m.proof.sha(data), size_bytes=len(data))
        outputs[name] = ref
        return ref

    for name, attr in (
        ("test_caller.py", None),
        ("test_proof.py", "PROOF_SHA"),
        ("test_self.py", "__file__"),
    ):
        path = tmp_path / name
        path.write_bytes(name.encode())
        if attr == "PROOF_SHA":
            monkeypatch.setattr(m, attr, m.proof.sha(path.read_bytes()))
            monkeypatch.setattr(m.proof, "__file__", str(path))
        elif attr:
            monkeypatch.setattr(m, attr, str(path))
    code_sha = m.proof.sha((tmp_path / "test_caller.py").read_bytes())
    monkeypatch.setattr(m, "PINS", {"test_caller.py": code_sha})
    code = {str(tmp_path / "test_caller.py"): code_sha}
    context = dict(observation_reserve_policy=dict(effective_from_native=m.START))
    arrays, metadata, index = arrays_and_index()
    metadata.update(
        schema="persistent_tracker_full_state_v1",
        initialize_count=1,
        restore_count=5,
        context=context,
    )
    common = dict(
        schema=m.TRACKING_STAGE,
        process_interval=[m.START, m.STOP],
        scientific_interval=[m.START, m.STOP],
        retained_interval=[m.FIRST, m.STOP],
        parent_checkpoint_sha256=m.PARENT_SHA,
        initialize_count=1,
        restore_count=5,
        new_begin_calls=0,
        external_pose_resets=0,
        source_sha256=code,
        tracker_context=context,
    )
    binding = dict(source_sha256=code, tracker_context=context)
    plan = dict(**common, checkpoint_binding=binding)
    artifact("plan.json", plan)
    descriptors = {}
    all_names = sorted([*arrays, *(f"unread_{i:02}" for i in range(35))])
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for i, name in enumerate(all_names):
            storage = f"array_{i:06d}"
            value = arrays[name] if name in arrays else np.zeros(1, np.int8)
            descriptors[name] = m.codec._array_record(value, storage)
            if name in arrays:
                buffer = io.BytesIO()
                np.lib.format.write_array(buffer, value, allow_pickle=False)
                data = buffer.getvalue()
            else:
                # Deliberately unreadable sentinels expose accidental full39 decode.
                data = b"unread member sentinel"
            archive.writestr(storage + ".npy", data)
    payload = artifact(m.CHECKPOINT + "/arrays.npz", stream.getvalue())
    meta_ref = artifact(m.CHECKPOINT + "/metadata.json", metadata)
    manifest = dict(
        schema=m.codec.SCHEMA,
        status="ok",
        bindings=binding,
        arrays=descriptors,
        metadata=dict(
            file="metadata.json", sha256=meta_ref["sha256"], bytes=meta_ref["size_bytes"]
        ),
        payload=dict(file="arrays.npz", sha256=payload["sha256"], bytes=payload["size_bytes"]),
    )
    manifest_ref = artifact(m.CHECKPOINT + "/manifest.json", manifest)
    batches = []
    for start in range(m.START, m.STOP, 100):
        stop = min(start + 100, m.STOP)
        prefix = f"diagnostic_batches/f{start:06d}-{stop:06d}"
        batches.append(
            dict(
                first_frame=start,
                stop_frame_exclusive=stop,
                frame_count=stop - start,
                full_codec_roundtrip_verified=True,
                archive=artifact(prefix + ".npz", b"not accessed"),
                metadata=artifact(prefix + ".json", {}),
            )
        )
    index.update(
        process_interval=[m.START, m.STOP],
        batch_size=100,
        diagnostic_families=m.FAMILIES,
        batches=batches,
    )
    artifact("diagnostic_index.json", index)
    summary = dict(
        **common,
        status="ok",
        finish_calls=1,
        checkpoint_count=1,
        native_frames=1660,
        process_frames=1660,
        diagnostic_frames=1660,
        all39_roundtrip=True,
        completed_checkpoint_decoder_accepted=True,
        prior_history_prefix_unchanged=True,
        finish_matches_checkpoint_backtrace=True,
        checkpoint_manifest_sha256=manifest_ref["sha256"],
        checkpoint=dict(path=str(root / m.CHECKPOINT)),
        diagnostic_batch_count=len(batches),
        observation_reserve=dict(
            changed_frames=1,
            reasons={"reserved": 1, "no_eligible": 1659},
            current_beam_selection_only=True,
            final_path_reserved_ancestor_count_not_computed=True,
        ),
    )
    artifact("summary.json", summary)
    seal = artifact(
        "tracking.provenance.json",
        dict(stage=m.TRACKING_STAGE, status="ok", inputs=code, outputs=copy.deepcopy(outputs)),
    )
    return dict(
        arrays=arrays,
        metadata=metadata,
        index=index,
        manifest=manifest,
        payload=payload,
        tracking_sha=seal["sha256"],
    )


def test_complete_synthetic_run_native75053_and_exactly_four_member_reads(
    tmp_path, monkeypatch, completed
):
    original = zipfile.ZipFile.open
    allowed = [completed["manifest"]["arrays"][name]["storage_key"] + ".npy" for name in m.FIELDS]
    reads = []

    def opened(self, name, *args, **kwargs):
        assert Path(self.filename).is_relative_to(tmp_path)
        assert name in allowed
        reads.append(name)
        return original(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", opened)
    out = tmp_path / "result"
    summary = m.run(completed["tracking_sha"], out)
    assert reads == allowed
    assert (
        summary["current_beam_reservation_count"]
        == summary["final_path_reserved_admission_count"]
        == 1
    )
    assert summary["native75053"] == dict(
        final_beam_slot=5,
        final_fitted_candidate_index=7,
        uses_reserved_admission=True,
        current_beam_changed=True,
    )
    trace = json.loads((out / "attribution.json").read_text())
    assert (
        len(trace["frame_idx"]) == 1660
        and trace["reserved_final_path_frames"][0]["frame_idx"] == 75053
    )
    assert trace["final_beam_slot"][-1] == 0
    assert not summary["checkpoint_array_evidence"]["current_full_archive_sha_recomputed"]
    assert summary["checkpoint_array_evidence"]["decompressed_members"] == 4
    assert json.loads((out / "attribution.provenance.json").read_text())["status"] == "ok"


def test_current_beam_change_can_have_zero_final_admissions_and_ties_keep_state_zero():
    arrays, metadata, index = arrays_and_index()
    arrays["history_parent"][:] = 0
    arrays["state_cost"][:] = 0
    result = m.attribute(arrays, metadata, index)
    assert result["current_beam_changed_native_frames"] == [75053]
    assert result["reserved_final_path_frames"] == []
    assert result["final_beam_slot"] == [0] * 1660


def test_invalid_parent_in_unselected_candidate_is_rejected():
    arrays, metadata, index = arrays_and_index()
    arrays["history_parent"][arrays["history_offsets"][m.START - m.FIRST] + 4] = 6
    with pytest.raises(ValueError, match="preceding beam"):
        m.attribute(arrays, metadata, index)


def test_wrong_native_row_is_rejected():
    arrays, metadata, index = arrays_and_index()
    arrays["history_frame_idx"][arrays["history_offsets"][75053 - m.FIRST] + 4] += 1
    with pytest.raises(ValueError, match="native arrays"):
        m.attribute(arrays, metadata, index)


@pytest.mark.parametrize("fault", ["outside_candidate", "wrong_replacement"])
def test_selection_assignment_contract_rejected(fault):
    arrays, metadata, index = arrays_and_index()
    reserve = index["rows"][75053 - m.START]["observation_reserve"]
    if fault == "outside_candidate":
        reserve["selected_indices"][-1] = 8
    else:
        reserve["replacement"]["admitted"] = 6
    with pytest.raises(ValueError, match=r"index disagrees|replacement disagrees"):
        m.attribute(arrays, metadata, index)


def test_selected_member_descriptor_corruption_rejected(completed):
    manifest = copy.deepcopy(completed["manifest"])
    manifest["arrays"]["history_parent"]["bytes_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="signature differs"):
        m.load_four_fields(manifest, completed["payload"])
