"""Fixed-method comparison tests, synthetic sources only; no archived inputs or GT."""

import copy
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import project

WORK = Path(__file__).resolve().parents[1] / "work/three_body"
sys.path.insert(0, str(WORK))
m = importlib.import_module("compare_long_support_160s")


@pytest.fixture
def synthetic(rig, monkeypatch):
    first, stop = 10, 14
    for key, value in dict(FIRST=first, STOP=stop, BLOCK=2).items():
        monkeypatch.setattr(m, key, value)
    frames = np.arange(first, stop, dtype=np.int64)
    body = np.array([[40.0, 60.0, 30.0], [120.0, 90.0, 100.0], [70.0, 150.0, 160.0]])[:, None]
    body = body + np.array([[5.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-12.0, 1.0, -1.0]])[None]
    pose = np.repeat(body[None], 4, axis=0)
    pose[:, :, :, 0] += np.arange(4)[:, None, None] * 0.1

    class Nodes:
        def at(self, view, frame):
            return project(rig.camera(view), pose[frame - first])

    nodes = Nodes()
    rows = np.broadcast_to(np.arange(3)[None, :, None], (4, 3, 3)).astype(np.int8).copy()
    raw = m.baseline.reconstruct_sources(
        frames, rows, pose[:, :, 1].astype(np.float32), nodes, rig, block_frames=2
    )
    raw.update(
        label_available=np.ones((4, 3), bool),
        slot_track=np.broadcast_to(np.arange(3), (4, 3)).astype(np.int32).copy(),
    )
    new = dict(
        frame_idx=frames.copy(),
        pose_xyz=pose.copy(),
        sources=np.repeat(rows[..., None], 3, axis=-1),
        observed=np.ones((4, 3, 3), bool),
        identity_review=np.zeros((4, 3), bool),
        xyz=pose[:, :, 1].copy(),
        accepted_xyz=pose[:, :, 1].copy(),
    )
    return raw, new, nodes, rig


def same_dict(left, right):
    assert set(left) == set(right)
    for key in left:
        assert left[key].dtype == right[key].dtype
        assert left[key].tobytes() == right[key].tobytes(), key


def test_rejected_finite_saved_chest_and_label_mask_remain_distinct(synthetic):
    raw, _, nodes, rig = synthetic
    raw["baseline_xyz"][1, 0] += 25
    raw["pose_xyz"][1, 0, 1] = raw["baseline_xyz"][1, 0]
    raw["label_available"][0, 1] = False
    raw["slot_track"][0, 1] = -1
    before = copy.deepcopy(raw)
    strict = m.baseline.classify_strict_support(raw, nodes, rig, block_frames=2)
    assert not strict["observed"][1, 0, 1]
    assert strict["observed"][0, 1, 1]
    metrics = m.metric_views(raw, strict, is_new=False)
    assert metrics["saved_pectoral_finite_fish_frames"] == 12
    assert metrics["strict"]["motion"]["observed_animal_frames"] == 11
    assert metrics["native_acceptance"]["motion"]["observed_animal_frames"] == 11
    assert metrics["strict_and_native_acceptance"]["motion"]["observed_animal_frames"] == 10
    # Two steps around missing t=1 cannot be replaced by a t=0 -> t=2 bridge.
    assert metrics["strict"]["motion"]["scored_adjacent_steps"] == 7
    same_dict(raw, before)


def test_native_frame_gap_is_not_motion_step():
    xyz = np.zeros((3, 3, 3))
    xyz[1:] = 100
    result = m.coordinate_metrics(xyz, np.array([10, 12, 13]))
    assert result["motion"]["scored_adjacent_steps"] == 3
    assert result["motion"]["step_p99_mm"] == 0


def test_review_never_cleared_by_direct_support(synthetic):
    _, new, _, _ = synthetic
    new["identity_review"][1:, 0] = True
    new["accepted_xyz"][1:, 0] = np.nan
    before = copy.deepcopy(new)
    metrics = m.metric_views(new, new, is_new=True)
    assert metrics["strict"]["motion"]["observed_animal_frames"] == 12
    assert metrics["native_acceptance"]["motion"]["observed_animal_frames"] == 9
    assert metrics["original_review_fish_frames"] == 3
    same_dict(new, before)


def test_all_missing_remains_in_denominator(synthetic):
    raw, _, nodes, rig = synthetic
    raw["pose_xyz"][:] = np.nan
    raw["baseline_xyz"][:] = np.nan
    raw["sources"][:] = -1
    strict = m.baseline.classify_strict_support(raw, nodes, rig, block_frames=2)
    metrics = m.metric_views(raw, strict, is_new=False)
    assert metrics["strict"]["motion"]["total_animal_frames"] == 12
    assert metrics["strict"]["motion"]["observed_animal_frames"] == 0
    assert metrics["strict"]["motion"]["step_p99_mm"] is None


@pytest.mark.parametrize("is_new", [False, True])
def test_real_partitioned_support_saved_block_schema_and_copy(synthetic, tmp_path, is_new):
    raw, new, nodes, rig = synthetic
    original = new if is_new else raw
    before = copy.deepcopy(original)
    name = "new" if is_new else "c34_lambda_2"
    report = m.process_method(name, original, nodes, rig, tmp_path)
    assert report["all_source_geometry"]["all_sources_and_observed_flags_pass"]
    assert [b["frame_range"] for b in report["blocks"]] == [[10, 12], [12, 14]]
    for block in report["blocks"]:
        path = tmp_path / block["path"]
        assert m.baseline.sha256(path) == block["sha256"]
        with np.load(path, allow_pickle=False) as saved:
            assert saved["sources"].dtype == np.int64
            assert saved["sources"].shape == (2, 3, 3, 3)
            assert saved["observed"].shape == (2, 3, 3)
            assert saved["original_label_available"].dtype == np.bool_
            assert saved["original_xyz"].dtype == (np.float64 if is_new else np.float32)
            assert np.array_equal(saved["valid_source"], saved["sources"] >= 0)
            assert np.array_equal(saved["observed"], saved["valid_source"].sum(2) >= 2)
            assert ("slot_track" in saved) == (not is_new)
            assert ("reconstructed_pose_xyz" in saved) == (not is_new)
    same_dict(original, before)


@pytest.mark.parametrize("change", ["omit", "reverse", "duplicate", "interval", "gap"])
def test_fixed_all_seven_conditions_cannot_be_selected_or_reordered(synthetic, change):
    methods = [
        dict(
            name=name,
            frame_range=[10, 14],
            blocks=[dict(frame_range=[10, 12]), dict(frame_range=[12, 14])],
        )
        for name in m.METHODS
    ]
    m.check_method_set(methods)
    if change == "omit":
        methods.pop()
    elif change == "reverse":
        methods.reverse()
    elif change == "duplicate":
        methods[-1] = copy.deepcopy(methods[-2])
    elif change == "interval":
        methods[-1]["frame_range"] = [11, 15]
    else:
        methods[-1]["blocks"].pop()
    with pytest.raises(ValueError):
        m.check_method_set(methods)


def test_sealed_outputs_checks_outputs_but_does_not_open_historical_videos(tmp_path):
    output = tmp_path / "payload.bin"
    output.write_bytes(b"frozen")
    seal = tmp_path / "seal.json"
    record = dict(
        stage="a",
        status="ok",
        inputs={"raw": dict(path="NONEXISTENT.mp4")},
        outputs={"payload": dict(path=str(output), size=6, sha256=m.baseline.sha256(output))},
    )
    seal.write_text(json.dumps(record))
    evidence = m.common.Digests()
    assert m.sealed_outputs(seal, m.baseline.sha256(seal), "a", evidence) == record
    assert set(evidence.paths) == {str(seal), str(output)}
    output.write_bytes(b"broken")
    with pytest.raises(ValueError):
        m.sealed_outputs(seal, m.baseline.sha256(seal), "a", m.common.Digests())


def test_incomplete_seal_rejected_before_payload_access(tmp_path):
    seal = tmp_path / "seal.json"
    seal.write_text(json.dumps(dict(stage="a", status="failed", outputs={"x": {}})))
    with pytest.raises(ValueError, match="completed"):
        m.sealed_outputs(seal, m.baseline.sha256(seal), "a", m.common.Digests())


@pytest.fixture
def hermetic_run(synthetic, monkeypatch, tmp_path):
    raw, new, nodes, rig = synthetic
    monkeypatch.setattr(m, "REPO", tmp_path)
    monkeypatch.setattr(m, "PINS", {})
    monkeypatch.setattr(m.common, "PINS", {})
    monkeypatch.setattr(m, "code_paths", lambda: [])
    monkeypatch.setattr(m, "completed_inputs", lambda evidence: (new, nodes, rig, {}, {}))
    lambdas = []

    def load(root, value, first, stop, **kwargs):
        lambdas.append(value)
        assert (first, stop) == (10, 14)
        return m.baseline.BaselineSupportInputs(copy.deepcopy(raw), nodes, rig, {}, {})

    monkeypatch.setattr(m.baseline, "load_baseline_sources", load)
    return tmp_path / "result", lambdas


def test_real_core_all8_orchestration_success_and_final_output_seal(hermetic_run):
    output, lambdas = hermetic_run
    report = m.run(output)
    assert lambdas == list(m.baseline.LAMBDAS)
    assert [x["name"] for x in report["methods"]] == list(m.METHODS)
    assert len(list(output.glob("*/support_*.npz"))) == 16
    record = json.loads((output / "comparison.provenance.json").read_text())
    assert record["status"] == "ok"
    assert record["stage"] == m.STAGE
    for item in record["outputs"].values():
        assert m.baseline.sha256(item["path"]) == item["sha256"]
    with pytest.raises(ValueError, match="new workspace"):
        m.run(output)


def test_final_guard_failure_retains_failed_seal(hermetic_run, monkeypatch):
    output, _ = hermetic_run

    def fail(self):
        raise ValueError("input changed during comparison")

    monkeypatch.setattr(m.common.Digests, "final_guard", fail)
    with pytest.raises(ValueError, match="input changed"):
        m.run(output)
    record = json.loads((output / "comparison.provenance.json").read_text())
    assert record["status"] == "failed"
    assert len(list(output.glob("*/support_*.npz"))) == 16
