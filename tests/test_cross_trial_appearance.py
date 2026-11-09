"""Checks for frozen-reference matching and cross-trial input provenance."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from medaka3d.provenance import Record

spec = importlib.util.spec_from_file_location(
    "cross_trial_appearance_tested",
    Path(__file__).parents[1] / "analyses/20261109_cross_trial_appearance/cross_trial_embed.py",
)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_centroid_matching_recovers_permutation_and_positive_margin():
    source = np.array([[0, 0], [10, 0], [0, 20]], dtype=float)
    target = source[[2, 0, 1]] + [0.1, 0]
    assignment, cost, margin, distances = analysis.best_assignment(target, source)
    assert assignment.tolist() == [2, 0, 1]
    assert cost == pytest.approx(0.3)
    assert distances == pytest.approx([0.1, 0.1, 0.1])
    assert margin > 10


def test_agreement_scores_complete_intervals_and_excludes_unknowns():
    frozen = {0: np.array([0, 1, 2]), 1: np.array([0, 1, 2]), 2: np.array([0, -1, 2])}
    own = {0: np.array([2, 0, 1]), 1: np.array([2, 1, 0]), 2: np.array([2, 0, 1])}
    mapping, best, second, total = analysis.duration_weighted_permutation(
        frozen, own, {0: 10.0, 1: 2.0, 2: 100.0}, [0, 1, 2]
    )
    assert mapping.tolist() == [1, 2, 0]
    assert (best, second, total) == (30.0, 6.0, 36.0)


@pytest.mark.parametrize("entry", [None, {}, {"sha256": "wrong"}, "valid"])
def test_shared_copy_requires_a_matching_crop_digest(tmp_path, monkeypatch, entry):
    run, shared = tmp_path / "run", tmp_path / "shared"
    (run / "clean").mkdir(parents=True)
    shared.mkdir()
    crop = run / "clean/top_cam.npz"
    crop.write_bytes(b"synthetic crop file for digest validation")
    digest = hashlib.sha256(crop.read_bytes()).hexdigest()
    monkeypatch.setattr(analysis.rn, "recorded_output", lambda *args: digest)
    files = {} if entry is None else {"clean\\top_cam.npz": entry}
    if entry == "valid":
        files["clean\\top_cam.npz"] = {"sha256": digest, "size": crop.stat().st_size}
    (shared / "staged_copy_manifest.json").write_text(json.dumps({"files": files}))
    record = Record(stage="test", started_at="synthetic")
    if entry == "valid":
        assert analysis.verify_target_clean(run, shared, "top_cam", record, "target") == crop
        assert record.inputs["target_clean_top_cam"]["sha256"] == digest
    else:
        with pytest.raises(ValueError, match="SHA-256"):
            analysis.verify_target_clean(run, shared, "top_cam", record, "target")
        assert not record.inputs


def test_model_provenance_retains_both_groups(tmp_path, monkeypatch):
    class Encoder:
        def load_state_dict(self, state):
            pass

        def eval(self):
            pass

        def to(self, device):
            pass

    payload = {
        "out_dim": 2,
        "encoder": "synthetic",
        "state_dict": {},
        "mean": [0, 0, 0],
        "std": [1, 1, 1],
        "views": ["top_cam"],
        "best_step": 0,
        "best_silhouette": 0,
    }
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=lambda *a, **k: payload))
    lab = SimpleNamespace(build_encoder=lambda *args: Encoder())
    record = Record(stage="test", started_at="synthetic")
    for group in ("K_1", "P_1"):
        run = tmp_path / group
        (run / "models").mkdir(parents=True)
        for view in ("top", "side"):
            (run / "models" / f"{view}.pt").write_text(f"synthetic {group} {view}")
        analysis.load_models(lab, run, "cpu", record, group)
    assert set(record.inputs) == {
        "K_1_model_top",
        "K_1_model_side",
        "P_1_model_top",
        "P_1_model_side",
    }
    assert len({e["sha256"] for e in record.inputs.values()}) == 4
