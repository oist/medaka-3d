"""CPU input/correspondence tests; no archived run, raw video or tracker is used."""

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import project
from medaka3d.types import VIEWS

PATH = Path(__file__).resolve().parents[1] / "work/three_body/long_baseline_support.py"
SPEC = importlib.util.spec_from_file_location("long_baseline_support_cpu_test", PATH)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


@pytest.fixture
def corpus(tmp_path, rig, monkeypatch):
    monkeypatch.setattr(module, "FRAME_COUNT", 12)
    monkeypatch.setattr(module, "CALIBRATION", "calibration")
    monkeypatch.setattr(module, "BASELINE", "baselines")
    root = tmp_path / "registered"
    root.mkdir()
    cal = root / module.CALIBRATION
    cal.mkdir(parents=True)
    for path in rig.manifest.parent.glob("*.json"):
        shutil.copyfile(path, cal / path.name)
    pose = np.array([[40., 60., 30.], [120., 90., 100.], [70., 150., 160.]])[:, None]
    pose = pose + np.array([[5., 0., 0.], [0., 0., 0.], [-12., 1., -1.]])[None]
    source = np.empty((12, 3, 3), np.int8)
    source[:] = -1
    for vi, view in enumerate(VIEWS):
        order = np.roll(np.arange(3), vi)
        points = project(rig.camera(view), pose)[order]
        relative = f"work/c01_s16_predictions/bottomup/{module.EXPERIMENT}__{view}.npz"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, experiment=module.EXPERIMENT, view=view, nodes=module.NODES,
                            frame_idx=np.repeat(np.arange(5, 8), 3),
                            points=np.tile(points, (3, 1, 1)))
        source[5:8, :, vi] = np.argsort(order)
    xyz = np.full((12, 3, 3), np.nan, np.float32)
    xyz[5:8] = pose[:, 1]
    slot = np.full((12, 3), -1, np.int32)
    slot[5:8] = np.arange(3)
    fields = dict(slot_track=slot, xyz=xyz, source=source,
                  residual_mm=np.zeros((12, 3), np.float32), margin_mm=np.zeros(12, np.float32),
                  n_views_used=np.full(12, 3, np.int8),
                  **{f"track__{v}": np.full(9, -1, np.int32) for v in VIEWS})
    baseline = root / module.BASELINE / f"C34-label__{module.EXPERIMENT}__lam2.npz"
    baseline.parent.mkdir(parents=True)
    np.savez_compressed(baseline, **fields)

    def catalog():
        files = [dict(relative=p.relative_to(root).as_posix(), bytes=p.stat().st_size,
                      sha256=module.sha256(p)) for p in root.rglob("*")
                 if p.is_file() and p.name != "staging_manifest.json"]
        manifest = root / "staging_manifest.json"
        manifest.write_text(json.dumps(dict(files=files)), encoding="utf-8")
        return module.sha256(manifest)

    def load():
        return module.load_baseline_sources(root, "2", 5, 8,
                                            expected_catalog_sha256=catalog(), block_frames=2)
    return root, baseline, fields, pose, catalog, load


def test_native_rows_reconstruct_without_relabel_and_preserve_saved_pectoral(corpus):
    _, baseline, fields, pose, _, load = corpus
    fields["xyz"][5, 0] += [0., 25., 0.]
    fields["slot_track"][6, 1] = -1
    np.savez_compressed(baseline, **fields)
    result = load()
    a = result.arrays
    np.testing.assert_allclose(a["reconstructed_pose_xyz"], np.tile(pose, (3, 1, 1, 1)), atol=1e-8)
    np.testing.assert_array_equal(a["original_source_rows"], fields["source"][5:8])
    np.testing.assert_array_equal(a["sources"], np.repeat(fields["source"][5:8, ..., None], 3, -1))
    np.testing.assert_array_equal(a["pose_xyz"][:, :, 1], fields["xyz"][5:8])
    assert a["baseline_xyz"].dtype == np.float32
    assert a["finite_two_ray_observed"].all() and a["observed"].all()
    assert not a["label_available"][1, 1]  # Original label status is not overwritten by geometry.
    assert np.linalg.norm(a["pose_xyz"][0, 0, 1] - pose[0, 1]) == 25
    assert result.metadata["strict_support_computed"] is False
    assert len(result.verified_hashes) == 9
    result.final_guard()
    before = {key: value.copy() for key, value in a.items()}
    strict = module.classify_strict_support(a, result.nodes, result.rig, block_frames=1)
    assert not strict["observed"][0, 0, 1]
    assert strict["observed"][1:].all()
    np.testing.assert_array_equal(strict["pose_xyz"][:, :, 1], fields["xyz"][5:8])
    for key, value in before.items():
        np.testing.assert_array_equal(a[key], value)


def test_minus_one_rows_and_nonfinite_nodes_remain_uninvented(corpus):
    root, baseline, fields, _, _, load = corpus
    fields["source"][5:8, 0, 0] = -1
    fields["source"][5:8, 1, :2] = -1
    np.savez_compressed(baseline, **fields)
    path = root / f"work/c01_s16_predictions/bottomup/{module.EXPERIMENT}__{VIEWS[2]}.npz"
    with np.load(path, allow_pickle=False) as data:
        values = {k: data[k] for k in data.files}
    values["points"][:, 2] = np.nan
    np.savez_compressed(path, **values)
    result = load()
    a = result.arrays
    assert (a["sources"][:, 0, 0] == -1).all()
    assert a["finite_two_ray_observed"][:, 0, :2].all()
    assert not a["finite_two_ray_observed"][:, 1].any()
    assert np.isnan(a["pose_xyz"][:, 1, [0, 2]]).all()
    assert not a["observed"][:, 1].any()  # Saved pectoral alone is not two-view observation.
    assert (a["sources"][:, 2, 2, 2] >= 0).all()  # Original row retained even with missing tail.
    assert not a["finite_ray_source"][:, :, 2, 2].any()
    strict = module.classify_strict_support(a, result.nodes, result.rig)
    assert not strict["observed"][:, 1].any()
    assert (strict["sources"][:, 1, :, [0, 2]] == -1).all()
    assert (strict["sources"][:, :, 2, 2] == -1).all()
    assert np.isnan(strict["pose_xyz"][:, 1, [0, 2]]).all()


def test_strict_blocking_and_thresholds_match_independent_geometry(corpus):
    result = corpus[-1]()
    whole = module.classify_strict_support(result.arrays, result.nodes, result.rig,
                                          block_frames=100)
    one = module.classify_strict_support(result.arrays, result.nodes, result.rig, block_frames=1)
    for key in whole:
        np.testing.assert_array_equal(one[key], whole[key])
    assert whole["observed"].all()
    for vi, view in enumerate(VIEWS):
        raw = result.arrays["sources"][:, :, vi]
        uv = np.array([[result.nodes.at(view, int(frame))[raw[t, f, n], n]
                        for f in range(3) for n in range(3)]
                       for t, frame in enumerate(result.arrays["frame_idx"])]).reshape(3, 3, 3, 2)
        origin, direction = result.rig.camera(view).rays(uv)
        origin, direction = origin.astype(np.float32), direction.astype(np.float32)
        delta = result.arrays["pose_xyz"] - origin
        d = np.linalg.norm(delta-np.sum(delta*direction, -1)[..., None]*direction, axis=-1)
        np.testing.assert_array_equal(d, whole["ray_distance_mm"][:, :, vi])
        assert np.max(d) < 1e-4


def test_common_partitioned_verifier_accepts_missing_and_filtered_declarations(corpus):
    # This integration uses the new shared verifier, including its whole-NaN rule.
    sys.path.insert(0, str(PATH.parent))
    import audit_long_final as common

    result = corpus[-1]()
    a = {key: value.copy() for key, value in result.arrays.items()}
    a["pose_xyz"][0, 0] = np.nan
    a["pose_xyz"][1, 1, 1] += [0., 25., 0.]
    strict = module.classify_strict_support(a, result.nodes, result.rig, block_frames=1)
    seen = []
    totals = common.verify_support_partitioned(
        strict, result.nodes, result.rig, block_frames=2,
        on_block=lambda summary, data: seen.append(data))
    assert totals["native_frames"] == 3 and totals["node_denominator"] == 27
    assert totals["observed_nodes_checked"] == int(strict["observed"].sum())
    assert not strict["observed"][0, 0].any() and not strict["observed"][1, 1, 1]
    assert np.isnan(strict["pose_xyz"][0, 0]).all()
    np.testing.assert_array_equal(np.concatenate([x["valid_source"] for x in seen]),
                                  strict["valid_source"])
    # A declaration based only on finite rays is not silently accepted as strict support.
    original = {key: value.copy() for key, value in result.arrays.items()}
    original["pose_xyz"][1, 1, 1] += [0., 25., 0.]
    with pytest.raises(ValueError, match="ray/pixel gates"):
        common.verify_support_partitioned(original, result.nodes, result.rig, block_frames=1)


@pytest.mark.parametrize("start,stop", [(-1, 3), (0, 13), (7, 7), (True, 7), (0, 2.5)])
def test_invalid_native_range_fails_before_files(corpus, start, stop):
    root, _, _, _, catalog, _ = corpus
    with pytest.raises(ValueError, match="native range"):
        module.load_baseline_sources(root, "2", start, stop,
                                     expected_catalog_sha256=catalog())


@pytest.mark.parametrize("corruption,reason", [
    ("row", "out of range"), ("duplicate", "multiple fish"),
    ("gated", "label slots"), ("dtype", "shape/dtype"),
    ("partial_pectoral", "whole finite or NaN"),
])
def test_corrupt_associations_are_rejected_without_repair(corpus, corruption, reason):
    _, baseline, fields, _, _, load = corpus
    if corruption == "row":
        fields["source"][5, 0, 0] = 12
    elif corruption == "duplicate":
        fields["source"][5, 1, 0] = fields["source"][5, 0, 0]
    elif corruption == "gated":
        fields["slot_track"][5, 0] = 17
    elif corruption == "dtype":
        fields["source"] = fields["source"].astype(np.int32)
    else:
        fields["xyz"][5, 0, 0] = np.nan
    np.savez_compressed(baseline, **fields)
    with pytest.raises(ValueError, match=reason):
        load()


def test_catalog_pin_and_payload_and_final_guards(corpus):
    root, baseline, _, _, catalog, load = corpus
    pin = catalog()
    with pytest.raises(ValueError, match="catalog SHA"):
        module.load_baseline_sources(root, "2", 5, 8, expected_catalog_sha256="0"*64)
    result = load()
    with baseline.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="input SHA/size"):
        module.load_baseline_sources(root, "2", 5, 8, expected_catalog_sha256=pin)
    with pytest.raises(ValueError, match="input changed"):
        result.final_guard()


def test_wrong_sleap_node_order_is_not_guessed(corpus):
    root, _, _, _, _, load = corpus
    path = root / f"work/c01_s16_predictions/bottomup/{module.EXPERIMENT}__{VIEWS[0]}.npz"
    with np.load(path, allow_pickle=False) as data:
        values = {k: data[k] for k in data.files}
    values["nodes"] = np.array(["tail", "pectoral", "mouth"])
    np.savez_compressed(path, **values)
    with pytest.raises(ValueError, match="experiment/view/nodes"):
        load()


def test_outside_catalog_path_rejected(corpus):
    root, _, _, _, catalog, _ = corpus
    catalog()
    path = root / "staging_manifest.json"
    data = json.loads(path.read_text())
    data["files"][0]["relative"] = "../foreign.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe"):
        module.load_baseline_sources(root, "2", 5, 8,
                                     expected_catalog_sha256=module.sha256(path))
