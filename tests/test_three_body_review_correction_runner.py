"""Exporter conservation and sealed I/O; synthetic arrays are not human labels."""

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

LAB = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))
SPEC = importlib.util.spec_from_file_location(
    "review_correction_runner_tests", LAB / "run_review_correction.py"
)
m = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = m
SPEC.loader.exec_module(m)


def data():
    pose = np.arange(108, dtype=np.float64).reshape(4, 3, 3, 3)
    observed = np.ones((4, 3, 3), bool)
    observed[2, 0, 2] = False
    sources = (
        np.broadcast_to(np.arange(3)[None, :, None, None], (4, 3, 3, 3)).astype(np.int32).copy()
    )
    sources[2, 0, :, 2] = -1
    review = np.zeros((4, 3), bool)
    review[2:, 1] = True
    accepted = np.where(review[..., None], np.nan, pose[:, :, 1])
    geometry = dict(
        frame_idx=np.arange(100, 104, dtype=np.int32),
        pose_xyz=pose,
        sources=sources,
        observed=observed,
        identity_review=review,
        accepted_xyz=accepted,
    )
    mapping = np.tile(np.arange(3, dtype=np.int32), (4, 1))
    mapping[2:, 1] = -1
    unknown = mapping < 0
    corrected_pose, corrected_source, corrected_observed = (
        pose.copy(),
        sources.copy(),
        observed.copy(),
    )
    corrected_pose[unknown] = np.nan
    corrected_source[unknown] = -1
    corrected_observed[unknown] = False
    node_xyz = np.where(corrected_observed[..., None], corrected_pose, np.nan)
    output = dict(
        frame_idx=geometry["frame_idx"].copy(),
        corrected_identity_to_raw_slot=mapping,
        corrected_pose_xyz=corrected_pose,
        corrected_sources=corrected_source,
        corrected_observed=corrected_observed,
        corrected_node_xyz=node_xyz,
        corrected_xyz=node_xyz[:, :, 1].copy(),
        identity_origin=(~unknown).astype(np.int8),
        claim_index=np.where(unknown, -1, 0).astype(np.int32),
        tracklet_index=np.where(unknown, -1, 0).astype(np.int32),
        identity_conflict=np.zeros((4, 3), bool),
        prior_raw_review_for_identity=review.copy(),
    )
    return geometry, dict(arrays=output, metadata={"synthetic_test_fixture": True})


def test_zero_review_preserves_missing_nodes_without_increasing_acceptance():
    geometry, result = data()
    counts = m.validate_core_result(result, geometry, 0)
    assert counts["newly_available_pectoral"] == 0
    assert counts["original_automatic_accepted_pectoral"] == 10
    assert counts["unresolved_identity_frame_slots"] == 2
    assert np.isnan(result["arrays"]["corrected_node_xyz"][2, 0, 2]).all()


@pytest.mark.parametrize(
    "failure", ["pose", "source", "observed", "duplicate", "fill", "origin", "auto_gain"]
)
def test_export_rejects_geometry_invention_or_false_identity_availability(failure):
    geometry, result = data()
    arrays = result["arrays"]
    if failure == "pose":
        arrays["corrected_pose_xyz"][0, 0, 0, 0] += 1
    elif failure == "source":
        arrays["corrected_sources"][0, 0, 0, 0] = 2
    elif failure == "observed":
        arrays["corrected_observed"][2, 0, 2] = True
    elif failure == "duplicate":
        arrays["corrected_identity_to_raw_slot"][0, 1] = 0
    elif failure == "fill":
        arrays["corrected_node_xyz"][2, 0, 2] = geometry["pose_xyz"][2, 0, 2]
    elif failure == "origin":
        arrays["identity_origin"][2, 1] = 2
    else:
        geometry["accepted_xyz"][0, 0] = np.nan
    with pytest.raises(ValueError):
        m.validate_core_result(result, geometry, 0)


def setup_main(tmp_path, monkeypatch, effect=None):
    geometry, result = data()
    root = tmp_path / "geometry"
    root.mkdir()
    candidate = root / "candidate.npz"
    np.savez_compressed(candidate, **geometry)
    source = tmp_path / "synthetic_core_source.py"
    source.write_text("# synthetic exported-core fixture\n")
    prepared = m.PreparedReviews(
        run=SimpleNamespace(root=root, frames=geometry["frame_idx"], params={"tracker_config": {}}),
        geometry=geometry,
        review_paths=(),
        review_bytes=(),
        bundles=(),
        combined=None,
        verified_hashes={str(candidate): m.base.sha256(candidate)},
        metadata={"candidate_sha256": m.base.sha256(candidate), "supplied_review_files": 0},
    )
    monkeypatch.setattr(m, "load_review_inputs", lambda *_: prepared)
    monkeypatch.setattr(m, "code_paths", lambda: [source])
    original_import = m.importlib.import_module

    def core(arrays, bundles, *, tracker_config, candidate_sha256):
        assert not bundles and tracker_config == {}
        assert candidate_sha256 == m.base.sha256(candidate)
        if effect == "source":
            source.write_text("# changed during the operation\n")
        elif effect == "input":
            candidate.write_bytes(b"changed input after initial load")
        elif effect == "geometry":
            arrays["pose_xyz"][0, 0, 0, 0] += 1
        elif effect == "snapshot":
            target = tmp_path / "output/code_snapshot" / source.relative_to(m.REPO)
            target.write_text("changed snapshot\n")
        return copy.deepcopy(result)

    fake = SimpleNamespace(build_review_correction=core)
    monkeypatch.setattr(
        m.importlib,
        "import_module",
        lambda name, *a, **kw: (
            fake if name == "review_correction" else original_import(name, *a, **kw)
        ),
    )
    output = tmp_path / "output"
    return root, candidate, output


def test_zero_review_main_preserves_original_npz_and_seals_every_output(tmp_path, monkeypatch):
    root, candidate, output = setup_main(tmp_path, monkeypatch)
    before = candidate.read_bytes()
    summary = m.main(["--geometry-run", str(root), "--output", str(output)])
    seal = json.loads((output / "correction.provenance.json").read_text())
    assert seal["status"] == "ok" and seal["params"]["post_inference_assets_opened"] is False
    assert summary["export_conservation"]["newly_available_pectoral"] == 0
    assert candidate.read_bytes() == before == (output / "raw_geometry.npz").read_bytes()
    assert not (output / "human_reviews").exists()
    assert json.loads((output / "constraints.json").read_text())["bundles"] == []
    for item in seal["outputs"].values():
        assert m.base.sha256(item["path"]) == item["sha256"]


@pytest.mark.parametrize("effect", ["source", "input", "geometry", "snapshot"])
def test_main_mutation_produces_failure_seal_not_success(tmp_path, monkeypatch, effect):
    root, _, output = setup_main(tmp_path, monkeypatch, effect)
    with pytest.raises(RuntimeError):
        m.main(["--geometry-run", str(root), "--output", str(output)])
    assert json.loads((output / "correction.provenance.json").read_text())["status"] == "failed"


def real_geometry():
    geometry, _ = data()
    geometry.update(
        body_supported=np.ones((4, 3), bool),
        mask_supported=np.zeros((4, 3), bool),
        missing=np.zeros((4, 3), np.int64),
        score_margin=np.full(4, np.inf),
        xyz=geometry["pose_xyz"][:, :, 1].copy(),
    )
    return geometry


def test_real_core_zero_review_main_conservation(tmp_path, monkeypatch):
    core = m.importlib.import_module("review_correction")
    geometry = real_geometry()
    root = tmp_path / "geometry"
    root.mkdir()
    candidate = root / "candidate.npz"
    np.savez_compressed(candidate, **geometry)
    digest = m.base.sha256(candidate)
    prepared = m.PreparedReviews(
        run=SimpleNamespace(
            root=root,
            frames=geometry["frame_idx"],
            params={
                "tracker_config": {"review_gap_frames": 4, "ambiguity_margin": 1.0},
            },
        ),
        geometry=geometry,
        review_paths=(),
        review_bytes=(),
        bundles=(),
        combined=None,
        verified_hashes={str(candidate): digest},
        metadata={"candidate_sha256": digest, "supplied_review_files": 0},
    )
    monkeypatch.setattr(m, "load_review_inputs", lambda *_: prepared)
    monkeypatch.setattr(m, "code_paths", lambda: [Path(core.__file__).resolve()])
    before = candidate.read_bytes()
    output = tmp_path / "output"
    summary = m.main(["--geometry-run", str(root), "--output", str(output)])
    assert summary["export_conservation"]["newly_available_pectoral"] == 0
    saved = m.verified._npz(output / "correction.npz")
    np.testing.assert_array_equal(saved["corrected_identity_to_raw_slot"][2:, 1], -1)
    assert np.isnan(saved["corrected_node_xyz"][2, 0, 2]).all()
    assert before == candidate.read_bytes() == (output / "raw_geometry.npz").read_bytes()
    assert json.loads((output / "correction.provenance.json").read_text())["status"] == "ok"


def test_real_core_human_endpoint_does_not_invent_unsupported_tail():
    # Entirely synthetic in-memory marks. No human review JSON file is written.
    from test_three_body_review_evaluation import human_input

    core = m.importlib.import_module("review_correction")
    geometry = real_geometry()
    geometry["observed"][2, 1, 2] = False
    geometry["sources"][2, 1, :, 2] = -1
    projected = np.stack([geometry["pose_xyz"][:, :, 0, :2] * 2 + v for v in range(3)], axis=2)
    review, sequence = human_input()
    sequence["frames"] = geometry["frame_idx"].tolist()
    manifest = json.dumps(sequence).encode()
    review["source_manifest_sha256"] = m.hashlib.sha256(manifest).hexdigest()
    review["source_frame_range"] = [100, 104]
    review["reviewer"] = "synthetic_cli_core_binding_fixture"
    for probe, (anchor, frame) in enumerate(zip(review["anchors"], [100, 102, 103], strict=True)):
        anchor["frame"], anchor["frame_choice"] = frame, "human_selected"
        for animal, label in enumerate(("A", "B", "C")):
            slot = (0, 1, 2)[animal] if probe == 0 else (1, 0, 2)[animal]
            anchor["animals"][label] = dict(
                visibility="visible",
                points={
                    view: projected[frame - 100, slot, v].tolist()
                    for v, view in enumerate(m.base.VIEWS)
                },
            )
    for label, continuity in review["continuity"].items():
        continuity.update(
            entry_to_after="confirmed" if label == "A" else "ambiguous",
            after_to_exit="unreviewed",
            basis="direct_visual_continuity",
        )
    bundle = m.review_constraints.extract_review_constraints(
        json.dumps(review).encode(),
        manifest,
        geometry,
        projected,
        candidate_sha256="c" * 64,
        calibration_sha256={"fixture": "d" * 64},
        projection_origin_sha256="e" * 64,
    )
    result = core.build_review_correction(
        geometry,
        [bundle],
        tracker_config={"review_gap_frames": 4, "ambiguity_margin": 1.0},
        candidate_sha256="c" * 64,
    )
    m.validate_core_result(result, geometry, 1)
    arrays = result["arrays"]
    assert arrays["corrected_identity_to_raw_slot"][2, 0] == 1
    assert arrays["identity_origin"][2, 0] == 2
    assert not arrays["corrected_observed"][2, 0, 2]
    assert np.isnan(arrays["corrected_node_xyz"][2, 0, 2]).all()
    assert np.isnan(arrays["corrected_node_xyz"][1, 0]).all()


def test_workload_deduplicates_bytes_and_keeps_unknown_duration():
    document = dict(
        elapsed_review_seconds=None,
        review_clip_range=[100, 140],
        source_frame_range=[0, 1000],
        anchors=[
            dict(
                animals={
                    "A": dict(
                        visibility="visible", points={"top_cam": [1, 2], "side1_cam": [3, 4]}
                    ),
                    "B": dict(visibility="ambiguous", points={"top_cam": [5, 6]}),
                }
            )
        ],
        continuity={
            "A": dict(entry_to_after="confirmed", after_to_exit="ambiguous"),
            "B": dict(entry_to_after="ambiguous", after_to_exit="unreviewed"),
        },
    )
    content = json.dumps(document).encode()
    result = m.summarize_review_workload([document, copy.deepcopy(document)], [content, content])
    assert result["distinct_document_sha256"] == 1 and result["repeated_document_files"] == 1
    assert result["saved_mouth_coordinate_pairs"] == 3
    assert result["visible_mouth_coordinate_pairs"] == 2
    assert result["confirmed_edge_count"] == 1
    assert result["elapsed_total_seconds"] is None
    assert result["known_elapsed_partial_seconds"] is None
    assert result["unknown_or_invalid_elapsed_documents"] == 1
    saved = result["unique_documents"][0]
    assert saved["elapsed_review_seconds"] is None
    assert saved["declared_review_clip_range"] == [100, 140]
    assert saved["supplied_file_indices"] == [0, 1]


def test_workload_known_zero_is_preserved_and_not_conflated_with_unknown():
    documents = [
        dict(elapsed_review_seconds=elapsed, anchors=[], continuity={})
        for elapsed in (0, 12.5, None)
    ]
    contents = [json.dumps(item).encode() for item in documents]
    complete = m.summarize_review_workload(documents[:2], contents[:2])
    assert complete["elapsed_total_seconds"] == 12.5
    assert complete["known_elapsed_documents"] == 2
    incomplete = m.summarize_review_workload(documents, contents)
    assert incomplete["elapsed_total_seconds"] is None
    assert incomplete["known_elapsed_partial_seconds"] == 12.5
    assert [row["elapsed_review_seconds"] for row in incomplete["unique_documents"]] == [
        0,
        12.5,
        None,
    ]
