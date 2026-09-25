"""Synthetic in-memory constraints only; no human review or full30 run is used."""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "tests"), str(REPO / "labs/20260909_three_body")]
import review_correction  # noqa: E402
import review_full_scope as module  # noqa: E402

from test_three_body_review_correction import (  # noqa: E402
    CANDIDATE_SHA,
    CONFIG,
    bundle,
    geometry,
    loss,
    projections,
)


def digest_bundle(value):
    value.pop("content_sha256", None)
    value["content_sha256"] = module.constraints._digest(value)


@pytest.fixture
def inputs():
    full = geometry()
    loss(full, 6, 10)
    index = np.arange(3, 30, dtype=np.int64)
    sliced = {k: v[index].copy() for k, v in full.items()}
    projected_slice = projections(sliced)
    projected_full = np.full((len(full["frame_idx"]), 3, 3, 2), np.nan)
    projected_full[index] = projected_slice
    claim = bundle(sliced, projected=projected_slice)
    return dict(
        slice_bundle=claim,
        full_geometry=full,
        slice_geometry=sliced,
        slice_projected_mouth=projected_slice,
        full_projected_mouth=projected_full,
        slice_indices=index,
        candidate_sha256=CANDIDATE_SHA,
        calibration_sha256={"manifest": "d" * 64},
        projection_origin_sha256="e" * 64,
        source_proof_sha256="f" * 64,
    )


def saved_arrays(values):
    return {k: (v.dtype, v.shape, v.tobytes()) for k, v in values.items()}


def test_expansion_preserves_all_endpoints_links_human_origin_and_inputs(inputs, monkeypatch):
    original = copy.deepcopy(inputs["slice_bundle"])
    before = {k: saved_arrays(inputs[k]) for k in ("full_geometry", "slice_geometry")}
    projection_before = {
        k: inputs[k].tobytes()
        for k in ("slice_projected_mouth", "full_projected_mouth", "slice_indices")
    }
    monkeypatch.setattr(
        module.constraints, "_document", lambda *a: pytest.fail("review bytes read")
    )
    expanded, proof = module.bind_slice_constraints_to_full(**inputs)
    for key in original:
        if key not in ("content_sha256", "provenance"):
            assert expanded[key] == original[key], key
    assert inputs["slice_bundle"] == original
    assert expanded["provenance"]["original_extraction_provenance"] == original["provenance"]
    assert expanded["provenance"]["review_and_manifest_bytes_verified_here"] is False
    assert (
        expanded["provenance"]["original_extraction_provenance"][
            "review_and_manifest_bytes_verified_here"
        ]
        is True
    )
    assert proof["slice_native_range"] == [103, 130]
    assert proof["full_native_range"] == [100, 140]
    assert proof["anchor_frames"] == [105, 112, 125]
    assert proof["adapter_version_digest"] == module.ADAPTER_DIGEST
    assert (
        expanded["provenance"]["scope_binding"]["proof_content_sha256"] == proof["content_sha256"]
    )
    for values in (expanded, proof):
        copied = copy.deepcopy(values)
        digest = copied.pop("content_sha256")
        assert module.constraints._digest(copied) == digest
    for k, value in before.items():
        assert saved_arrays(inputs[k]) == value
    for k, value in projection_before.items():
        assert inputs[k].tobytes() == value
    proof["origin"]["candidate_sha256"] = "0" * 64
    expanded["endpoints"][0]["phase"] = "changed returned copy"
    assert inputs["slice_bundle"] == original


@pytest.mark.parametrize(
    "fault",
    [
        "slice_value",
        "full_value",
        "dtype",
        "one_index",
        "duplicate_index",
        "boolean_index",
        "huge_index",
        "old_origin",
        "calibration",
        "projector",
        "projection_value",
        "projection_dtype",
        "outside_projection",
        "content_hash",
        "endpoint_state",
        "endpoint_source",
        "endpoint_review",
        "accepted_match",
        "endpoint_observed",
        "link",
        "count",
    ],
)
def test_wrong_content_native_slice_origin_or_endpoint_state_rejected(inputs, fault):
    claim = inputs["slice_bundle"]
    reseal = False
    if fault == "slice_value":
        inputs["slice_geometry"]["pose_xyz"][0, 0, 0, 0] += 1
    elif fault == "full_value":
        inputs["full_geometry"]["pose_xyz"][3, 0, 0, 0] += 1
    elif fault == "dtype":
        inputs["slice_geometry"]["sources"] = inputs["slice_geometry"]["sources"].astype(np.int64)
        claim["provenance"]["candidate_arrays_sha256"]["sources"] = (
            module.constraints._array_digest(inputs["slice_geometry"]["sources"])
        )
        reseal = True
    elif fault == "one_index":
        inputs["slice_indices"][0] += 1
    elif fault == "duplicate_index":
        inputs["slice_indices"][1] = inputs["slice_indices"][0]
    elif fault == "boolean_index":
        inputs["slice_indices"] = np.ones(27, bool)
    elif fault == "huge_index":
        inputs["slice_indices"] = np.full(27, 2**64 - 1, np.uint64)
    elif fault == "old_origin":
        inputs["candidate_sha256"] = "a" * 64
    elif fault == "calibration":
        inputs["calibration_sha256"] = {"manifest": "a" * 64}
    elif fault == "projector":
        inputs["projection_origin_sha256"] = "a" * 64
    elif fault == "projection_value":
        inputs["full_projected_mouth"][5, 0, 0, 0] += 1
    elif fault == "projection_dtype":
        inputs["full_projected_mouth"] = inputs["full_projected_mouth"].astype(np.float32)
    elif fault == "outside_projection":
        inputs["full_projected_mouth"][0] = 0
    elif fault == "content_hash":
        claim["human_review_metadata"]["status"] = "changed without matching content hash"
    elif fault == "endpoint_state":
        claim["endpoints"][0]["source_state_sha256"] = "a" * 64
        reseal = True
    elif fault == "endpoint_source":
        claim["endpoints"][0]["mouth_source_rows"][0] = 2
        reseal = True
    elif fault == "endpoint_review":
        claim["endpoints"][0]["raw_identity_review"] = True
        reseal = True
    elif fault == "accepted_match":
        claim["endpoints"][3]["accepted_match"]["slot"] = 0
        reseal = True
    elif fault == "endpoint_observed":
        for data, i in ((inputs["full_geometry"], 5), (inputs["slice_geometry"], 2)):
            data["observed"][i, 0, 0] = False
        claim["provenance"]["candidate_arrays_sha256"]["observed"] = (
            module.constraints._array_digest(inputs["slice_geometry"]["observed"])
        )
        reseal = True
    elif fault == "link":
        claim["links"][0]["right_raw_slot"] = 2
        reseal = True
    else:
        claim["endpoints"].pop()
        reseal = True
    if reseal:
        digest_bundle(claim)
    with pytest.raises(ValueError):
        module.bind_slice_constraints_to_full(**inputs)


def test_sparse_anchor_projection_and_unresolved_endpoint_preserved(inputs):
    sliced = inputs["slice_geometry"]
    full = inputs["full_geometry"]
    # One actual endpoint is directly unsupported; no elimination or repair.
    for data, row in ((sliced, 9), (full, 12)):
        data["observed"][row, 0, 0] = False
        data["sources"][row, 0, :, 0] = -1
    projected = inputs["slice_projected_mouth"]
    anchor_rows = [2, 9, 22]
    projected[[i for i in range(len(projected)) if i not in anchor_rows]] = np.nan
    inputs["full_projected_mouth"][:] = np.nan
    inputs["full_projected_mouth"][inputs["slice_indices"]] = projected
    inputs["slice_bundle"] = bundle(
        sliced,
        projected=projected,
        states={"A": ("confirmed", "confirmed"), "B": ("not_visible", "ambiguous")},
    )
    expanded, proof = module.bind_slice_constraints_to_full(**inputs)
    assert expanded["endpoints"][3]["raw_slot"] is None
    assert expanded["links"][:2] == inputs["slice_bundle"]["links"][:2]
    assert all(v["status"] == "endpoint_unresolved" for v in expanded["links"][:2])
    assert expanded["links"][2]["human_status"] == "not_visible"
    assert proof["verified_endpoint_count"] == 9


def test_frozen_dense_core_accepts_expanded_scope_and_preserves_open_interval(inputs):
    full = inputs["full_geometry"]
    with pytest.raises(ValueError, match="array binding"):
        review_correction.build_review_correction(
            full, [inputs["slice_bundle"]], tracker_config=CONFIG, candidate_sha256=CANDIDATE_SHA
        )
    expanded, _ = module.bind_slice_constraints_to_full(**inputs)
    corrected = review_correction.build_review_correction(
        full, [expanded], tracker_config=CONFIG, candidate_sha256=CANDIDATE_SHA
    )
    mapping = corrected["arrays"]["corrected_identity_to_raw_slot"]
    assert mapping[5, 0] == mapping[12, 0] == 0
    assert np.all(mapping[6:12, 0] == -1)
    assert np.isnan(corrected["arrays"]["corrected_node_xyz"][6:12, 0]).all()
    assert full["identity_review"][12, 0]
    assert corrected["arrays"]["identity_origin"][12, 0] == 2
    assert expanded["endpoints"] == inputs["slice_bundle"]["endpoints"]


def test_already_expanded_bundle_cannot_be_reinterpreted_as_original_extraction(inputs):
    expanded, _ = module.bind_slice_constraints_to_full(**inputs)
    inputs["slice_bundle"] = expanded
    with pytest.raises(ValueError, match="original extractor"):
        module.bind_slice_constraints_to_full(**inputs)
