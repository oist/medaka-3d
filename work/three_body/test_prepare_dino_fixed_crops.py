"""Synthetic transforms/bindings only; no real PNG, candidate or model is read."""

import copy
import json
import sys

import numpy as np
import prepare_dino_fixed_crops as m
import pytest


def counts():
    # Fixed metadata cardinalities only; no experimental pixel/feature fixture.
    values = ((4, 3, 5), (4, 4, 4), (3, 4, 4), (5, 5, 4), (3, 4, 4), (3, 3, 5))
    return {(n, v): values[i][j] for i, n in enumerate(m.FRAMES) for j, v in enumerate(m.VIEWS)}


def descriptor(mask):
    core = m.partial.eroded_core(mask)
    return dict(core_sha256=m.signature(core)["sha256"], core_pixels=int(core.sum()))


def test_exact_all71_45_700_plan_and_first_mask_without_slot_selection():
    values = counts()
    before = copy.deepcopy(values)
    plan = m.make_plan(values)
    assert values == before
    assert len(plan["crops"]) == 71 and len(plan["requests"]) == 45 and len(plan["pairs"]) == 700
    assert plan["mask_pair_counts_by_view"] == dict(zip(m.VIEWS, (200, 219, 281), strict=True))
    assert plan["maximum_forward_calls"] == 142 and plan["conditions"] == ["raw", "masked"]
    assert plan["crops"][0] == dict(
        frame_idx=66524, view="top_cam", mask_row=0, crop_id="f66524_top_cam_m000"
    )
    for request in plan["requests"]:
        pairs = [p for p in plan["pairs"] if p["request_index"] == request["request_index"]]
        expected = [
            (i, j)
            for i in range(request["earlier_mask_count"])
            for j in range(request["later_mask_count"])
        ]
        assert [(p["earlier_mask_row"], p["later_mask_row"]) for p in pairs] == expected
        assert all(p["earlier_frame"] < p["later_frame"] for p in pairs)


@pytest.mark.parametrize(
    "fault", ["missing", "extra", "zero", "one_more", "fractional", "missing_observation"]
)
def test_plan_refuses_changed_fixed_scope_instead_of_reselecting(fault):
    values = counts()
    key = next(iter(values))
    if fault == "missing":
        del values[key]
    elif fault == "extra":
        values[999, m.VIEWS[0]] = 1
    else:
        values[key] = {"zero": 0, "one_more": 5, "fractional": 4.0, "missing_observation": None}[
            fault
        ]
    with pytest.raises(ValueError):
        m.make_plan(values)


def test_crop_bgr_order_nonmutation_full_geometry_and_no_second_transform():
    bgr = np.zeros((20, 24, 3), np.uint8)
    bgr[:] = (30, 90, 210)
    mask = np.ones((20, 24), bool)
    before = bgr.tobytes(), mask.tobytes()
    arrays, metadata = m.prepared_crop(bgr, mask, descriptor(mask))
    assert before == (bgr.tobytes(), mask.tobytes())
    assert metadata["crop_status"] == "ready" and set(arrays) == set(m.ARRAY_SHAPES)
    assert metadata["image_rgb_sha256"] == m.signature(bgr[:, :, ::-1])["sha256"]
    np.testing.assert_array_equal(
        arrays["raw_rgb"][112, 112], np.array([210, 90, 30], np.float32) / 255
    )
    assert arrays["raw_normalized_chw"].flags.c_contiguous
    for key, array in arrays.items():
        assert metadata["array_signatures"][key] == m.signature(array)
        assert not np.shares_memory(array, bgr) and not np.shares_memory(array, mask)


@pytest.mark.parametrize("empty", [False, True])
def test_empty_and_zero_core_crops_kept_not_substituted(empty):
    image = np.ones((8, 8, 3), np.uint8)
    mask = np.zeros((8, 8), bool)
    if not empty:
        mask[3, 3] = True
    arrays, metadata = m.prepared_crop(image, mask, descriptor(mask))
    assert metadata["eligible_token_count"] == 0
    assert not arrays["token_eligible"].any()
    assert metadata["crop_status"] == ("empty_mask" if empty else "ready")
    for condition in m.CONDITIONS:
        assert (
            np.isnan(arrays[condition + "_normalized_chw"]).all()
            if empty
            else np.isfinite(arrays[condition + "_normalized_chw"]).all()
        )


@pytest.mark.parametrize("fault", ["core", "dtype", "missing_array", "changed_origin", "mutation"])
def test_crop_schema_or_original_input_drift_rejected(monkeypatch, fault):
    image, mask = np.zeros((8, 8, 3), np.uint8), np.ones((8, 8), bool)
    prior = descriptor(mask)
    real = m.partial.prepare_mask_crop

    def altered(rgb, current_mask):
        result = real(rgb, current_mask)
        if fault == "core":
            result["core_pixels"] += 1
        elif fault == "dtype":
            result["token_center_native_xy"] = result["token_center_native_xy"].astype(np.float32)
        elif fault == "missing_array":
            del result["token_native_xyxy"]
        elif fault == "changed_origin":
            result["image_rgb_sha256"] = "0" * 64
        else:
            current_mask[0, 0] = False
        return result

    monkeypatch.setattr(m.partial, "prepare_mask_crop", altered)
    with pytest.raises(ValueError):
        m.prepared_crop(image, mask, prior)


@pytest.fixture
def native_fixture():
    image = np.zeros((1200, 1200, 3), np.uint8)
    image[100, 100] = (1, 2, 3)
    masks = np.zeros((2, 1200, 1200), bool)
    masks[0, 90:100, 90:100] = True
    detections = np.arange(18, dtype=np.float32).reshape(3, 3, 2)
    available = np.array(True)
    record = dict(
        mask_count=2,
        bgr_sha256=m.signature(image)["sha256"],
        mask_sha256=m.signature(masks)["sha256"],
        detection_signature=m.signature(detections),
    )
    descriptors = dict(
        all_mask_count=2,
        masks=[dict(mask_row=i, mask_sha256=m.signature(a)["sha256"]) for i, a in enumerate(masks)],
    )
    return image, masks, available, detections, record, descriptors


def test_native_original_bgr_empty_mask_row_and_sl_are_verified(native_fixture):
    m.check_native(*native_fixture)
    assert len(native_fixture[1]) == 2 and not native_fixture[1][1].any()


@pytest.mark.parametrize(
    "fault", ["bgr", "mask_order", "sl", "sl_dtype", "unavailable", "descriptor_order"]
)
def test_native_pixel_mask_or_row_drift_rejected(native_fixture, fault):
    image, masks, available, detections, record, descriptors = native_fixture
    if fault == "bgr":
        image[0, 0, 0] = 1
    elif fault == "mask_order":
        masks = masks[::-1]
    elif fault == "sl":
        detections[0, 0, 0] += 1
    elif fault == "sl_dtype":
        detections = detections.astype(np.float64)
    elif fault == "unavailable":
        available = np.array(False)
    else:
        descriptors["masks"].reverse()
    with pytest.raises(ValueError):
        m.check_native(image, masks, available, detections, record, descriptors)


@pytest.fixture
def seal_fixture(tmp_path):
    root = tmp_path / "prior"
    root.mkdir()
    payload = root / "payload.npz"
    payload.write_bytes(b"synthetic sealed bytes; no NumPy data read")
    item = dict(path=str(payload), sha256=m.assets.sha(payload), size=payload.stat().st_size)
    seal = dict(
        stage="fixture",
        status="ok",
        finished_at="2026-09-10T00:00:00+00:00",
        inputs={},
        outputs={"payload.npz": item},
    )
    path = root / "seal.json"
    path.write_text(json.dumps(seal))
    return root, path, seal


def test_pinned_all_output_bytes_and_prior_input_membership(seal_fixture):
    root, path, _ = seal_fixture
    evidence = {}
    result = m.verify_seal(root, path.name, "fixture", m.assets.sha(path), evidence)
    assert len(evidence) == 2
    with pytest.raises(ValueError, match="membership"):
        m.member(result, path, evidence)
    result["inputs"][str(path)] = dict(path=str(path), sha256=m.assets.sha(path))
    assert m.member(result, path, evidence) == path
    assert m.sealed_output(root, result, "payload.npz", evidence) == root / "payload.npz"


@pytest.mark.parametrize(
    "fault", ["payload", "role", "failed", "unsealed", "wrong_stage", "seal_sha"]
)
def test_prior_completed_evidence_drift_rejected(seal_fixture, fault):
    root, path, seal = seal_fixture
    if fault == "payload":
        (root / "payload.npz").write_bytes(b"changed")
    elif fault == "role":
        seal["outputs"]["other.npz"] = seal["outputs"].pop("payload.npz")
        (root / "other.npz").write_bytes(b"synthetic sealed bytes; no NumPy data read")
    elif fault == "failed":
        seal["status"] = "failed"
    elif fault == "unsealed":
        seal.pop("finished_at")
    elif fault == "wrong_stage":
        seal["stage"] = "other"
    path.write_text(json.dumps(seal))
    digest = "0" * 64 if fault == "seal_sha" else m.assets.sha(path)
    with pytest.raises(ValueError):
        m.verify_seal(root, path.name, "fixture", digest, {})


def test_digest_conflict_and_final_guard_cannot_be_silently_rebound(tmp_path):
    path = tmp_path / "evidence"
    path.write_bytes(b"original")
    evidence = {str(path): m.assets.sha(path)}
    path.write_bytes(b"replacement")
    with pytest.raises(ValueError, match="conflicts"):
        m.bind(path, m.assets.sha(path), evidence)
    with pytest.raises(ValueError, match="final evidence"):
        m.final_guard(evidence)


def test_import_has_no_model_and_runtime_only_reads_package_metadata():
    before = set(sys.modules)
    m.no_model_imports()
    assert m.runtime()["model_imported"] is False
    assert not any(n == "torch" or n.startswith("dinov2") for n in set(sys.modules) - before)


def test_existing_output_is_refused_before_input_io(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "WORK", tmp_path)
    output = tmp_path / "exists"
    output.mkdir()
    monkeypatch.setattr(m, "load_inputs", lambda *_: pytest.fail("must not read inputs"))
    with pytest.raises(ValueError, match="new workspace"):
        m.run(output)


def test_fixed_rows_reject_duplicate_even_with_eighteen_rows():
    rows = [dict(frame_idx=n, view=v) for n in m.FRAMES for v in m.VIEWS]
    assert len(m.fixed_rows(rows)) == 18
    rows[-1] = rows[0]
    with pytest.raises(ValueError, match="exact18"):
        m.fixed_rows(rows)


@pytest.fixture
def source_contract():
    raw_summary = dict(status="ok", horizon_stop=68060, frames=list(m.FRAMES))
    # Actual identity summary schema has no horizon_stop; do not invent it.
    identity_summary = dict(
        status="ok", frames=list(m.FRAMES), methods=["directional", "bounded"], views=list(m.VIEWS)
    )
    inputs = {
        str((m.AUDIT / "audit.provenance.json").resolve()): dict(
            path=str((m.AUDIT / "audit.provenance.json").resolve()), sha256=m.AUDIT_SHA
        )
    }
    for index, method in enumerate(("directional", "bounded")):
        path = (m.AUDIT / f"{method}_same68060_candidate.npz").resolve()
        inputs[str(path)] = dict(path=str(path), sha256=str(index) * 64)
    return (
        raw_summary,
        identity_summary,
        dict(inputs=copy.deepcopy(inputs)),
        dict(inputs=copy.deepcopy(inputs)),
    )


def test_real_schema_without_identity_horizon_uses_both_sealed_same_candidates(source_contract):
    before = copy.deepcopy(source_contract)
    result = m.validate_source_contract(*source_contract)
    assert len(result) == 3 and source_contract == before


@pytest.mark.parametrize(
    "fault",
    [
        "missing_candidate",
        "candidate_digest",
        "audit_digest",
        "raw_horizon",
        "identity_frames",
        "method",
    ],
)
def test_same_horizon_lineage_cannot_drift(source_contract, fault):
    raw_summary, identity_summary, raw, identity = source_contract
    paths = list(identity["inputs"])
    if fault == "missing_candidate":
        del identity["inputs"][paths[-1]]
    elif fault == "candidate_digest":
        identity["inputs"][paths[-1]]["sha256"] = "f" * 64
    elif fault == "audit_digest":
        raw["inputs"][paths[0]]["sha256"] = identity["inputs"][paths[0]]["sha256"] = "f" * 64
    elif fault == "raw_horizon":
        raw_summary["horizon_stop"] = 76360
    elif fault == "identity_frames":
        identity_summary["frames"][-1] += 1
    else:
        identity_summary["methods"][-1] = "tail"
    with pytest.raises(ValueError):
        m.validate_source_contract(*source_contract)
