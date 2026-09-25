"""Focused new legacy-input edges; no video/model/SL payload in fixtures."""

import copy
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

WORK = Path(__file__).resolve().parents[1] / "work/three_body"
sys.path.insert(0, str(WORK))
import prepare_temporal_known_gap64 as p  # noqa: E402
from prepare_sequence import PackedMasks  # noqa: E402


def metadata():
    old = dict(
        experiment=p.native.EXPERIMENT,
        frames=list(range(p.START, p.SOURCE_STOP)),
        source_interval_half_open=[p.START, p.SOURCE_STOP],
        fps=166,
        image_shape_hw=[1200, 1200],
        raw_codec="FFV1 AVI",
        mask_schema="bbox_packbits_instances_v1",
        raw={},
        masks={},
        models={},
        runtime={"opencv": "legacy"},
        code_sha256={"producer.py": "a" * 64},
    )
    for i, v in enumerate(p.VIEWS):
        old["raw"][v] = dict(frame_bgr_sha256=["b" * 64] * 200)
        counts = [0] * 200
        counts[0] = p.INITIAL_MASK_COUNTS[i]
        counts[1] = p.WINDOW_MASK_COUNTS[i] - counts[0]
        old["masks"][v] = dict(instances_per_frame=counts)
    compact = copy.deepcopy(old)
    compact.update(raw_storage="original_source_video", storage_qa_status="verified")
    for v in p.VIEWS:
        compact["raw"][v].update(source_offset_frames=p.START, source_total_frames=348600)
    audit = dict(
        status="verified",
        frame_count=200,
        images=600,
        code_and_inputs_unchanged=True,
        source_video_full_sha256_before_after_verified=True,
        compact_manifest_sha256=p.MANIFEST_SHA,
        per_view={v: dict(images=200, masks=200, detections=200) for v in p.VIEWS},
    )
    seal = dict(
        params=dict(
            source_code_sha256=old["code_sha256"],
            threshold=0.5,
            source_interval_half_open=[p.START, p.SOURCE_STOP],
        )
    )
    return old, compact, audit, seal


def test_legacy_runtime_unknown_not_filled_with_current_defaults():
    values = metadata()
    result = p.validate_metadata(*values)
    assert result["runtime_recorded"] == {"opencv": "legacy"}
    assert result["unrecorded_runtime_fields"] == list(p.UNRECORDED_FLAGS)
    assert result["current_runtime_contract_imputed"] is False
    assert "autocast" not in result["runtime_recorded"]


@pytest.mark.parametrize(
    "fault",
    [
        "relative_offset",
        "native_order",
        "pixel_sha",
        "mask_count",
        "audit_incomplete",
        "audit_sha",
        "runtime",
        "threshold",
    ],
)
def test_legacy_storage_edges_reject_changes(fault):
    old, compact, audit, seal = metadata()
    if fault == "relative_offset":
        compact["raw"][p.VIEWS[0]]["source_offset_frames"] = 0
    elif fault == "native_order":
        compact["frames"][1] += 1
    elif fault == "pixel_sha":
        compact["raw"][p.VIEWS[0]]["frame_bgr_sha256"][7] = "c" * 64
    elif fault == "mask_count":
        old["masks"][p.VIEWS[0]]["instances_per_frame"][0] += 1
    elif fault == "audit_incomplete":
        audit["per_view"][p.VIEWS[2]]["images"] = 64
    elif fault == "audit_sha":
        audit["compact_manifest_sha256"] = "c" * 64
    elif fault == "runtime":
        compact["runtime"]["autocast"] = False
    else:
        seal["params"]["threshold"] = 0.6
    with pytest.raises(ValueError):
        p.validate_metadata(old, compact, audit, seal)


def test_unhashed_old_video_roles_are_not_fabricated_or_resolved():
    seal = dict(inputs={"top_cam_video": dict(path="unavailable-old-video.avi", size=9)})
    manifest = dict(models={})
    for kind in ("top", "side"):
        manifest["models"][kind] = {}
        for role in ("weights", "config"):
            path = "does-not-exist/" + kind + role
            manifest["models"][kind][role] = dict(path=path, sha256="a" * 64, bytes=5)
            seal["inputs"][kind + "_" + role] = dict(path=path, sha256="a" * 64, size=5)
    p.legacy_asset_roles(seal, manifest)
    assert "sha256" not in seal["inputs"]["top_cam_video"]
    seal["inputs"]["side_weights"]["sha256"] = "b" * 64
    with pytest.raises(ValueError):
        p.legacy_asset_roles(seal, manifest)


class Capture:
    def __init__(self):
        self.pos = 0
        self.reads = 0
        self.seeks = []

    def isOpened(self):
        return True

    def set(self, prop, value):
        self.pos = value
        self.seeks.append(value)
        return True

    def get(self, prop):
        return {
            cv2.CAP_PROP_FPS: 166,
            cv2.CAP_PROP_FRAME_COUNT: 348600,
            cv2.CAP_PROP_FRAME_HEIGHT: 2,
            cv2.CAP_PROP_FRAME_WIDTH: 3,
            cv2.CAP_PROP_POS_FRAMES: self.pos,
        }[prop]

    def read(self):
        self.pos += 1
        self.reads += 1
        return True, np.zeros((2, 3, 3), np.uint8)


def test_fullsource_absolute333519_not_clip_zero_and_no65():
    cap = Capture()
    digest = p.signature(np.zeros((2, 3, 3), np.uint8))["sha256"]
    result = list(p.read64(cap, [digest] * 64, shape=(2, 3)))
    assert cap.seeks == [333519] and cap.reads == 64 and cap.pos == 333583
    assert [n for n, _ in result] == list(range(333519, 333583))
    with pytest.raises(ValueError):
        list(p.read64(Capture(), [digest] * 65, shape=(2, 3)))


def test_packed_relative_index_zero_and_native_bounds():
    enc = PackedMasks(shape=(5, 7))
    for n in (333519, 333520):
        enc.append(n, np.zeros((0, 5, 7), bool), np.empty(0), np.empty((0, 4)), np.empty(0, int))
    data = enc.arrays()
    obs = p.observation(data, 333519, 0)
    assert obs["packed_frame_idx"].tolist() == [333519]
    assert obs["frame_idx"].shape == () and int(obs["frame_idx"]) == 333519
    for bad in (0, 63, 66483, 333518, 333583):
        with pytest.raises(ValueError):
            p.observation(data, bad, 0)


def test_all10_query_rows_and_native_metadata_no_old_globals(monkeypatch):
    old_start = p.prior.START
    for count in p.INITIAL_MASK_COUNTS:
        image = np.zeros((32, 32, 3), np.uint8)
        image[8:24, 8:24] = 255
        masks = np.ones((count, 32, 32), bool)
        obs = dict(
            frame_idx=np.asarray(333519, np.int64), masks=masks, mask_available=np.asarray(True)
        )
        before = p.core.select_queries(image, masks)
        arrays, rows = p.queries(image, obs)
        assert p.signature(arrays["xy"]) == p.signature(before["xy"])
        assert rows["mask_count"] == count and len(rows["rows"]) == count
        assert int(arrays["frame_idx"]) == 333519 and len(arrays["xy"]) <= 64 * count
        assert np.array_equal(arrays["mask_index"], before["mask_index"])
        obs["frame_idx"] = np.asarray(66483, np.int64)
        with pytest.raises(ValueError):
            p.queries(image, obs)
    assert p.prior.START == old_start == 66483
    assert sum(p.INITIAL_MASK_COUNTS) * 64 == 640


def test_actual_small_json_contract_no_payload_or_avi_read():
    if not (p.COMPACT / "manifest.json").exists():
        pytest.skip("workspace metadata absent")
    evidence = {}
    result = p.metadata_inputs(evidence)
    assert len(result["references"]) == 13
    assert not any(Path(k).suffix in {".npz", ".mp4", ".avi", ".pth", ".png"} for k in evidence)
    assert not any(Path(r["path"]).suffix == ".avi" for r in result["references"])
    assert (
        sum(sum(result["manifest"]["masks"][v]["instances_per_frame"][:64]) for v in p.VIEWS) == 608
    )
    assert [result["manifest"]["masks"][v]["instances_per_frame"][0] for v in p.VIEWS] == [3, 4, 3]
    assert len(result["shared"]) == 7
