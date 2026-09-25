"""One isolated synthetic six-capture run; source headers are an explicit stub.

The actual complete-body selector, variable-mask observation preparation,
containment measurements, compact records and output NPZ roundtrip run on CPU.
No original archive, calibration, image, video or scientific metadata is read.
"""

import copy
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import run_nodewise_body_mask_containment as m

NP_LOAD = np.load


@pytest.fixture(autouse=True)
def isolation(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("isolated caller test reached an original input reader")

    for module in (m, m.saved, m.frozen, m.proof):
        for name in ("REPO", "WORK"):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, tmp_path)
    monkeypatch.setattr(m, "GEOMETRY", tmp_path / "unused_geometry")
    monkeypatch.setattr(m, "load_headers", forbidden)
    monkeypatch.setattr(m, "source_code", forbidden)
    monkeypatch.setattr(m, "load_rig", forbidden)
    monkeypatch.setattr(m.saved, "load_headers", forbidden)
    monkeypatch.setattr(m.frozen, "capture_masks", forbidden)
    monkeypatch.setattr(m.prior.prepare_sequence, "decode_frame_masks", forbidden)
    monkeypatch.setattr(m.prior.cv2, "VideoCapture", forbidden)
    monkeypatch.setattr(np, "load", forbidden)


def test_six_captures_all_complete_bodies_and_actual_output_roundtrip(monkeypatch, tmp_path):
    output = tmp_path / "output"
    source = tmp_path / "synthetic_source.py"
    source.write_text("# synthetic source/header trust boundary\n", encoding="utf-8")
    body_pose = np.array([[2.0, 4.0, 1.0], [6.0, 4.0, 1.0], [14.0, 4.0, 1.0]])
    context = dict(cases={}, rig_refs={}, geometry_plan={"source_code": {}})
    specifications, mask_expected = {}, {}
    for frame, start, size, legacy_count, complete in (
        (82720, 81340, 20, 10, (0, 18, 19)),
        (90258, 89640, 10, 9, (0, 9)),
    ):
        pool = []
        for index in range(size):
            pose = body_pose.copy()
            measured = index in complete
            if not measured:
                pose[2] = np.nan
            pool.append(
                dict(
                    pool_index=index,
                    pose_xyz=pose,
                    triangulated_nodes=[True, True, measured],
                )
            )
        geometry = dict(
            frame_idx=frame,
            pool=pool,
            summary=dict(
                legacy_candidate_count=legacy_count,
                nodewise_scenes={"complete_bodies": len(complete)},
            ),
        )
        case = dict(
            geometry=geometry,
            observations={},
            points={},
            chunk={"start": start, "stop": start + 1660},
            captured={},
            geometry_case={"synthetic": True},
            visual_evidence={"synthetic": True},
        )
        context["cases"][frame] = case
        for vi, view in enumerate(m.VIEWS):
            if vi == 0:
                masks = np.zeros((3, 12, 20), bool)
                masks[1, 3:6, 1:17] = True  # original index0 is empty, retained best is1
                masks[2, 11, 19] = True
            elif vi == 1:
                masks = np.zeros((2, 12, 20), bool)
                masks[0, 3:6, 1:9] = True
                masks[1, 3:6, 9:16] = True
            else:
                masks = np.empty((0, 12, 20), bool)
            key = f"f{frame}_{view}"
            mask_expected[key] = masks.copy()
            # No SL row matches these masks. Containment must still use every
            # nonempty mask, while preserving incompatibility as separate data.
            points = np.full((1, 3, 2), 100.0, np.float32)
            case["points"][view] = points
            digest = m.prior.prepare_sequence.array_sha256(masks)
            available = vi != 2
            arrays = dict(
                frame_idx=np.arange(start, start + 1660, dtype=np.int64),
                view_available=np.full(1660, available, bool),
                frame_mask_sha256=np.full(1660, digest.encode(), dtype="S64"),
            )
            path = tmp_path / f"synthetic_{key}.npz"
            path.write_bytes(b"not-an-NPZ; only the synthetic capture spy may consume this")
            record = dict(
                path=str(path),
                sha256=m.proof.sha(path),
                arrays={name: m.proof.signature(a) for name, a in arrays.items()},
                file_stat=m.frozen.observations.file_stat(path),
            )
            case["captured"][view] = record
            case["observations"][view] = dict(
                mask_sha256=digest, mask_available=available, mask_count=len(masks)
            )
            specifications[path] = dict(
                key=key, arrays=arrays, record=record, masks=masks, position=frame - start
            )

    header_calls, captures, decodes, released, output_loads = [], [], [], [], []

    def synthetic_headers(evidence):
        header_calls.append(True)
        for case in context["cases"].values():
            case["bodies"] = m.complete_bodies(case["geometry"])
        return context

    monkeypatch.setattr(m, "load_headers", synthetic_headers)
    monkeypatch.setattr(m, "source_code", lambda c, e: {str(source): e.add(source)})
    rig = SimpleNamespace(camera=lambda view: SimpleNamespace(project=lambda xyz: xyz[..., :2]))
    monkeypatch.setattr(m, "load_rig", lambda c, e: rig)

    class Packed(dict):
        pass

    def capture(path, digest, frames):
        if released:
            assert released[-1]() is None, "previous whole-chunk container survived"
        path = Path(path).resolve()
        assert path.is_relative_to(tmp_path) and path in specifications
        spec = specifications[path]
        assert digest == spec["record"]["sha256"]
        np.testing.assert_array_equal(frames, spec["arrays"]["frame_idx"])
        captures.append(path)
        packed = Packed({k: a.copy() for k, a in spec["arrays"].items()})
        packed["synthetic_key"] = path
        released.append(weakref.ref(packed))
        return packed, copy.deepcopy(spec["record"])

    def decode(packed, position):
        path = packed["synthetic_key"]
        spec = specifications[path]
        assert position == spec["position"]
        decodes.append((path, position))
        return spec["masks"].copy()

    def output_only_load(path, *args, **kwargs):
        path = Path(path).resolve()
        assert path == output / "frame_masks.npz" and path.is_relative_to(tmp_path)
        output_loads.append(path)
        return NP_LOAD(path, *args, **kwargs)

    monkeypatch.setattr(m.frozen, "capture_masks", capture)
    monkeypatch.setattr(m.prior.prepare_sequence, "decode_frame_masks", decode)
    monkeypatch.setattr(np, "load", output_only_load)
    m.run(output)

    assert header_calls == [True] and len(captures) == len(set(captures)) == 6
    assert [pos for _, pos in decodes] == [1380] * 3 + [618] * 3
    assert [p for p, _ in decodes] == captures and all(ref() is None for ref in released)
    assert len(output_loads) == 1  # producer reads only its newly written six-key NPZ
    summary = m.proof.read_json(output / "summary.json")
    assert summary["mask_archive_captures"] == summary["target_mask_decodes"] == 6
    assert not summary["SL_compatibility_filters_masks"] and not summary["physical_ID_truth"]
    for result, complete in zip(summary["cases"], ((0, 18, 19), (0, 9)), strict=True):
        assert result["complete_body_count"] == len(complete)
        assert result["legacy_complete_count"] == 1
        assert result["additional_complete_count"] == len(complete) - 1
        case = m.proof.read_json(output / f"cases/f{result['frame_idx']}.json")
        assert case["pool_indices"] == list(complete)
        assert set(case["excluded_incomplete_pool_indices"]) == (
            set(range(len(context["cases"][result["frame_idx"]]["geometry"]["pool"])))
            - set(complete)
        )
        first, second, missing = [case["views"][v] for v in m.VIEWS]
        assert first["containment"]["mask_indices"] == [1, 2]
        assert first["SL_row_mask_compatibility"]["compatible"] == [[False, False]]
        for row in result["body_metrics"]:
            assert row["views"][m.VIEWS[0]]["best_single_mask"] == 1
            assert row["views"][m.VIEWS[0]]["single_mask_max_distance_px"] == 0
            assert row["views"][m.VIEWS[1]]["single_mask_max_distance_px"] == 6
            assert row["views"][m.VIEWS[1]]["union_max_distance_px"] == 0
            assert row["views"][m.VIEWS[2]] == {"available": False}
        assert second["containment"]["mask_indices"] == [0, 1]
        assert missing["containment"]["available"] is False
        assert "single_mask_max_distance_px" not in missing["containment"]
    with NP_LOAD(output / "frame_masks.npz", allow_pickle=False) as archive:
        assert set(archive.files) == set(mask_expected) and len(archive.files) == 6
        for key, expected in mask_expected.items():
            np.testing.assert_array_equal(archive[key], expected)
    seal = m.proof.read_json(output / "measurement.provenance.json")
    assert seal["status"] == "ok" and len(seal["outputs"]) == 6
    assert set(seal["inputs"]) == {str(source), *(str(p) for p in specifications)}
