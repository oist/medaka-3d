"""Independent CPU checksum and first-frame geometric replay; no tracking/GT."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from medaka3d.calib import load_rig, triangulate  # noqa: E402
from medaka3d.provenance import provenance  # noqa: E402
from medaka3d.types import VIEWS  # noqa: E402

SOURCE = REPO / "work/three_body/long_seed_registration_49800_v1"
OUTPUT = REPO / "work/three_body/long_seed_registration_independent_audit_v1"
REGISTRATION_SHA = "34da64c0c61a4c4e589b3e38a01627baff0e934b1d1a4325a280645e8a132b40"
SEAL_SHA = "19854185523cc760a0f31857625678607a8f104a64c05989891b271ebaa01b6b"
FRAME = 49800


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(values):
    for path, digest in values.items():
        if sha(path) != digest:
            raise ValueError(f"changed audit evidence: {path}")


def rows_only(path, member, start, count):
    with zipfile.ZipFile(path) as archive, archive.open(member + ".npy") as stream:
        version = np.lib.format.read_magic(stream)
        header = {
            (1, 0): np.lib.format.read_array_header_1_0,
            (2, 0): np.lib.format.read_array_header_2_0,
        }[version]
        shape, fortran, dtype = header(stream)
        assert not fortran and not dtype.hasobject
        assert 0 <= start < shape[0] and 1 <= count <= shape[0] - start
        stride = int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize
        remaining = start * stride
        while remaining:
            skipped = stream.read(min(remaining, 1024 * 1024))
            assert skipped
            remaining -= len(skipped)
        data = stream.read(count * stride)
        assert len(data) == count * stride
        return np.frombuffer(data, dtype=dtype).reshape(count, *shape[1:]).copy(), shape


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    record_path = SOURCE / "registration.provenance.json"
    registration_path = SOURCE / "registration.json"
    assert sha(record_path) == SEAL_SHA and sha(registration_path) == REGISTRATION_SHA
    record = json.loads(record_path.read_text())
    registration = json.loads(registration_path.read_text())
    assert record["status"] == "ok"
    hashes = {str(record_path): SEAL_SHA, str(registration_path): REGISTRATION_SHA}
    for entry in (*record["inputs"].values(), *record["outputs"].values()):
        path, digest = str(Path(entry["path"]).resolve()), entry["sha256"]
        assert path not in hashes or hashes[path] == digest
        hashes[path] = digest
    hashes[str(Path(__file__).resolve())] = sha(__file__)
    verify(hashes)
    for relative, digest in registration["code_hashes"].items():
        assert sha(REPO / relative) == digest
        assert sha(SOURCE / "code_snapshot" / relative) == digest
    calibration_root = SOURCE / "input_snapshot/calibration"
    calibration = json.loads((calibration_root / "calibration_manifest.json").read_text())
    for model in calibration["model_files"]:
        assert sha(calibration_root / model["path"]) == model["sha256"]
    rig = load_rig(calibration_root)
    with np.load(SOURCE / "seed.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    keys = {
        "frame_idx",
        "pose_xyz",
        "source_rows",
        "sources",
        "observed",
        "projected_uv",
        "selected_points_uv",
        "ray_distance_mm",
        "reprojection_error_px",
        "direct_view",
        "segment_lengths_mm",
    } | {"sleap_points_" + view for view in VIEWS}
    assert set(arrays) == keys and arrays["frame_idx"].tolist() == [FRAME]
    pose = arrays["pose_xyz"]
    assert pose.shape == (3, 3, 3) and pose.dtype == np.float64
    assert np.load(SOURCE / "initial_pose_xyz.npy", allow_pickle=False).tobytes() == pose.tobytes()
    prior = json.loads((SOURCE / "input_snapshot/prior_audit.json").read_text())
    assert np.asarray(prior["seed_pose_xyz"], np.float64).tobytes() == pose.tobytes()
    assert np.asarray(registration["pose_xyz"], np.float64).tobytes() == pose.tobytes()
    baseline = next(
        Path(path)
        for path in registration["input_hashes"]
        if Path(path).name.startswith("C34-label")
    )
    source, baseline_shape = rows_only(baseline, "source", FRAME, 1)
    available, slot_shape = rows_only(baseline, "slot_track", FRAME, 1)
    assert baseline_shape == (348600, 3, 3) and slot_shape == (348600, 3)
    assert np.array_equal(source[0], arrays["source_rows"]) and (available >= 0).all()
    c01_records = []
    for view in VIEWS:
        path = next(
            Path(path)
            for path in registration["input_hashes"]
            if "c01_s16_predictions" in path and path.endswith("__" + view + ".npz")
        )
        with np.load(path, allow_pickle=False) as cache:
            # Native timestamp metadata only; later point coordinates are not loaded.
            native = cache["frame_idx"]
            assert (
                str(cache["view"]) == view
                and str(cache["experiment"]) == registration["experiment"]
            )
        assert np.all(np.diff(native) >= 0)
        lo, hi = (
            int(np.searchsorted(native, FRAME)),
            int(np.searchsorted(native, FRAME, side="right")),
        )
        points, _ = rows_only(path, "points", lo, hi - lo)
        assert points.dtype == arrays["sleap_points_" + view].dtype
        assert points.tobytes() == arrays["sleap_points_" + view].tobytes()
        c01_records.append(dict(view=view, points=hi - lo, first_row=lo, stop_row=hi))
    selected = np.stack(
        [arrays["sleap_points_" + view][source[0, :, vi]] for vi, view in enumerate(VIEWS)], axis=1
    )
    assert selected.tobytes() == arrays["selected_points_uv"].tobytes()
    rays = [rig.camera(view).rays(selected[:, vi]) for vi, view in enumerate(VIEWS)]
    origins = np.stack([r[0] for r in rays], axis=-2)
    directions = np.stack([r[1] for r in rays], axis=-2)
    exact, rms = triangulate(origins, directions, valid=np.ones((3, 3, 3), bool))
    assert exact.tobytes() == pose.tobytes()
    # Separate stacked least-squares formulation: no producer skeleton wrapper.
    least_squares = np.empty_like(pose)
    for fish in range(3):
        for node in range(3):
            projection = (
                np.eye(3) - directions[fish, node, :, :, None] * directions[fish, node, :, None, :]
            )
            matrix = projection.reshape(9, 3)
            target = np.einsum("vij,vj->vi", projection, origins[fish, node]).reshape(9)
            xyz, _, rank, _ = np.linalg.lstsq(matrix, target, rcond=None)
            assert rank == 3
            least_squares[fish, node] = xyz
    np.testing.assert_allclose(least_squares, pose, rtol=0, atol=1e-10)
    # Restore the frozen float32 assigned-ray cache convention before support scoring.
    origin32 = np.swapaxes(origins, 1, 2).astype(np.float32)
    direction32 = np.swapaxes(directions, 1, 2).astype(np.float32)
    delta = pose[:, None] - origin32
    distance = np.linalg.norm(
        delta - (delta * direction32).sum(-1, keepdims=True) * direction32, axis=-1
    )
    projected = np.stack([rig.camera(view).project(pose) for view in VIEWS], axis=1)
    pixel_error = np.linalg.norm(projected - selected, axis=-1)
    direct = (distance <= 2.0) & np.isfinite(pixel_error) & (pixel_error <= 12.0)
    plan = np.repeat(source[0, ..., None], 3, axis=-1)
    strict_source = np.where(direct, plan, -1).astype(np.int32)
    for key, expected in (
        ("projected_uv", projected),
        ("ray_distance_mm", distance),
        ("reprojection_error_px", pixel_error),
        ("direct_view", direct),
        ("sources", strict_source),
        ("observed", direct.sum(1) >= 2),
        ("segment_lengths_mm", np.linalg.norm(np.diff(pose, axis=1), axis=-1)),
    ):
        assert (
            arrays[key].dtype == expected.dtype and arrays[key].tobytes() == expected.tobytes()
        ), key
    assert direct.all() and arrays["observed"].all()
    assert np.all(pose >= [0, 0, -6]) and np.all(pose <= [192, 192, 186])
    violations = np.argwhere(pose[..., 2] < 0).tolist()
    assert violations == registration["physical_floor_z0_violations"] == [[0, 0]]
    files = {str(path.resolve()) for path in SOURCE.rglob("*") if path.is_file()}
    declared = {str(Path(entry["path"]).resolve()) for entry in record["outputs"].values()}
    assert files == declared | {str(record_path.resolve())}
    summary = dict(
        schema="independent_first_frame_seed_audit_v1",
        producer_registration_sha256=REGISTRATION_SHA,
        producer_seal_sha256=SEAL_SHA,
        input_source_entries_verified=len(record["inputs"]),
        producer_outputs_verified=len(record["outputs"]),
        all_code_snapshot_sha_equal=True,
        first_frame_only_numeric_export=True,
        native_frames_exported=[FRAME],
        baseline_members_deserialized=["source.npy", "slot_track.npy"],
        baseline_rows_deserialized=[FRAME],
        c01_points_deserialized=c01_records,
        baseline_coordinates_opened=False,
        human_truth_opened=False,
        replay_common_triangulation_bitwise_equal=True,
        independent_stacked_lstsq_max_abs_mm=float(np.abs(least_squares - pose).max()),
        support_projected_and_ray_error_arrays_bitwise_equal=True,
        direct_views_per_node=direct.sum(1).tolist(),
        ray_rms_mm=rms.tolist(),
        maximum_ray_distance_mm=float(distance.max()),
        maximum_reprojection_error_px=float(pixel_error.max()),
        minimum_z_mm=float(pose[..., 2].min()),
        nominal_floor_violations=violations,
        engineering_bounds_passed=True,
        surveyed_floor_or_biological_identity_certified=False,
        calibration_guardrail=calibration["guardrail"],
        source_scope_notes=[
            "The selected C34 source row inherits whole-recording optimization; this seed is optimistic.",
            "The old audit materialized source/slot-track members then selected frame 49800; neither its result nor this export contains later coordinates.",
            "Registration streams only selected baseline rows. ZIP seek may decode skipped bytes; this is not a claim that only one row of compressed bytes was read.",
            "Catalog/source snapshots contain future file names and code, not future measurements or human labels.",
            "No raw image, independent identity truth, tracking run, or physical floor survey was added by this audit.",
        ],
    )
    OUTPUT.mkdir()
    with provenance("independent_long_seed_registration_audit", OUTPUT, name="audit") as audit:
        for path, digest in hashes.items():
            audit.input(path, path, sha256=digest)
        local = OUTPUT / "audit_source.py"
        shutil.copyfile(__file__, local)
        audit.output("source", local, sha256=sha(local))
        report = OUTPUT / "summary.json"
        report.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        audit.output("summary", report, sha256=sha(report))
        report = OUTPUT / "RESULT.md"
        report.write_text(
            "49800の初期姿勢登録は、独立CPU再計算と来歴確認を通過しました。\n\n"
            f"- 元の入力・ソース{len(record['inputs'])}件、出力{len(record['outputs'])}件と全snapshotのSHAが一致。登録・成功記録の指定SHAも一致。\n"
            "- 出力数値は49800の9ノードと同フレームの3視点SLEAPだけ。後続の姿勢・個体対応列・人手正解は含みません。\n"
            "- 元C34のsource/slot_trackから49800の行だけを別のストリーム読込で照合。SLEAPも当該フレームのpoint行のみ照合しました。\n"
            "- 保存SLと校正からの三角測量は旧auditとbitwise一致。独立の積み重ね最小二乗法も最大差"
            f"{summary['independent_stacked_lstsq_max_abs_mm']:.3g} mmで一致しました。\n"
            f"- 全9ノードが3視点支持。最大ray距離{distance.max():.9f} mm、最大再投影誤差{pixel_error.max():.9f} px。元の2 mm/12 px条件を満たします。\n\n"
            "S0の口のz=-0.413303356 mmは、そのまま保存されています。既存の工学境界z≥-6 mmには収まりますが、"
            "名目上の床z=0より下です。座標の切詰めや校正変更はなく、実際の床位置への適合を確認した結果ではありません。"
            "校正manifestも、訓練キューブで座標系を定めており、絶対位置の測量精度を保証しないと明記しています。\n\n"
            "初期source対応は全動画のC34最適化に由来するため、楽観的な初期化という制約が残ります。"
            "3視点の幾何整合は生物IDの証明ではなく、30分追跡の成功も意味しません。"
            "旧auditはsource/slot_track全memberを配列化してから49800を選択していました。"
            "新登録の行限定読込と、過去に一切後続情報がメモリへ入らなかったという主張は区別します。\n",
            encoding="utf-8",
        )
        audit.output("report", report, sha256=sha(report))
        verify(hashes)
        verify({entry["path"]: entry["sha256"] for entry in audit.outputs.values()})
        audit.param("all_input_source_output_sha_unchanged", True)
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "direct_views": direct.sum(1).tolist(),
                "lstsq_max_abs_mm": summary["independent_stacked_lstsq_max_abs_mm"],
            }
        )
    )


if __name__ == "__main__":
    main()
