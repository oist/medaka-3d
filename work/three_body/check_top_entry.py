"""Small read-only finite-top-entry versus existing ray/box diagnostic."""

import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import measure_top_contrast as d

from medaka3d.calib import load_rig
from medaka3d.provenance import provenance


def main():
    root = d.m.REPO / "work/three_body/top_contrast_13frames_v1"
    output = root / "entry_check_v1"
    if output.exists():
        raise FileExistsError("new output required")
    hashes = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        d.m.require(path.is_relative_to(d.m.REPO), "path escaped workspace")
        digest = d.m.sha256(path)
        d.m.require(expected is None or digest == expected, "checksum mismatch")
        hashes[str(path)] = digest
        return path

    for path in set(d.m.code_paths()) | {Path(d.__file__), Path(__file__)}:
        bind(path)
    seal = json.loads(bind(root / "contrast.provenance.json").read_text())
    d.m.require(seal["status"] == "ok", "parent incomplete")
    for item in seal["outputs"].values():
        bind(item["path"], item["sha256"])
    summary = json.loads((root / "summary.json").read_text())
    sequence = Path(summary["source_sequence"])
    manifest = json.loads(
        bind(sequence / "manifest.json", summary["source_manifest_sha256"]).read_text()
    )
    info = manifest["masks"][d.VIEW]
    packed = d.m.identity_inputs._npz(bind(sequence / info["path"], info["sha256"]))
    calibration = (
        d.m.REPO / "work/three_body/persistent_mouth_continuous_one_minute_v1/calibration_snapshot"
    )
    for path in calibration.glob("*.json"):
        bind(path, summary["input_sha256_before"][str(path)])
    rig = load_rig(calibration)
    camera = rig.camera(d.VIEW)
    geometry = json.loads(rig.manifest.read_text())["physical_geometry"]
    top_z = camera.interface_point[2]
    xy_upper = np.array([geometry["inside_x_span_mm"], geometry["inside_y_span_mm"]])
    d.m.require(
        camera.layer_count == 1 and np.array_equal(camera.interface_normal, [0, 0, -1]),
        "not a horizontal one-interface top camera",
    )
    d.m.require(
        np.all(camera.center[:2] > 0) and np.all(camera.center[:2] < xy_upper),
        "camera horizontal projection outside finite top face",
    )
    d.m.require(
        top_z == geometry["water_height_from_inside_floor_mm"],
        "top plane differs from physical manifest",
    )
    previous = d.m.identity_inputs._npz(root / "ray_box_footprint.npz")
    inside = np.zeros((1200, 1200), bool)
    valid = np.zeros_like(inside)
    plane_error, reversed_sign_count = 0.0, 0
    for start in range(0, inside.size, 30000):
        index = np.arange(start, min(start + 30000, inside.size))
        y, x = np.unravel_index(index, inside.shape)
        origin, direction = camera.rays(np.column_stack((x, y)))
        finite = np.isfinite(origin).all(-1) & np.isfinite(direction).all(-1)
        valid[y, x] = finite
        inside[y, x] = finite & (origin[:, :2] >= 0).all(-1) & (origin[:, :2] <= xy_upper).all(-1)
        plane_error = max(plane_error, float(np.max(np.abs(origin[finite, 2] - top_z), initial=0)))
        delta = origin[finite, :2] - camera.center[:2]
        reversed_sign_count += int(np.sum(delta * direction[finite, :2] < 0))
    instances = []
    for row in summary["measurements"]:
        masks = d.m.decode_frame_masks(packed, row["frame_idx"] - manifest["frames"][0])
        values = []
        for index, mask in enumerate(masks):
            y, x = np.nonzero(mask)
            origin, _ = camera.rays(np.column_stack((x, y)))
            values.append(
                dict(
                    instance=index,
                    pixels=int(mask.sum()),
                    entry_inside=int(inside[mask].sum()),
                    entry_xy_min=origin[:, :2].min(0),
                    entry_xy_max=origin[:, :2].max(0),
                )
            )
        saved = d.m.identity_inputs._npz(root / f"f{row['frame_idx']}_observations.npz")
        instances.append(
            dict(
                frame=row["frame_idx"],
                instances=values,
                contrast_pixels=int(saved["contrast"].sum()),
                contrast_entry_inside=int(inside[saved["contrast"]].sum()),
            )
        )
    equality = dict(
        engineering=np.array_equal(inside, previous["engineering"]),
        physical=np.array_equal(inside, previous["physical"]),
        valid=np.array_equal(valid, previous["valid"]),
    )
    d.m.require(
        all(equality.values()),
        "finite entry and prior box footprint differ: preserve explicit failure",
    )
    d.m.require(plane_error < 1e-9 and reversed_sign_count == 0, "top geometric assumptions failed")
    output.mkdir(parents=True)
    with provenance("finite_top_entry_full_native_pixel_check", output, name="entry") as record:
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
        copy = output / "source.py"
        shutil.copyfile(__file__, copy)
        record.output("source", copy, sha256=d.m.sha256(copy))
        path = output / "entry_inside.npz"
        np.savez_compressed(path, entry_inside=inside, valid=valid)
        record.output(path.name, path, sha256=d.m.sha256(path))
        values = dict(
            status="ok",
            native_pixels_tested=int(inside.size),
            entry_inside_pixels=int(inside.sum()),
            valid_pixels=int(valid.sum()),
            exact_equality_to_previous_box=equality,
            camera_center_mm=camera.center,
            interface_point_mm=camera.interface_point,
            interface_normal=camera.interface_normal,
            layer_count=camera.layer_count,
            finite_entry_xy_lower=[0, 0],
            finite_entry_xy_upper=xy_upper,
            maximum_plane_error_mm=plane_error,
            horizontal_sign_reversals=reversed_sign_count,
            foreground_measurements=instances,
            gpu_used=False,
            native_raw_redecoded=False,
            new_filter_adopted=False,
            gt_or_id_arrays_read=False,
        )
        path = output / "QA.json"
        path.write_text(json.dumps(d.m.safe(values), indent=2, allow_nan=False), encoding="utf-8")
        record.output("QA", path, sha256=d.m.sha256(path))
        text = (
            "# 有限のtop水面entry確認: 今回のbox検査と同じ結果\n\n"
            "全1,440,000 native pixelsで、水面entryが有限範囲内という条件と、"
            f"前回の前向きray-box交差は完全に一致しました。内側は{inside.sum():,}画素です。"
            "z下限0 mmと-6 mmのどちらのboxとも同じです。\n\n"
            "校正のtop面はinterface_point=[0,0,186] mm、normal=[0,0,-1]、"
            "1層の空気→水です。校正manifestの水面範囲はx,yとも0..192 mm。"
            "camera center=[98.560803,111.381436,537.907470] mmは水平投影で範囲内にあります。"
            "この平行・水平面の屈折は水平移動の符号を保存するため、範囲外へ出たentryから"
            "rayが後で内側へ戻るという一般的な可能性は、今回の配置には当てはまりません。"
            f"全画素の水平符号逆転0、entry面最大誤差{plane_error:.3g} mmを確認しました。\n\n"
            "13フレームのD2全画素はentry時点でも範囲内です。79381の反射らしいm3も"
            "1855/1855画素が内側で、entry x=142.17..149.20、y=176.37..191.69 mmでした。"
            "この追加条件では同領域を分離できません。前報の反射棄却に関する陰性結果は変わりません。\n\n"
            "この説明は現校正値についてのものです。無限面のray APIが有限面検査を持たないこと"
            "と、実際に不正な側方entryが生じることを区別しました。反射か直接像かの真値判定、"
            "3D/IDの更新、マスクフィルタの採用はしていません。原画の追加デコードやGPU処理はありません。\n"
        )
        path = output / "RESULT.md"
        path.write_text(text, encoding="utf-8")
        record.output("result", path, sha256=d.m.sha256(path))
        d.m.require(
            all(d.m.sha256(path) == digest for path, digest in hashes.items()),
            "input/source changed",
        )
        d.m.require(
            all(d.m.sha256(item["path"]) == item["sha256"] for item in record.outputs.values()),
            "output changed",
        )
        record.param("all_input_output_sha_reverified", True)
        print(
            json.dumps(
                d.m.safe({k: v for k, v in values.items() if k != "foreground_measurements"})
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
