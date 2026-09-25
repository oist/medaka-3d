"""Project fixed nominal floor references onto sealed old-minute median PNGs."""

# ruff: noqa: RUF001
# Japanese report punctuation and multiplication sign are intentional.

import hashlib
import json
import shutil
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from medaka3d.calib import load_rig
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "work/three_body"
RUN = BASE / "persistent_mouth_continuous_one_minute_v1"
REGISTRATION = BASE / "reflection_local_probe_registration_v1"
OUT = BASE / "nominal_floor_projection_v1"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    require(not OUT.exists(), "new output required")
    hashes = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        require(path.is_relative_to(ROOT), "workspace evidence required")
        digest = sha(path)
        require(expected is None or digest == expected, f"SHA mismatch: {path}")
        require(hashes.get(str(path), digest) == digest, "input changed")
        hashes[str(path)] = digest
        return path

    def read(path, expected=None):
        return json.loads(bind(path, expected).read_text(encoding="utf-8"))

    bind(__file__)
    registration = read(REGISTRATION / "registration.provenance.json")
    require(registration["status"] == "ok"
            and registration["stage"] == "reflection_local_registration",
            "successful reflection registration required")
    entry = registration["outputs"]["plan"]
    require(Path(entry["path"]).resolve() == REGISTRATION / "plan.json", "plan role mismatch")
    plan = read(REGISTRATION / "plan.json", entry["sha256"])
    require(plan["run"] == str(RUN), "registration binds another minute")
    evidence = {**plan["input_hashes"], **plan["code_hashes"]}
    require({k: v["sha256"] for k, v in registration["inputs"].items()} == evidence,
            "registration input metadata differ")
    run_seal = read(RUN / "experiment.provenance.json",
                    evidence[str(RUN / "experiment.provenance.json")])
    require(run_seal["status"] == "ok", "old minute incomplete")
    sequence = Path(run_seal["params"]["sequence"]).resolve(strict=True)
    manifest = read(RUN / "sequence_manifest.json",
                    evidence[str(RUN / "sequence_manifest.json")])
    live_manifest = sequence / "manifest.json"
    bind(live_manifest, evidence[str(live_manifest)])
    require(sha(live_manifest) == sha(RUN / "sequence_manifest.json"), "manifest copies differ")
    require(manifest["image_shape_hw"] == [1200, 1200], "native dimensions changed")
    calibration = RUN / "calibration_snapshot"
    calibration_manifest = read(calibration / "calibration_manifest.json",
                                 evidence[str(calibration / "calibration_manifest.json")])
    for model in calibration_manifest["model_files"]:
        path = calibration / model["path"]
        require(evidence[str(path)] == model["sha256"], "calibration model binding differs")
        bind(path, model["sha256"])
    source_paths = [Path(__file__).resolve()]
    producer_code = {Path(k).as_posix(): v
                     for k, v in run_seal["params"]["code_hashes"].items()}
    for name in ("calib.py", "types.py", "provenance.py"):
        path = ROOT / "src/medaka3d" / name
        digest = producer_code[path.relative_to(ROOT).as_posix()]
        require(evidence[str(path)] == digest, "registered and producer source differ")
        bind(path, digest)
        bind(RUN / "code_snapshot" / path.relative_to(ROOT), digest)
        source_paths.append(path)
    package_init = ROOT / "src/medaka3d/__init__.py"
    bind(package_init, evidence[str(package_init)])
    bind(REGISTRATION / "code_snapshot/src/medaka3d/__init__.py", evidence[str(package_init)])
    source_paths.append(package_init)
    backgrounds, background_info = {}, {}
    for view in VIEWS:
        item = manifest["backgrounds"][view]
        path = (sequence / item["path"]).resolve(strict=True)
        require(path.is_relative_to(sequence) and evidence[str(path)] == item["sha256"],
                "background input role/hash differs")
        bind(path, item["sha256"])
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        require(image is not None and image.shape == (1200, 1200, 3)
                and image.dtype == np.uint8, "background image shape/dtype invalid")
        backgrounds[view] = image
        background_info[view] = dict(path=str(path), file_sha256=item["sha256"],
                                    decoded_bgr_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                                    median_source_frames=item["frames"])
    rig = load_rig(calibration)
    require(tuple(rig.views) == tuple(VIEWS), "calibration view order changed")
    corners_xy = np.array([[0., 0.], [192., 0.], [192., 192.], [0., 192.]])
    fractions = np.linspace(0., 1., 257)
    edge_xy = np.stack([
        corners_xy[i] + fractions[:, None] * (corners_xy[(i + 1) % 4] - corners_xy[i])
        for i in range(4)])
    heights = (0., -6.)
    edges = np.stack([np.concatenate((edge_xy, np.full((4, len(fractions), 1), z)), axis=-1)
                      for z in heights])
    corners = np.stack([np.column_stack((corners_xy, np.full(4, z))) for z in heights])
    uv = np.stack([rig.camera(view).project(edges) for view in VIEWS])
    corner_uv = np.stack([rig.camera(view).project(corners) for view in VIEWS])
    finite = np.isfinite(uv).all(axis=-1)
    in_image = finite & ((uv >= 0) & (uv <= 1199)).all(axis=-1)
    require(finite.any(axis=(1, 2, 3)).all(), "no drawable footprint in a view")
    OUT.mkdir()
    with provenance("nominal_floor_projection_diagnostic", OUT, name="projection") as record:
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
        for path in source_paths:
            target = OUT / "code_snapshot" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            require(sha(target) == hashes[str(path)], "snapshot differs")
            record.output(target.relative_to(OUT).as_posix(), target, sha256=sha(target))
        fig, axes = plt.subplots(1, 3, figsize=(18, 6.8), dpi=200)
        cyan, magenta = "#00ffff", "#ff52c9"
        for v, (view, ax) in enumerate(zip(VIEWS, axes, strict=True)):
            ax.imshow(cv2.cvtColor(backgrounds[view], cv2.COLOR_BGR2RGB),
                      extent=(-.5, 1199.5, 1199.5, -.5), interpolation="none")
            for level, (color, style) in enumerate(((cyan, "-"), (magenta, "--"))):
                for edge in uv[v, level]:
                    display = np.where(np.isfinite(edge).all(axis=-1)[:, None], edge, np.nan)
                    ax.plot(display[:, 0], display[:, 1], color=color, linestyle=style,
                            linewidth=1.8, zorder=4)
            for letter, point in zip("ABCD", corner_uv[v, 0], strict=True):
                if np.isfinite(point).all() and np.all((point >= 0) & (point <= 1199)):
                    ax.scatter(*point, s=25, color=cyan, edgecolor="black", zorder=5)
                    # Label toward the image centre, retaining visible endpoint markers.
                    offset = np.where(point < 600, 9, -22)
                    ax.annotate(letter, point, xytext=tuple(offset), textcoords="offset points",
                                fontsize=11, color=cyan, weight="bold",
                                bbox=dict(facecolor="black", alpha=.7, pad=2, edgecolor="none"))
            ax.set(xlim=(-.5, 1199.5), ylim=(1199.5, -.5), aspect="equal",
                   xlabel="native u [px]", ylabel="native v [px]")
            ax.set_title(view + "  |  sealed median background", fontsize=11)
            ax.set_xticks([0, 300, 600, 900, 1199])
            ax.set_yticks([0, 300, 600, 900, 1199])
        fig.suptitle("Fixed nominal bottom footprint through the existing refractive calibration",
                     fontsize=15, y=.975)
        handles = [Line2D([0], [0], color=cyan, lw=2.5,
                          label="z = 0 mm: nominal floor, not surveyed"),
                   Line2D([0], [0], color=magenta, lw=2.5, ls="--",
                          label="z = -6 mm: engineering reference only; NOT a reflector")]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .055),
                   ncol=2, facecolor="#333333", labelcolor="white", fontsize=10)
        fig.text(.5, .025,
                 "Floor corners (x,y) mm: A=(0,0), B=(192,0), C=(192,192), D=(0,192).  "
                 "No plane fitting, occlusion test or physical-floor validation.",
                 ha="center", fontsize=10)
        fig.subplots_adjust(left=.045, right=.985, bottom=.16, top=.88, wspace=.16)
        target = OUT / "nominal_floor_three_views.png"
        fig.savefig(target, dpi=200)
        plt.close(fig)
        record.output("figure", target, sha256=sha(target))
        target = OUT / "projected_references.npz"
        np.savez_compressed(target, views=np.asarray(VIEWS), z_mm=np.asarray(heights),
                            world_edges_xyz=edges, world_corners_xyz=corners,
                            projected_edges_uv=uv, projected_corners_uv=corner_uv,
                            projection_finite=finite, in_native_image=in_image)
        record.output("arrays", target, sha256=sha(target))
        summary = dict(
            frames_used="No video frames decoded; three saved median PNGs only.",
            background=background_info, native_image_shape_hw=[1200, 1200],
            edge_samples_per_edge=257, projection="frozen camera.project; infinite interfaces",
            z_mm=list(heights), footprint_xy_mm=[0., 192.], plane_fit=False, GT=False, GPU=False,
            calibration_guardrail=calibration_manifest["guardrail"],
            finite_edge_samples_by_view_level=finite.sum(axis=(2, 3)).tolist(),
            in_image_edge_samples_by_view_level=in_image.sum(axis=(2, 3)).tolist(),
            corner_uv_by_view_level=corner_uv.tolist(),
            interfaces={view: dict(normal=rig.camera(view).interface_normal.tolist(),
                                   first_air_interface_point_mm=rig.camera(view).interface_point.tolist(),
                                   glass_thickness_mm=rig.camera(view).glass_thickness_mm,
                                   layer_count=rig.camera(view).layer_count) for view in VIEWS},
            runtime=dict(numpy=np.__version__, opencv=cv2.__version__,
                         matplotlib=matplotlib.__version__),
        )
        target = OUT / "summary.json"
        target.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        record.output("summary", target, sha256=sha(target))
        report = """固定校正による床の参照図を3視点の保存済み中央値背景へ重ねました。

シアン実線は `[0,192]×[0,192] mm` の nominal な `z=0` の輪郭、
マゼンタ破線は `z=-6 mm` の工学上の参照です。破線は反射面でも実測の床でもありません。
四隅 A/B/C/D はそれぞれ (x,y)=(0,0)/(192,0)/(192,192)/(0,192) です。
3パネルとも同じネイティブ画素座標・同じ縮尺で表示しています。

既存 `camera.project` を変更せず、各辺257点を投影しました。
無限界面モデルの投影であり、有限壁による遮蔽や反射を描いたものではありません。
保存校正の入口面は top が z=186、side1 が y=-4（内面 y=0）、
side2 が x=196（内面 x=192）です。これらは今回fitしていません。
床 z=0 と校正座標の絶対的な位置関係を、この図から確定することはできません。
校正manifestも、訓練キューブが世界座標の基準を固定する一方、
絶対座標の測量精度を保証しないと明記しています。

背景は旧1分入力に保存された8時点の中央値PNGです。時系列の魚像や境界が
中央値処理で変化する可能性があるため、これ自体を床や魚の正解注釈には使いません。
3背景PNGのSHAを旧manifest・反射比較の事前登録と照合し、校正4JSON、
校正ソースと旧snapshotのSHAも確認しました。原動画の再読込・推論・GT参照はありません。

[3視点図](nominal_floor_three_views.png) · [投影配列](projected_references.npz) ·
[数値記録](summary.json) · [seal](projection.provenance.json)
"""
        target = OUT / "RESULT.md"
        target.write_text(report, encoding="utf-8")
        record.output("report", target, sha256=sha(target))
        for path, digest in hashes.items():
            require(sha(path) == digest, "input/source changed during plotting")
        for entry in record.outputs.values():
            require(sha(entry["path"]) == entry["sha256"], "output/snapshot changed")
    print(json.dumps(dict(output=str(OUT), status="ok",
                          finite=finite.sum(axis=(2, 3)).tolist(),
                          in_image=in_image.sum(axis=(2, 3)).tolist())))


if __name__ == "__main__":
    main()
