"""QA a frozen raw-only review pack and its shared native probe; no builder or labels."""

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import audit_scene0_probe_support as a
from review_scene import bind_template, render_html

from medaka3d.calib import load_rig
from medaka3d.provenance import provenance

PROBES = [78325, 79177, 79316]
START, STOP = 78309, 79332


def main():
    root = a.m.REPO / "work/three_body/minute_reviews/entry_bridge_78309-79332"
    old_root = a.m.REPO / "work/three_body/minute_reviews/scene0_79236-79488"
    output = root / "bridge_qa"
    if output.exists():
        raise FileExistsError("QA already exists")
    hashes = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        a.m.require(path.is_relative_to(a.m.REPO), "input outside workspace")
        digest = a.m.sha256(path)
        a.m.require(expected is None or digest == expected, "changed input")
        hashes[str(path)] = digest
        return path

    for path in sorted(set(a.m.code_paths()) | {Path(a.__file__), Path(__file__)}):
        bind(path)
    primary = json.loads(bind(root / "review_scene.provenance.json").read_text())
    a.m.require(primary["status"] == "ok", "new raw pack incomplete")
    for item in primary["outputs"].values():
        bind(item["path"], item["sha256"])
    prior = json.loads(bind(old_root / "review_scene.provenance.json").read_text())
    a.m.require(prior["status"] == "ok", "old raw pack incomplete")
    for item in prior["outputs"].values():
        bind(item["path"], item["sha256"])
    summary = json.loads((root / "summary.json").read_text())
    old_summary = json.loads((old_root / "summary.json").read_text())
    a.m.require(
        summary["probe_frames"] == PROBES and summary["review_clip_range"] == [START, STOP],
        "wrong bridge probes/range",
    )
    a.m.require(
        summary["source_manifest_sha256"] == old_summary["source_manifest_sha256"],
        "different raw source",
    )
    a.m.require(
        summary["source_frame_range"] == old_summary["source_frame_range"] == [76620, 86580],
        "different full source range",
    )
    manifest_path = bind(summary["source_manifest"], summary["source_manifest_sha256"])
    manifest = json.loads(manifest_path.read_text())
    expected, frames, offset = bind_template(
        manifest["frames"], summary["source_manifest_sha256"], START, STOP, PROBES
    )
    a.m.require(
        summary["displayed_frames"] == frames and summary["first_source_clip_offset"] == offset,
        "native range alignment differs",
    )
    template = json.loads((root / "truth_template.json").read_text())
    a.m.require(template == expected, "template has unexpected changes or labels")
    html_source = bind(a.m.LAB / "review_scene.html", summary["html_template_sha256"])
    html = (root / "index.html").read_text(encoding="utf-8")
    a.m.require(
        html == render_html(html_source.read_text(encoding="utf-8"), expected, frames),
        "HTML differs from exact frozen rendering",
    )
    for view in a.m.VIEWS:
        a.m.require(
            summary["native_bgr_sha256"][view]
            == manifest["raw"][view]["frame_bgr_sha256"][offset : offset + len(frames)],
            "native source hash stream differs",
        )
        a.m.require(
            all((root / "frames" / f"f{frame}_{view}.jpg").is_file() for frame in frames),
            "image link missing",
        )
    a.m.require(summary["native_pixel_hashes_verified"] == len(frames) * 3, "missing native checks")
    for key in ("review_video", "intermediate_video"):
        qa = summary[key]
        a.m.require(
            qa["decoded_frame_count"] == len(frames)
            and qa["playback_fps"] == 20
            and qa["image_shape_hw"] == [636, 1800],
            "full video decode QA differs",
        )
    shared = []
    for frame in PROBES:
        for view in a.m.VIEWS:
            path = root / f"f{frame}_{view}.png"
            image = cv2.imread(str(path))
            a.m.require(image is not None and image.shape == (1200, 1200, 3), "invalid native PNG")
            digest = a.m.array_sha256(image)
            a.m.require(
                digest == manifest["raw"][view]["frame_bgr_sha256"][frame - manifest["frames"][0]],
                "probe PNG differs from original raw pixels",
            )
            if frame == 79316:
                old = old_root / path.name
                a.m.require(
                    np.array_equal(image, cv2.imread(str(old))), "shared probe pixels differ"
                )
                a.m.require(a.m.sha256(path) == a.m.sha256(old), "shared PNG file bytes differ")
                shared.append(
                    dict(
                        frame=frame,
                        view=view,
                        pixel_sha256=digest,
                        png_sha256=a.m.sha256(path),
                        old_new_bytes_equal=True,
                    )
                )
    # Recheck only the geometric eligibility of the proposed frames. This data
    # stays in QA, never in the raw page or its unreviewed label template.
    run = a.m.REPO / "work/three_body/persistent_mouth_continuous_one_minute_v1"
    seal = json.loads(bind(run / "experiment.provenance.json").read_text())
    a.m.require(seal["status"] == "ok", "persistent run incomplete")
    path = run / "candidate.npz"
    matches = [e for e in seal["outputs"].values() if Path(e["path"]).resolve() == path]
    a.m.require(len(matches) == 1, "candidate binding missing")
    geometry = a.m.identity_inputs._npz(bind(path, matches[0]["sha256"]))
    rig = load_rig(run / "calibration_snapshot")
    for path in (run / "calibration_snapshot").glob("*.json"):
        inputs = [
            e
            for e in seal["inputs"].values()
            if "calibration" in e["path"] and Path(e["path"]).name == path.name
        ]
        a.m.require(len(inputs) == 1, "calibration input binding is not unique")
        bind(inputs[0]["path"], inputs[0]["sha256"])
        bind(path, inputs[0]["sha256"])
    geometry_code = a.m.LAB / "node_geometry.py"
    geometry_hashes = [
        value
        for key, value in seal["params"]["code_hashes"].items()
        if key.replace("\\", "/") == "labs/20260909_three_body/node_geometry.py"
    ]
    a.m.require(
        len(geometry_hashes) == 1 and a.m.sha256(geometry_code) == geometry_hashes[0],
        "node support code differs from producer",
    )
    nodes = a.CachedNodes(
        a.m.REPO / "work/generative_tracking_inputs",
        manifest["experiment"],
        np.arange(START, STOP, dtype=np.int64),
    )
    for path in nodes.paths.values():
        entry = seal["inputs"][str(path)]
        bind(path, entry["sha256"])
    supports = []
    for frame in PROBES:
        t = frame - int(geometry["frame_idx"][0])
        points = {view: nodes.at(view, frame) for view in a.m.VIEWS}
        rows, uv, gate, _ = a.inspect_state(
            geometry["pose_xyz"][t],
            geometry["sources"][t],
            geometry["observed"][t],
            geometry["identity_review"][t],
            points,
            rig,
        )
        a.m.require(
            np.all(gate.sum(1) >= 2) and np.isfinite(uv).all(),
            "proposed probe no longer geometrically eligible",
        )
        supports.append(dict(frame=frame, slot_geometry=rows))
    output.mkdir(parents=True)
    with provenance("raw_entry_bridge_artifact_qa", output, name="qa") as record:
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
        source = output / "source.py"
        shutil.copyfile(__file__, source)
        record.output("source", source, sha256=a.m.sha256(source))
        qa = dict(
            status="ok",
            original_primary_outputs_sha_checked=len(primary["outputs"]),
            old_scene_outputs_sha_checked=len(prior["outputs"]),
            clip_range=[START, STOP],
            probes=PROBES,
            source_seconds=len(frames) / 166,
            native_frames=len(frames),
            native_bgr_sha_checks=3 * len(frames),
            source_manifest_sha256=summary["source_manifest_sha256"],
            frame_offset=offset,
            shared_probe=shared,
            native_probe_pngs_verified=9,
            jpeg_links_verified=3 * len(frames),
            video_full_decode_qa=summary["review_video"],
            intermediate_full_decode_qa=summary["intermediate_video"],
            all_probes_geometrically_eligible=supports,
            template_has_no_human_labels=True,
            generated_html_matches_frozen_source_exactly=True,
            ui_operation_verified=False,
            browser_used=False,
            export_filename="raw_review_78309_<timestamp>.json",
            top_probe_visual_inspection_by_parent_frames=PROBES,
            initial_qa_attempt=(
                "pre-output KeyError for Windows-backslash producer code key; "
                "normalized key spelling in this new QA only. Raw pack unchanged."
            ),
        )
        path = output / "QA.json"
        path.write_text(json.dumps(a.m.safe(qa), indent=2, allow_nan=False), encoding="utf-8")
        record.output("QA", path, sha256=a.m.sha256(path))
        text = (
            "# 入口への橋渡しレビュー\n\n"
            "[原画レビューを開く](../index.html) / [3視点動画](../raw_three_views_slow.mp4)\n\n"
            "前のscene0の最初79316に至る映像の連続性を確認するための追加パックです。"
            "原区間は78309..79331の1023frame、166fpsで約6.16秒です。"
            "全native frameを残した20fps動画は約51.15秒になります。\n\n"
            "提案frameは78325、79177、79316です。原画を最初から最後まで前後に送り、"
            "3つの静止画だけでなく全区間の連続性を確認してください。"
            "判別できない区間は「曖昧」「見えない」のままで構いません。"
            "frameの選択と口の位置も映像で確認してください。\n\n"
            "A/B/Cはこのclip内で入口から任意に選ぶ匿名の名前です。"
            "前のclipの同じ文字と自動的に同一個体とは扱いません。"
            "共有frame79316は旧scene0と同じmanifest・同じ3視点原画です。"
            "旧clipのJSONをこのページへ読み込むことはできません。"
            "それぞれのレビュー結果を別々に保存してください。\n\n"
            "書き出しファイル名は `raw_review_78309_<timestamp>.json` です。"
            "現在のテンプレートには人手ラベルがなく、生物個体IDの正解はまだありません。\n\n"
            "CPUのみで全3069原画のnative SHA、全JPEGリンク、9枚のlossless probe PNG、"
            "動画全1023frameのdecodeを既存処理で検査しました。"
            "追加QAでは記録と全出力SHA、静的HTML、共有79316の3PNGのfile/pixel完全一致を確認しています。"
            "骨格・マスク・モデルIDはページに表示しません。"
            "ブラウザを起動していないため、UI操作の検証は未実施です。\n"
        )
        path = output / "RESULT.md"
        path.write_text(text, encoding="utf-8")
        record.output("result", path, sha256=a.m.sha256(path))
        a.m.require(
            all(a.m.sha256(path) == digest for path, digest in hashes.items()),
            "QA input/source changed",
        )
        a.m.require(
            all(a.m.sha256(e["path"]) == e["sha256"] for e in record.outputs.values()),
            "QA output changed",
        )
        record.param("all_input_output_sha_reverified", True)
        print(
            json.dumps(
                dict(
                    status="ok",
                    frames=len(frames),
                    shared_probe_pngs=len(shared),
                    primary_outputs=len(primary["outputs"]),
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
