"""Static JSON/import-guard and Python video-image QA; no browser or DOM."""

import hashlib
import json
import shutil
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from medaka3d.provenance import provenance  # noqa: E402

PACK = REPO / "work/three_body/review_scene_smoke_76625-76630_v2"
QA = PACK / "artifact_qa"
GUARD = REPO / "work/three_body/review_scene_import_guard_qa.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def embedded_json(html, prefix):
    start = html.index(prefix) + len(prefix)
    return json.JSONDecoder().raw_decode(html[start:])[0]


def main():
    if QA.exists():
        raise FileExistsError("artifact QA already exists")
    summary = json.loads((PACK / "summary.json").read_text(encoding="utf-8"))
    seal = json.loads((PACK / "review_scene.provenance.json").read_text(encoding="utf-8"))
    if seal["status"] != "ok" or seal["outputs"]["summary"]["sha256"] != sha256(
        PACK / "summary.json"
    ):
        raise ValueError("review pack is not sealed")
    for name, digest in summary["output_sha256"].items():
        if sha256(PACK / name) != digest:
            raise ValueError("review artifact differs from its seal")
    guard = json.loads(GUARD.read_text(encoding="utf-8"))
    html = (PACK / "index.html").read_text(encoding="utf-8")
    template = json.loads((PACK / "truth_template.json").read_text(encoding="utf-8"))
    assert embedded_json(html, "let truth=") == template
    assert embedded_json(html, "const frames=") == summary["displayed_frames"]
    assert embedded_json(html, "viewNames=") == ["top_cam", "side1_cam", "side2_cam"]
    assert template["source_frame_range"] == [76620, 77286]
    assert template["review_clip_range"] == [76625, 76630]
    assert template["source_manifest_sha256"] == summary["source_manifest_sha256"]
    assert template["status"] == "unreviewed_template_not_ground_truth"
    assert all(
        not fish["points"] for anchor in template["anchors"] for fish in anchor["animals"].values()
    )
    assert guard["all_passed"] and guard["case_count"] == 23
    assert guard["source_html_sha256"] == summary["html_template_sha256"]
    function = html.split("/* REVIEW_IMPORT_GUARD_BEGIN */", 1)[1].split(
        "/* REVIEW_IMPORT_GUARD_END */", 1
    )[0]
    assert hashlib.sha256(function.encode()).hexdigest() == guard["function_sha256"]
    paths = [
        PACK / "summary.json",
        PACK / "review_scene.provenance.json",
        PACK / "index.html",
        PACK / "truth_template.json",
        PACK / "raw_three_views_slow.mp4",
        GUARD,
        Path(__file__),
    ]
    hashes = {str(path): sha256(path) for path in paths}
    QA.mkdir()
    with provenance("review_scene_static_artifact_qa", QA, name="artifact_qa") as record:
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
        for source in (GUARD, Path(__file__)):
            target = QA / source.name
            shutil.copyfile(source, target)
            record.output(target.name, target, sha256=sha256(target))
        capture = cv2.VideoCapture(str(PACK / "raw_three_views_slow.mp4"))
        decoded, images = 0, []
        try:
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                if decoded in (0, 2, 4):
                    path = QA / f"decoded_f{summary['displayed_frames'][decoded]}.png"
                    if not cv2.imwrite(str(path), image):
                        raise RuntimeError("decoded QA PNG write failed")
                    images.append(path.name)
                    record.output(path.name, path, sha256=sha256(path))
                decoded += 1
        finally:
            capture.release()
        assert decoded == 5
        if hashes != {path: sha256(path) for path in hashes}:
            raise ValueError("pack or QA code changed during independent artifact QA")
        result = dict(
            status="passed",
            full_source_range=[76620, 77286],
            displayed_clip_range=[76625, 76630],
            html_embedded_data_exact=True,
            import_guard_cases_passed=23,
            import_guard_execution_engine=guard["execution_engine"],
            actual_published_function_matches_tested_function=True,
            raw_native_bgr_images_verified=summary["native_pixel_hashes_verified"],
            video_decoded_frames=decoded,
            decoded_images=images,
            anonymous_template_has_no_marks=True,
            browser_used=False,
            ui_operations_verified=False,
            outputs_unchanged=True,
        )
        path = QA / "summary.json"
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        record.output("summary", path, sha256=sha256(path))
        print(json.dumps(result))


if __name__ == "__main__":
    main()
