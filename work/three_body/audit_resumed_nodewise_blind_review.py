"""Check completed review artifacts without changing their scientific status."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

import cv2
import numpy as np

WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK.parents[1] / "src"))
from medaka3d.provenance import provenance  # noqa: E402

ROOT = WORK / "resumed_nodewise_blind_review_v1"
LEDGER = WORK / "resumed_nodewise_risk_review_v1/scenes.json"
OUT = WORK / "resumed_nodewise_blind_review_output_qa_v1"
NODE = Path.home() / "AppData/Local/OpenAI/Codex/runtimes/cua_node/1b23c930bdf84ed6/bin/node.exe"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


class Inspector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self.visible = [], []
        self.hidden = False

    def handle_starttag(self, tag, attrs):
        self.hidden = self.hidden or tag in ("style", "script")
        self.links.extend(value for key, value in attrs if key in ("href", "src"))

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.hidden = False

    def handle_data(self, data):
        if not self.hidden:
            self.visible.append(data)


def audit():
    ledger, summary = read(LEDGER), read(ROOT / "summary.json")
    ledger_sha = hashlib.sha256(LEDGER.read_bytes()).hexdigest()
    assert read(ROOT / "review.provenance.json")["status"] == "ok"
    assert summary["status"] == "ok" and summary["all_selected_scenes_rendered"]
    assert summary["scene_ids"] == [s["scene_id"] for s in ledger["scenes"]]
    results, thumbs = [], []
    for scene in ledger["scenes"]:
        sid = scene["scene_id"]
        root = ROOT / sid
        result = read(root / "summary.json")
        first, stop = scene["start_frame"], scene["stop_frame_exclusive"]
        assert result["selection_ledger_sha256"] == ledger_sha
        assert result["native_interval"] == [first, stop]
        assert result["native_bgr_hashes_verified"] == 3 * (stop - first)
        assert result["human_review_status"] == "unreviewed"
        assert result["model_outputs_shown"] is False
        assert result["video_qa"]["all_frames_decoded"]
        with (root / "frame_map.csv").open(encoding="utf8", newline="") as stream:
            frame_map = list(csv.DictReader(stream))
        assert [int(r["native_frame"]) for r in frame_map] == list(range(first, stop))
        assert [int(r["encoded_frame"]) for r in frame_map] == list(range(stop - first))
        template = read(root / "truth_template.json")
        assert [a["frame"] for a in template["anchors"]] == scene["anchor_frames"]
        assert all(
            not a["points"] for anchor in template["anchors"] for a in anchor["animals"].values()
        )
        assert all(
            c[edge] == "unreviewed"
            for c in template["continuity"].values()
            for edge in ("entry_to_after", "after_to_exit")
        )
        for view in result["camera_views"]:
            for frame in template["still_frames_available"]:
                assert (root / f"frames/f{frame}_{view}.jpg").is_file()
            for phase in ("entry", "middle", "exit"):
                image = cv2.imread(str(root / f"{phase}_{view}.png"))
                assert image is not None and image.shape == (1200, 1200, 3)
        text = (root / "index.html").read_text(encoding="utf8")
        inspector = Inspector()
        inspector.feed(text)
        for link in inspector.links:
            assert (root / link).resolve().is_file(), (sid, link)
        visible = "".join(inspector.visible)
        assert not any(
            word in visible for word in ("Nodewise", "Sparse", "high_risk", "random_low")
        )
        js = text.split("<script>", 1)[1].split("</script>", 1)[0]
        js_path = OUT / f"{sid}.js"
        js_path.write_text(js, encoding="utf8")
        subprocess.run([str(NODE), "--check", str(js_path)], check=True, capture_output=True)
        thumbnail = cv2.imread(str(root / "entry_middle_exit.jpg"))
        assert thumbnail is not None and thumbnail.shape == (1920, 1800, 3)
        thumbnail = cv2.resize(thumbnail, (900, 960), interpolation=cv2.INTER_AREA)
        caption = np.zeros((35, 900, 3), np.uint8)
        cv2.putText(caption, sid, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        thumbs.append(np.concatenate((caption, thumbnail)))
        results.append(dict(scene_id=sid, all_checks_passed=True, native_frames=stop - first))
    for n, index in enumerate(range(0, len(thumbs), 4), 1):
        group = thumbs[index : index + 4]
        group += [np.zeros_like(thumbs[0]) for _ in range(4 - len(group))]
        sheet = np.concatenate(
            (np.concatenate(group[:2], axis=1), np.concatenate(group[2:], axis=1))
        )
        assert cv2.imwrite(str(OUT / f"contact_sheet_{n:02}.jpg"), sheet)
    index = Inspector()
    index.feed((ROOT / "index.html").read_text(encoding="utf8"))
    assert all((ROOT / link).resolve().is_file() for link in index.links)
    with (ROOT / "response_sheet.csv").open(encoding="utf8", newline="") as stream:
        responses = list(csv.DictReader(stream))
    assert len(responses) == len(results) * 3
    report = dict(
        status="ok",
        scenes_checked=len(results),
        all_selected_scenes_present=True,
        all_three_camera_anchor_images_1200px=True,
        frame_maps_match_selection=True,
        template_clicks_empty=True,
        all_responses_unreviewed=True,
        reviewer_pages_hide_methods_and_selection=True,
        html_links_resolve=True,
        javascript_syntax="all scenes passed node --check",
        browser_interaction_qa=(
            "not performed; Browser URL security policy rejected file navigation"
        ),
        biological_identity_review_performed=False,
        source_frame_hashes_verified=sum(3 * r["native_frames"] for r in results),
        scenes=results,
    )
    (OUT / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps({key: value for key, value in report.items() if key != "scenes"}))


def main():
    OUT.mkdir(exist_ok=False)
    with provenance("resumed_nodewise_blind_review_output_audit_v1", OUT, name="audit") as record:
        for name, path in (
            ("renderer_seal", ROOT / "review.provenance.json"),
            ("renderer_summary", ROOT / "summary.json"),
            ("selection", LEDGER),
            ("audit_source", Path(__file__).resolve()),
        ):
            record.input(name, path, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        audit()
        for path in OUT.iterdir():
            record.output(path.name, path, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
