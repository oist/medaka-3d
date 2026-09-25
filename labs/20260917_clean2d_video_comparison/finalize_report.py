"""Verify the canonical offline export, copy it, and record delivery provenance."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import shutil
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

from medaka3d.aquarium import sha256
from medaka3d.provenance import provenance

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "work/clean2d_video_comparison_v1"
APP = BASE / "report_app"
INPUTS = BASE / "report_inputs"


class Inspector(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.parts, self.external = [], []
        self.capturing = False
        self.snapshot_hash = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "script":
            self.capturing = (
                attrs.get("id") == "data-app-reviewed-snapshot"
                or "data-app-snapshot-chunk" in attrs
            )
            if attrs.get("src") and not attrs["src"].startswith("data:"):
                self.external.append(attrs["src"])
        if (
            tag == "link"
            and attrs.get("rel") == "stylesheet"
            and not attrs.get("href", "").startswith("data:")
        ):
            self.external.append(attrs.get("href"))
        if tag == "meta" and attrs.get("name") == "data-app-snapshot-sha256":
            self.snapshot_hash = attrs.get("content")

    def handle_endtag(self, tag):
        if tag == "script":
            self.capturing = False

    def handle_data(self, data):
        if self.capturing:
            self.parts.append(data)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path):
    return str(path.relative_to(ROOT)).replace("\\", "/")


def write(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def main():
    portable = APP / ".data-app-offline/exports/comparison_report.html"
    target = BASE / "comparison_report.html"
    snapshot_path = APP / "src/data.json"
    snapshot = read(snapshot_path)
    assert snapshot["buildStatus"] == "complete"
    assert snapshot["protocol"]["geometry_used"] is False
    build = read(APP / "dist/data-app-build.json")
    assert build["html"]["sha256"] == sha256(APP / "dist/index.html")
    browser = read(INPUTS / "browser_checks.json")
    assert browser["split_build_sha256"] == build["html"]["sha256"]
    html = portable.read_text(encoding="utf-8")
    inspector = Inspector()
    inspector.feed(html)
    assert json.loads("".join(inspector.parts)) == snapshot
    assert inspector.snapshot_hash == sha256(snapshot_path)
    assert not inspector.external
    source_hashes = {}
    for query in snapshot["queries"].values():
        for path, digest in query.get("source", {}).get("hashes", {}).items():
            assert sha256(ROOT / path) == digest, path
            source_hashes[path] = digest
    inline = {}
    pattern = r"data:((?:image/(?:png|jpeg))|(?:video/mp4));base64,([A-Za-z0-9+/=]+)"
    for mime, content in re.findall(pattern, html):
        data = base64.b64decode(content, validate=True)
        inline[hashlib.sha256(data).hexdigest()] = dict(mime=mime, bytes=len(data))
    assets = [
        row for query in ("media", "inspection") for row in snapshot["queries"][query]["rows"]
    ]
    for row in assets:
        assert row["path"].startswith("work/clean2d_video_comparison_v1/")
        assert row["sha256"] in inline
        assert sha256(ROOT / row["path"]) == row["sha256"]
    videos = [row for row in assets if row.get("kind") == "video"]
    assert len(videos) == 42
    assert sum(r["output_frames"] for r in videos) == 39 * 166 + 3 * 600
    checks = dict(
        status="pass",
        html_sha256=sha256(portable),
        html_bytes=portable.stat().st_size,
        snapshot_sha256=sha256(snapshot_path),
        snapshot_object_equal=True,
        media_all_embedded_with_matching_sha256=True,
        declared_media=len(assets),
        source_file_hashes_verified=len(source_hashes),
        video_count=len(videos),
        external_script_and_style_count=0,
        file_url_browser_test="Not performed; browser file URL policy is restrictive.",
        browser_checks=browser,
    )
    check_path = INPUTS / "offline_final_validation.json"
    write(check_path, checks)
    sources = [
        Path(__file__),
        Path(__file__).with_name("build_report.py"),
        INPUTS / "browser_checks.json",
        check_path,
        snapshot_path,
        APP / "dist/data-app-build.json",
        APP / "src/content/report/ReportContent.jsx",
        APP / "src/content/report/report.css",
        APP / "src/content/report/media.js",
        ROOT / "work/clean2d_video_comparison_v1/summary.json",
        ROOT / "work/clean2d_video_comparison_v1/audit.json",
        ROOT / "work/clean2d_video_comparison_v1/review/manifest.json",
        BASE / "review/validation.json",
        BASE / "report_media/validation.json",
        BASE / "implementation_verification.json",
        BASE / "independent_review.json",
        *[
            ROOT / p
            for p in snapshot["queries"]["media"]["source"]["files"]
            if p != "work/clean2d_video_comparison_v1/review/manifest.json"
        ],
        portable,
        *sorted(INPUTS.glob("final_*.png")),
    ]
    with provenance("clean2d_report_delivery", INPUTS, name="report_delivery") as rec:
        sources.extend(sorted(INPUTS.glob("*.json")))
        sources.extend(
            ROOT / p
            for q in snapshot["queries"].values()
            for p in q.get("source", {}).get("files", [])
        )
        sources = sorted(
            set(p for p in sources if p.is_file() and p.name != "report_delivery.provenance.json")
        )
        for p in sources:
            rec.input(rel(p), p, sha256=sha256(p))
        shutil.copy2(portable, target)
        assert sha256(target) == checks["html_sha256"]
        manifest = dict(
            status="complete",
            generated_at=datetime.now(UTC).isoformat(),
            app_id=snapshot["id"],
            analysis_directory="work/clean2d_video_comparison_v1",
            delivery=dict(
                path=rel(target),
                sha256=sha256(target),
                bytes=target.stat().st_size,
                portable=True,
                canonical_offline_export=True,
            ),
            validation=checks,
            inputs=[dict(path=rel(p), sha256=sha256(p)) for p in sources],
        )
        manifest_path = BASE / "report_manifest.json"
        write(manifest_path, manifest)
        rec.output("portable_report", target, sha256=sha256(target))
        rec.output("manifest", manifest_path, sha256=sha256(manifest_path))
    print(json.dumps(manifest["delivery"], ensure_ascii=False))


if __name__ == "__main__":
    main()
