"""Seal verified stage A-C artifacts and write the scientific verdict."""

# ruff: noqa: RUF001

from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

from medaka3d.aquarium import sha256
from medaka3d.provenance import provenance

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "work/variable_count_tracklets_v1"


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


class References(HTMLParser):
    def __init__(self):
        super().__init__()
        self.references = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ("href", "src") and value:
                self.references.append(value)


def main():
    summary, audit = read("summary.json"), read("audit.json")
    media, report = read("review/manifest.json"), read("report_manifest.json")
    tests, browser = read("test_receipt.json"), read("browser_receipt.json")
    scope = read("scientific_scope_audit.json")
    assert scope["complete"] and not scope["parameters_retuned"]
    assert audit["status"] == "passed" and audit["method_case_view_arrays"] == 288
    assert tests["status"] == "passed" and browser["status"] == "passed"
    assert len(media["assets"]) == 42 and media["all_videos_fully_decoded"]
    assert report["all_method_view_payloads"] == 36
    assert report["report_sha256"] == sha256(OUT / "report.html")
    assert browser["html_sha256"] == sha256(OUT / "report.html")
    for asset in media["assets"]:
        for kind in ("video", "before", "center", "after"):
            assert sha256(OUT / "review" / asset[kind]) == asset[kind + "_sha256"]
    for payload in report["payloads"].values():
        for kind in ("script", "plot"):
            assert sha256(OUT / "report_data" / payload[kind]) == payload[kind + "_sha256"]
    records = summary["records"]
    main_rows = [r for r in records if r["case"] == "cg20s"]
    table, totals = [], {}
    for method in summary["protocol"]["conditions"]:
        rows = [r for r in main_rows if r["method"] == method]
        total = {
            k: sum(r[k] for r in rows)
            for k in (
                "core_rows",
                "contiguous_runs",
                "adjacent_connections",
                "eligible_observations",
                "unconnected_observations",
                "gap_connections",
            )
        }
        totals[method] = total
        assert total["core_rows"] == 29386 and total["eligible_observations"] == 29377
        connection = total["adjacent_connections"] / total["eligible_observations"]
        table.append(
            f"| {method} | {total['contiguous_runs']:,} | {connection:.2%} | "
            f"{1 - connection:.2%} | {total['gap_connections']} |"
        )
    difference = totals["P1"]["contiguous_runs"] - totals["P0"]["contiguous_runs"]
    decoded = sum(a["full_decode"]["decoded_frames"] for a in media["assets"])
    result = f"""# Stages A-C results: variable-detection-count 2D tracklets

**Verdict: parked (not promoted).** Stages A-C are complete: implementation, runs of all
conditions, HTML and comparison material, and verification.
P1 fragmented the main 20 s into {difference:,} more intervals than P0 and did not meet the
acceptance criteria.
Settings were not changed to fit the results. No improvement in physical-fish ID accuracy is
claimed.

## Main 20 s [75270,78590), 166 fps, the same 3 fish

All conditions retain 29,386 rows. Original points, dropouts, detection scores and global_row
match C01.
The denominator of the link rate is 29,377 observations (excluding the first core frame).
This is the fraction of adjacent links, not the rate of correct individual continuation.

| Condition | Contiguous intervals | Adjacent link rate | Unlinked rate | Same label across a gap |
|---|---:|---:|---:|---:|
{chr(10).join(table)}

Adjacent links have both ends inside the core. Same labels across a gap also include those whose
end is inside the core and whose start is in the leading context.
The protocol's general metric_edge_scope wording is correct only for adjacent links.
For the number of gaps with both ends strictly inside the core, see
`scientific_scope_audit.json`.
P0/P1 commit adjacent links only; P1's relations across a gap stay in the candidate table and
are not merged.

S1 isolates the returned-row ordering problem through object correspondence. S2 changes only the
OKS scale from S1, to top=14400 / side=10000 px². Both are re-runs with local history in the
current environment, not identical reproductions of the earlier whole-recording processing of
the saved C27.
P0 was confirmed byte-identical to the original source and identical in result on all 33 known
case-views.

P1's competition evaluation with open endpoints came with many holds and forward/reverse
disagreements.
The auxiliary M/T support reduced fragmentation, but it remains higher than P0. Strata by
proximity test, node dropout and detection count, and the event evidence, are saved in the
HTML/CSV. The cause is not asserted to be true fish ID errors or reflections.
For the actual mechanism of candidate competition, see the scientific audit record.

## Scope and undetermined items

The plan's 11 cases × 3 views were extended with the 20 s [332,3652), selected in advance from
metadata only, and run after the settings were fixed. The selection rationale, which excluded
known annotation positions, review intervals, long-interval experiments and background-image
positions together with their context, and its SHA are in `selection_audit.json`.
This is the same recording and the same group, not a test demonstrating generalization to
independent groups.
Detections and saved C27 for the whole recording already existed, so this is also not proof
that "nobody has looked at it before".

Local 17-frame tallies and review questions for 13 anchors × 3 views are saved in
`case_findings.json`.
The no-swap reviews of f76791/f323819/f77415 are limited in camera, panel and per-2D-row
correspondence, and were not repurposed as all-individual 2D ground truth. No new human labels
were created.
The denominator of confirmed individual ground truth is 0, so the correct-continuation rate and
wrong-link rate are undetermined.
Reflections are undetermined for all rows. 3D reconnection and assignment correction remain as
stage D.

## Artifacts and verification

- [HTML report](../../work/variable_count_tracklets_v1/report.html)
- 42 MP4 (3 full 20 s, 39 slow-motion event clips), 126 PNG, 36 native-frame SVG.
- Decoded all {decoded:,} output frames of every video. The full videos are sampled at 30 fps;
  the event videos play all 166 native frames at 30 fps. The HTML can switch between 6
  conditions.
- Independent audit of the 288 condition/case/view arrays, 21 real-detection-count regressions,
  6 full-row shuffles of real input, and 416 existing files unchanged.
- Synthetic tracking, SLEAP control, rendering and lab-structure tests and Ruff pass. Details in
  `test_receipt.json`.
- Browser switching, playback and reference links checked. Details in `browser_receipt.json`.
- Saved: C01, committed links, hold evidence, predictions in separate arrays, reconnection
  candidates, node quality, all settings, all stratified tallies, provenance and SHA.

Adoption is on hold because of over-fragmentation on real data and the absence of complete
individual ground-truth annotation.
The next design change will examine how open-endpoint competition is represented, and compare
it as a separate version from the current fixed conditions.
This lab expires 2026-10-17. Production and earlier results were not changed.
"""
    (HERE / "RESULT.md").write_text(result, encoding="utf-8")
    parser = References()
    parser.feed((OUT / "report.html").read_text(encoding="utf-8"))
    checked = []
    for reference in parser.references:
        parsed = urlsplit(reference)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        path = (OUT / unquote(parsed.path)).resolve()
        assert path.exists(), f"Broken report reference: {reference}"
        checked.append(reference)
    receipt = dict(
        status="passed",
        html_static_references_checked=sorted(set(checked)),
        numerical_arrays=288,
        video_count=42,
        still_count=126,
        plot_count=36,
        decoded_video_frames=decoded,
        main20s_totals=totals,
        audit_status=audit["status"],
        tests=tests,
        browser=browser,
        scientific_scope_audit_present=bool(scope),
        decision="parked_not_promoted",
        stages_completed=["A", "B", "C"],
    )
    with provenance("variable_count_final_delivery", OUT) as rec:
        for name in (
            "protocol.json",
            "summary.json",
            "audit.json",
            "report_manifest.json",
            "scientific_scope_audit.json",
            "test_receipt.json",
            "browser_receipt.json",
        ):
            rec.input(name, OUT / name, sha256=sha256(OUT / name))
        rec.output("result", HERE / "RESULT.md", sha256=sha256(HERE / "RESULT.md"))
        target = OUT / "final_receipt.json"
        target.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
        rec.output("receipt", target, sha256=sha256(target))
        manifest = {
            p.relative_to(OUT).as_posix(): sha256(p)
            for p in OUT.rglob("*")
            if p.is_file()
            and p.name
            not in ("delivery_manifest.json", "variable_count_final_delivery.provenance.json")
        }
        manifest_path = OUT / "delivery_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        rec.output("manifest", manifest_path, sha256=sha256(manifest_path))
    print("Stage A-C delivery verified and sealed")


if __name__ == "__main__":
    main()
