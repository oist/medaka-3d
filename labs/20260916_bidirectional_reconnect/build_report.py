"""Build an offline, source-linked review of the frozen development experiment."""

# Embedded HTML/CSS keeps its long lines; typographic dashes and signs are intentional.
# ruff: noqa: E501, RUF001

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

import numpy as np

from medaka3d.aquarium import sha256
from medaka3d.provenance import provenance


def esc(value):
    return html.escape(str(value), quote=True)


def table(headers, rows):
    return (
        '<div class="scroll"><table><thead><tr>'
        + "".join(f"<th>{esc(h)}</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
        + "</tbody></table></div>"
    )


def link(path, label):
    return f'<a href="{esc(path)}">{esc(label)}</a>'


def angle(value):
    return "direction undefined" if value is None else f"{value:.1f}°"


def validate_render_inputs(root, cases):
    """Retained v1 renders are usable only if their core inputs are unchanged."""
    checked = []
    for case in cases:
        name = case["case"]["name"]
        old = root / "analysis" / name / "poses.npz"
        new = root / "analysis_v2" / name / "poses.npz"
        with np.load(old, allow_pickle=False) as a, np.load(new, allow_pickle=False) as b:
            for key in ("frames", "pose", "xyz", "source", "observed", "uv", "slot_track"):
                if not np.array_equal(a[key][a["core"]], b[key][b["core"]], equal_nan=True):
                    raise ValueError(f"Core render input changed: {name}/{key}")
            first, stop = case["case"]["start"], case["case"]["stop"]
            held_a = (a["diag_frames"] >= first) & (a["diag_frames"] < stop)
            held_b = (b["diag_frames"] >= first) & (b["diag_frames"] < stop)
            for key in (
                "frames",
                "step_mm",
                "past_angle_deg",
                "future_angle_deg",
                "distance",
                "forward",
                "bidirectional",
            ):
                if not np.array_equal(
                    a[f"diag_{key}"][held_a], b[f"diag_{key}"][held_b], equal_nan=True
                ):
                    raise ValueError(f"Primary detection changed: {name}/{key}")
        checked.append(dict(case=name, old_sha256=sha256(old), new_sha256=sha256(new)))
    return dict(core_inputs_and_primary_diagnostics_exactly_equal=True, cases=checked)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("work/bidirectional_reconnect_v1"))
    args = parser.parse_args()
    root = args.root
    target = root / "index.html"
    if target.exists():
        raise FileExistsError(target)
    analysis = root / "analysis_v2"
    summary = json.loads((analysis / "summary.json").read_text(encoding="utf-8"))
    revision_validation = validate_render_inputs(root, summary["cases"])
    manifest = json.loads((root / "review/manifest.json").read_text(encoding="utf-8"))
    main_case = json.loads((analysis / "cg20s/analysis.json").read_text(encoding="utf-8"))
    parts = []
    parts.append("""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Medaka tracking: bidirectional jump detection test</title>
<style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#111b24;color:#e7edf3;font:16px/1.75 system-ui,sans-serif}
main{max-width:1180px;margin:auto;padding:28px}h1{font-size:28px}h2{margin-top:40px;font-size:22px}h3{font-size:18px}
a{color:#84d5ee}video{display:block;width:100%;max-height:72vh;background:#000;border-radius:8px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:9px 12px;border-bottom:1px solid #3a4854;vertical-align:top}th{background:#1c2e3e}.scroll{overflow-x:auto}.lead{padding:20px;background:#203445;border-left:4px solid #71d2e4}.muted{color:#b9c7d3;font-size:14px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:20px}details{margin:12px 0;padding:12px;background:#192732;border-radius:6px}summary{cursor:pointer}img{display:block;width:100%;height:auto}button,select{font:inherit;border:1px solid #7895aa;border-radius:4px;background:#203445;color:#eef8ff;padding:5px 10px;cursor:pointer}code{overflow-wrap:anywhere}.pills a{display:inline-block;margin:5px 15px 5px 0}button:focus-visible,a:focus-visible{outline:3px solid #ffd46a}.warn{color:#ffd482}
</style><main><h1>Bidirectional jump detection: 20 s and known failures</h1>
<p class="muted">2026-09-16 · C34 λ0.767 · 166 fps · development test on frozen current tracking data</p>
<div class="lead"><strong>The bidirectional test detected additional anomaly candidates. However, 0 automatic corrections were accepted.</strong>
<p>In the 20 s interval there were 21 candidates by distance only, 96 forward and 110 bidirectional. Source-image comparison sheets were made for all 14 additional ones.
Candidate counts are counts of "individual × adjacent frame", not swap counts or ground-truth counts. The 12 known scenes, 3 intervals of past miscorrections and 3 control intervals were also evaluated.</p></div>
<nav class="pills"><a href="#movie">20 s video</a><a href="../aquarium_cg/c34_75270_78590/aquarium.mp4">CG only</a><a href="#slow">Slow motion</a><a href="#added">14 additions</a><a href="#known">12 known scenes</a><a href="#scores">Numbers for all intervals</a><a href="#sensitivity">Threshold comparison</a></nav>
<h2 id="movie">Source video from 3 views and 3D CG</h2>
<p>The tracked coordinates at each time point are applied to a simplified fish. On the skeleton the circle is the mouth, the cross the pectoral fin, and the end the tail. Colour and number are the current tracking slot.
The 20 s overview video is decimated to 30 fps. Use the slow-motion videos and stills below to check single-frame anomalies.</p>
<video id="mainvideo" controls preload="metadata" src="review/cg20s_raw_and_measured_cg.mp4"></video>
<p class="muted">f75270–78589 (20 s). Labels: distance, forward, and bidirectional-only (added only in the backward direction). Labels are also shown on the 17 native frames before and after each candidate so they are easy to find. The CG has no coordinate interpolation or smoothing.</p>
<h2 id="slow">Slow motion retaining every 166 fps frame</h2>
<p>The original 2 s of 332 frames is played at 30 fps, so each video is about 11.07 s long, about 5.53× slower.</p><div class="grid">
<section><h3>Around f76235: a large position jump</h3><video controls preload="metadata" src="review/native_slow_f76235.mp4"></video>
<p>Between closely packed fish the correspondence of mouth, pectoral fin and tail may break down. This scene checks whether a whole-fish ID exchange alone can explain it.</p></section>
<section><h3>Around f76930: both directions prefer the exchange, but it is held</h3><video controls preload="metadata" src="review/native_slow_f76930.mp4"></video>
<p>The S0↔S1 correspondence gives the minimum error in both directions, but the margin over the runner-up and the per-individual errors do not meet the prespecified conditions. The correction was not accepted.</p></section></div>""")

    group = next(
        g for g in main_case["episodes"] if g["start_frame"] <= 76930 < g["stop_frame_exclusive"]
    )
    audit = group["persistent_audit"]
    comparison = [r for r in audit["scores"] if r["permutation"] in ([0, 1, 2], [1, 0, 2])]
    parts.append(
        table(
            [
                "f76930 correspondence",
                "Forward: summed error of 3 fish",
                "Backward: summed error of 3 fish",
                "Max per-individual error (forward / backward)",
            ],
            [
                [
                    esc(r["permutation"]),
                    f"{r['forward_sum_mm']:.2f} mm",
                    f"{r['backward_sum_mm']:.2f} mm",
                    f"{r['forward_max_fish_mm']:.2f} / {r['backward_max_fish_mm']:.2f} mm",
                ]
                for r in comparison
            ],
        )
    )
    parts.append(
        f'<p class="muted">Required for acceptance: a margin over the runner-up of at least 5 mm in both directions, and each individual\'s prediction error at most 3 mm in both directions. '
        f"The actual margins were {audit['forward_margin_mm']:.2f} / {audit['backward_margin_mm']:.2f} mm. "
        "This test checks tracklet correspondence before and after; it does not determine the exact exchange time within the interval.</p>"
    )

    parts.append(
        '<h2 id="added">14 added by the backward direction</h2><p>Each image pairs the source image with the projection of the 3D nodes. Click to open at full size. '
        "Image observations by the assistant are not treated as independent individual-ID ground-truth labels. "
        "In 6 of the 14 the forward angle is undefined, for reasons such as a low past reference speed, and they were judged from the future side only.</p>"
    )
    for item in manifest["contact_sheets"]:
        if item["role"] != "bidirectional_only":
            continue
        frame = item["center_frame"]
        label = f"f{frame} · S{item['diagnostic_slot']} · {item['step_mm']:.2f} mm · angle {angle(item['past_angle_deg'])} / {angle(item['future_angle_deg'])} (past / future)"
        path = "review/" + item["path"]
        parts.append(
            f'<details><summary>{esc(label)}</summary><p><button data-seek="{max(0, (frame - 75270) / 166 - 0.5):.4f}">Go to this point in the 20 s video</button> '
            f'{link(path, "Open comparison image")}</p><a href="{esc(path)}"><img loading="lazy" alt="3-view comparison for {esc(label)}" src="{esc(path)}"></a></details>'
        )
    note_files = sorted((root / "review").glob("*visual_notes*.json"))
    parts.append(
        "<p>Image observations: "
        + " · ".join(link(p.relative_to(root).as_posix(), p.stem) for p in note_files)
        + "</p>"
    )

    parts.append(
        '<h2 id="known">The 12 known scenes</h2><p>The original notes refer to earlier C30 results. They include scenes already fixed in C34. '
        "Therefore 'detection near a note' is not treated as recall. Past slot names are also not used directly as current individual ground truth.</p>"
    )
    for item in manifest["contact_sheets"]:
        if item["role"] != "historical_failure_neighborhood":
            continue
        path = "review/" + item["path"]
        label = f"f{item['center_frame']} · past note: {item['historical_note']['kind']}"
        flags = item["current_c34_detection"]
        parts.append(
            f"<details><summary>{esc(label)}</summary><p>Current candidates near the note: "
            f"distance {'yes' if flags['distance'] else 'no'} / forward {'yes' if flags['forward'] else 'no'} / bidirectional {'yes' if flags['bidirectional'] else 'no'}.</p>"
            f'<p>{esc(item["historical_note"]["annotator_note"])}</p><a href="{esc(path)}"><img loading="lazy" alt="3-view comparison for {esc(label)}" src="{esc(path)}"></a></details>'
        )

    parts.append(
        '<h2 id="scores">Results for all 19 intervals</h2><p>Each method cell is "candidate count in individual × adjacent-frame units / number of intervals after grouping nearby candidates". '
        "The known case at f76930 overlaps the 20 s interval, so a simple sum over rows is not interpreted as an independent detection count.</p>"
    )
    rows = []
    for case in summary["cases"]:
        info = case["case"]
        row = [
            link(f"analysis_v2/{info['name']}/analysis.json", info["name"]),
            f"{info['start']}–{info['stop'] - 1}",
        ]
        row += [
            f"{case['methods'][m]['slot_steps']} / {case['methods'][m]['episodes']}"
            for m in ("distance", "forward", "bidirectional")
        ]
        row += [
            str(case["bidirectional_extra_slot_steps"]),
            str(len(case["candidate_geometry_proposals"])),
            str(case["accepted_identity_corrections"]),
        ]
        rows.append(row)
    parts.append(
        table(
            [
                "Interval (detail JSON)",
                "native frames",
                "Distance",
                "Forward",
                "Bidirectional",
                "Added by backward",
                "Passed geometry conditions",
                "Corrections accepted",
            ],
            rows,
        )
    )
    parts.append("""<p>Linking verdicts for the 55 candidate intervals in the 20 s: original correspondence has minimum error 37, reference interval insufficient 15, ambiguous 3.
These are not classifications that prove "the ID is correct". Temporary exchanges of 1–3 frames were also tested, but no proposal passed all prespecified conditions.</p>
<h3>ID retention checked against independent female annotation</h3>
<p><strong>Both before correction and after acceptance: 295 kept / 2 swaps / 2 undeterminable.</strong>
The 299 adjacent intervals of the 300 annotations were fixed; no interval was deleted or bridged because of missing observations. The values are identical because 0 corrections were accepted.
The annotation spans containing a swap are f76698–77285 and f130760–138892. These are not annotations of the exact exchange frame.
Some events, such as out-and-back exchanges, are not captured by the annotation, so these numbers alone cannot guarantee identity for all individuals at all times.</p>""")
    parts.append(
        link("analysis_v2/female_baseline.json", "Full evaluation against the fixed annotation")
    )

    parts.append(
        '<h2 id="sensitivity">Direction window and angle comparison on the 20 s interval</h2><p>The best threshold was not re-selected after seeing the results. The primary result is 17 frames, 80°. The distance criterion gives 21 in every setting.</p>'
        "<p>Primary test: a one-frame move over 4 mm, or over 1 mm with an angle of at least 80° to the reference direction. "
        "The reference direction is the median of the moves over 17 frames on each side, excluding the 2 moves before and after the target. "
        "Below a reference speed of 0.05 mm/frame the angle is not used.</p>"
    )
    parts.append(
        table(
            [
                "Direction window (166 fps)",
                "Angle",
                "Forward candidates",
                "Bidirectional candidates",
                "Bidirectional intervals",
            ],
            [
                [
                    str(s["window"]),
                    f"{s['angle_deg']}°",
                    str(s["forward"]["slot_steps"]),
                    str(s["bidirectional"]["slot_steps"]),
                    str(s["bidirectional"]["episodes"]),
                ]
                for s in main_case["sensitivity"]
            ],
        )
    )
    parts.append("""<h2>Decision and next validation</h2>
<p>The bidirectional test can serve as an entry point for extracting lateral wobble that distance alone misses. However, the additional detections cannot yet be called an improvement in swap detection.
The source images contain scenes with suspected node position jitter and mixed correspondence during close approaches, which the 6 whole-individual exchanges alone may not be able to handle.</p>
<p>The next implementation candidate re-examines the 3-view detection correspondence in the local intervals extracted here, evaluating the grouping of mouth, pectoral fin and tail together with reprojection error.
It uses the stable intervals before and after as boundary conditions and evaluates node/view inconsistency separately from individual-ID exchange. Using independent short-interval ID annotation, both improvements and miscorrections need to be checked on the known cases and on untuned intervals.</p>
<h2>Records for reproduction</h2>""")
    records = [
        ("analysis_v2/protocol.json", "Prespecified conditions and intervals"),
        ("analysis_v2/summary.json", "All numbers"),
        (
            "analysis_v2/bidirectional_development.provenance.json",
            "Analysis inputs and code record",
        ),
        ("review/manifest.json", "Video and comparison-sheet record"),
        ("review/bidirectional_raw_review.provenance.json", "Video generation record"),
    ]
    parts.append("<p>" + " · ".join(link(path, label) for path, label in records) + "</p>")
    parts.append(f'<p class="muted">C34 SHA256: <code>{esc(summary["baseline_sha256"])}</code></p>')
    parts.append("""</main><script>
document.querySelectorAll('[data-seek]').forEach(button=>button.addEventListener('click',()=>{
const video=document.getElementById('mainvideo');video.currentTime=Number(button.dataset.seek);video.pause();video.scrollIntoView({behavior:'smooth',block:'center'});
}));</script></html>""")
    document = "\n".join(parts)
    # Offline links must resolve; links inside this report never require fetch/CDNs.
    for path in re.findall(r'(?:href|src)="([^"]+)"', document):
        if path.startswith("#"):
            continue
        if not (root / html.unescape(path)).is_file():
            raise FileNotFoundError(path)
    with provenance("bidirectional_report", root) as rec:
        for path in [
            Path(__file__),
            analysis / "summary.json",
            analysis / "cg20s/analysis.json",
            root / "review/manifest.json",
            *note_files,
        ]:
            rec.input(path.name, path, sha256=sha256(path))
        target.write_text(document, encoding="utf-8")
        rec.output("offline_review", target, sha256=sha256(target))
        validation_path = root / "revision_validation.json"
        validation_path.write_text(json.dumps(revision_validation, indent=2), encoding="utf-8")
        rec.output("revision_validation", validation_path, sha256=sha256(validation_path))
        rec.param("local_links_checked", True)
        rec.param("browser_interaction_verified", False)
    print(target)


if __name__ == "__main__":
    main()
