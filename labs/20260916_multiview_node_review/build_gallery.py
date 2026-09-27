"""Index the exported scientific figures and raw-pixel correspondence comparisons."""

# Embedded HTML keeps its natural line lengths; typographic dashes and signs are intentional.
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


def esc(x):
    return html.escape(str(x), quote=True)


def number(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:.2f}"


def link(path, label):
    return f'<a href="{esc(path)}">{esc(label)}</a>'


def table(headers, rows):
    return (
        '<div class="scroll"><table><thead><tr>'
        + "".join(f"<th>{esc(x)}</th>" for x in headers)
        + "</tr></thead><tbody>"
        + "".join("<tr>" + "".join(f"<td>{x}</td>" for x in row) + "</tr>" for row in rows)
        + "</tbody></table></div>"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("work/multiview_node_review_v1"))
    args = parser.parse_args()
    root = args.root
    target = root / "index.html"
    if target.exists():
        raise FileExistsError(target)
    sources = [root / "analysis/summary.json", root / "review/manifest.json"]
    report = json.loads(sources[0].read_text(encoding="utf-8"))
    manifest = json.loads(sources[1].read_text(encoding="utf-8"))
    entries = list(manifest["entries"])
    for filename in ("supplemental_manifest_v2.json", "supplemental_manifest_v3.json"):
        path = root / "review" / filename
        if path.exists():
            extra = json.loads(path.read_text(encoding="utf-8"))
            entries.extend(extra["entries"])
            sources.append(path)
    entries = list({entry["path"]: entry for entry in entries}.values())
    arrays, details, loss = {}, {}, {}
    for row in report["cases"]:
        name = row["case"]["name"]
        a_path = root / "analysis" / name / "evaluation.npz"
        j_path = a_path.with_suffix(".json")
        sources.extend([a_path, j_path])
        with np.load(a_path, allow_pickle=False) as a:
            arrays[name] = {k: a[k] for k in a.files}
        details[name] = json.loads(j_path.read_text(encoding="utf-8"))
        a = arrays[name]
        old = np.isfinite(a["original_uv"]).all(-1).sum(axis=2)
        new = np.isfinite(a["candidate_uv"]).all(-1).sum(axis=2)
        loss[name] = (new < old).any(axis=(1, 2))
    primary = arrays["cg20s"]
    priority = primary["supported"] & primary["core"] & ~loss["cg20s"]
    priority_frames = primary["frames"][priority].tolist()
    if len(priority_frames) != 6:
        raise ValueError("Primary no-view-loss audit differs from reviewed6frame result")
    parts = [
        """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Medaka: re-evaluating 3-view and body-part correspondence</title><style>
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;color:#24333f;background:#f4f6f8;font:16px/1.8 system-ui,sans-serif}main{max-width:1180px;margin:auto;padding:28px}h1{font-size:28px}h2{font-size:23px;margin-top:38px}h3{font-size:18px}a{color:#145d8b}video{display:block;background:#111;width:100%;max-height:78vh}img{width:100%;height:auto;display:block}.lead{padding:20px;background:#e8f0f5;border-left:5px solid #28678c}.warn{padding:14px;background:#fff3d7;border-left:4px solid #b77400}.muted{color:#586c79;font-size:14px}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:14px}th,td{border-bottom:1px solid #cbd7de;padding:8px 12px;vertical-align:top;text-align:left}th{background:#e3eaf0}details{padding:12px;margin:12px 0;background:white;border:1px solid #d7e0e7;border-radius:5px}summary{cursor:pointer;font-weight:600}nav a{display:inline-block;margin:6px 18px 6px 0}select,input{font:inherit;padding:6px;border:1px solid #7892a2;border-radius:4px}video,img{border-radius:5px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:20px}section{scroll-margin-top:20px}.pill{font-size:13px;padding:3px 8px;background:#e8edf0;border-radius:4px}.supportloss{font-weight:700;color:#895600}[hidden]{display:none!important}:focus-visible{outline:3px solid #bd7100}code{overflow-wrap:anywhere}
</style><main><h1>Re-evaluating 3-view detection correspondence and the grouping of mouth, pectoral fin and tail</h1>
<p class="muted">2026-09-16 · C34 λ0.767 · native 166 fps · comparison with the original tracking retained</p>
<div class="lead"><strong>55 candidate intervals in the 20 s, 348 frames in total, were re-evaluated.</strong>
<p>In 129 frames a correspondence candidate different from the original was selected, and 24 frames passed the prespecified conditions on geometry, segment length and so on.
Checking the number of supporting views for each body part further, <strong>6 frames meet the conditions without reducing the number of views</strong>. These are set up for priority visual review.</p>
<p>The 12 known scenes and 6 control intervals were evaluated the same way. No candidate was accepted; this is not a result confirming individual-ID correction or fewer swaps.</p></div>
<nav><a href="#priority">6 priority frames</a><a href="#videos">Comparison videos</a><a href="#support">Caution on view counts</a><a href="#images">All comparison images</a><a href="#graphs">Graphs</a><a href="#numbers">Numbers and details</a></nav>
<h2>How to read</h2><p>For each camera, images show the original correspondence and the re-evaluated candidate one above the other; videos show them side by side.
Colours are the S0, S1, S2 tracking slots. <strong>Circle = selected 2D detection point, × and line = position projected from 3D back into the image, white box = changed detection point</strong>.
Lines join mouth → pectoral fin → tail in that order. If a projected position has an × but no circle, there is no actual observation from that view.
Videos are slow motion stretching 1 s of 166 frames to about 5.53 s, with no frames omitted.</p>"""
    ]

    def detail_at(name, frame):
        return next((d for d in details[name]["frames"] if d["frame"] == frame), None)

    def status_at(name, frame):
        a = arrays[name]
        t = int(np.searchsorted(a["frames"], frame))
        d = detail_at(name, frame)
        if not d or not a["changed"][t]:
            return "no correspondence change"
        if loss[name][t]:
            return "supporting views for a body part decrease: treat the residual drop with caution"
        return (
            "supporting-view count kept; priority visual review"
            if a["supported"][t]
            else "conditions not passed; not accepted"
        )

    def source_table(name, frame):
        a = arrays[name]
        t = int(np.searchsorted(a["frames"], frame))
        rows = []
        counts = [
            np.isfinite(a[key][t]).all(-1).sum(axis=1) for key in ("original_uv", "candidate_uv")
        ]
        for slot in range(3):
            for node, label in enumerate(("mouth", "pectoral fin", "tail")):
                old, new = int(counts[0][slot, node]), int(counts[1][slot, node])
                support = f"{old} → {new}"
                if new < old:
                    support = f'<span class="supportloss">{support} decrease</span>'
                own = a["original_source"][t, slot, :, node].tolist()
                cand = a["candidate_source"][t, slot, :, node].tolist()
                rows.append(
                    [
                        f"S{slot} {label}",
                        support,
                        f"{number(a['baseline_ray_mm'][t, slot, node])} → {number(a['candidate_ray_mm'][t, slot, node])}",
                        esc(own) + " → " + esc(cand),
                    ]
                )
        return table(
            [
                "Body part",
                "Supporting views",
                "Ray residual mm",
                "Detection rows [top, side1, side2] (-1 = none)",
            ],
            rows,
        )

    parts.append(
        '<h2 id="priority">The 6 frames that do not reduce supporting views</h2><p>This is an additional diagnostic applied to the 24 frames that passed the original review conditions. The prespecified conditions and analysis results were not rewritten. '
        "Not losing support is not a guarantee of correct ID; it is a check so that comparisons use the same amount of observation.</p>"
    )
    rows = []
    for frame in priority_frames:
        d = detail_at("cg20s", frame)
        changes = " / ".join(
            f"S{x['slot']} {x['view']} {x['node']} {x['original_row']}→{x['candidate_row']}"
            for x in d["source_changes"]
        )
        rows.append(
            [
                str(frame),
                esc(changes),
                f"{number(d['original_cost'])} → {number(d['candidate_cost'])}",
                esc(str(d["forward_valid"])) + " / " + esc(str(d["backward_valid"])),
            ]
        )
    parts.append(
        table(
            [
                "frame",
                "Changed correspondence",
                "Combined cost (lower = more consistent)",
                "Slots with a valid past / future reference",
            ],
            rows,
        )
    )
    parts.append(
        "<p>f76791 is a pectoral-fin correspondence candidate with usable references in both directions. At f75398–75399 and f76989–76992 the reference in one direction is insufficient, so ID retention before and after needs separate checking.</p>"
    )

    parts.append(
        '<h2 id="videos">Slow-motion comparison: original correspondence / re-evaluated candidate</h2>'
    )
    captions = {
        75398: "Candidate revising the body-part correspondence in the top view. Supporting views do not decrease, but the past-side reference is insufficient.",
        76791: "Candidate exchanging the S1 and S2 pectoral-fin correspondence in side 2. Supporting views are kept, and references before and after are usable.",
        76988: "The tail residual drops, but S1's tail goes from 3 views to 2. This cannot be interpreted as restored 3-view agreement. From the neighbouring 76989 onward there are also other candidates that use unused detections.",
        76235: "Candidate changing the mouth and tail correspondence in side 1. A pectoral-fin jump and out-of-tank nodes remain, and it does not meet the conditions.",
        76930: "A candidate that reconstructs the missing tail is selected, but pectoral fin–tail becomes about 37 mm, which does not meet the segment-length condition.",
        30735: "Candidate moving the tail observation to another slot, which leaves the original slot's 3D body part missing. It does not meet the review conditions.",
        31479: "The correspondence change lowers the combined cost, but some mouth and pectoral-fin ray residuals worsen. References before and after are also insufficient, and it does not meet the conditions.",
        323819: "A large 3-view inconsistency remains, and no correspondence change is selected at the centre frame. This scene calls for re-checking the effect of 2D detection and reflections.",
        77415: "The original correspondence remains at the centre frame. Whether body-part correspondence during close approach is correct cannot be confirmed from geometric conditions alone.",
    }
    videos = [e for e in entries if e["kind"] == "video"]
    order = {
        76791: 0,
        75398: 1,
        76988: 2,
        76235: 3,
        76930: 4,
        30735: 5,
        31479: 6,
        323819: 7,
        77415: 8,
    }
    for e in sorted(videos, key=lambda x: order.get(x["center_frame"], 99)):
        frame, name = e["center_frame"], e["case"]
        path = "review/" + e["path"]
        parts.append(
            f'<section><h3>f{frame} · {esc(status_at(name, frame))}</h3><p>{esc(captions.get(frame, "Compares a re-evaluated candidate that was not accepted."))}</p><video controls preload="metadata" src="{esc(path)}"></video>'
        )
        plots = [
            p
            for p in entries
            if p["kind"] == "plot"
            and p["case"] == name
            and p["center_frame"] == frame
            and (p.get("scope") == "local_video" or p["name"].endswith("_local_diagnostics"))
        ]
        if plots:
            p = "review/" + plots[0]["path"]
            parts.append(
                f'<details><summary>Error and segment-length graph for this 1 s</summary><a href="{esc(p)}"><img loading="lazy" src="{esc(p)}" alt="Local diagnostic graph for f{frame}"></a></details>'
            )
        parts.append(
            f"<details><summary>Per-body-part observations and residuals at the centre frame</summary>{source_table(name, frame)}</details></section>"
        )

    parts.append("""<h2 id="support">Distinguishing a residual drop from a loss of supporting views</h2>
<div class="warn">For S1's tail at f76988, the ray residual goes 8.29→0.69 mm while the supporting views drop 3→2.
S2 increases 2→3, and the tail residual worsens 0.32→1.21 mm. Counting only the total number of observations misses this trade.</div>
<p>A point reconstructed from 2 views tends to fit those 2 views. The check that predicts with the third view left out also becomes impossible to run for S1's tail in this candidate.
This residual drop is therefore not counted as an improvement in 3-view consistency.</p>""")
    support_plot = root / "review/tail_support_76988.png"
    if not support_plot.exists():
        raise FileNotFoundError(support_plot)
    sources.extend([support_plot, support_plot.with_suffix(".json")])
    parts.append(
        '<a href="review/tail_support_76988.png"><img loading="lazy" src="review/tail_support_76988.png" alt="Tail residual and supporting-view count over time"></a>'
    )

    parts.append(
        '<h2 id="images">All comparison images</h2><p>Includes representative frames from the 55 candidate intervals of the main interval, the 14 added by the backward direction, and the 12 known cases. Representative frames were chosen by the improvement in combined cost, so they are not a random sample of successes.</p>'
        '<label>Show <select id="filter"><option value="all">All</option><option value="priority">Passed conditions, no loss of supporting views</option><option value="loss">Candidates that lose supporting views</option><option value="known">Known scenes</option></select></label>'
    )
    for e in [x for x in entries if x["kind"] == "image"]:
        name, frame = e["case"], e["center_frame"]
        a = arrays[name]
        t = int(np.searchsorted(a["frames"], frame))
        path = "review/" + e["path"]
        flags = []
        if a["supported"][t] and not loss[name][t]:
            flags.append("priority")
        if a["changed"][t] and loss[name][t]:
            flags.append("loss")
        if name.startswith("known_"):
            flags.append("known")
        parts.append(
            f'<details class="sheet" data-types="{esc(" ".join(flags))}"><summary>{esc(name)} · f{frame} · {esc(status_at(name, frame))}</summary><p>{link(path, "Open full-size image")}</p><a href="{esc(path)}"><img loading="lazy" src="{esc(path)}" alt="Original correspondence and re-evaluated candidate for {esc(name)} f{frame}"></a>{source_table(name, frame)}</details>'
        )

    parts.append(
        '<h2 id="graphs">Whole-interval graphs</h2><p>Dashed lines are the original correspondence, solid lines the candidate. Grey bands are the re-evaluated frames. Segment length is per slot; other errors are the maximum over the displayed slots. '
        "The faint band for segment length is the reference median ± scale obtained outside the candidates; it is not a confidence interval. Missing observations are not filled in with lines.</p>"
    )
    for e in [x for x in entries if x["kind"] == "plot" and x.get("scope") == "full_case"]:
        path = "review/" + e["path"]
        parts.append(
            f'<details><summary>{esc(e["case"])} over time</summary><a href="{esc(path)}"><img loading="lazy" src="{esc(path)}" alt="Whole-interval diagnostic graph for {esc(e["case"])}"></a></details>'
        )
    parts.append(
        '<h2 id="numbers">Totals for all intervals</h2><p>Counts are native frame counts. The same neighbourhood of f76930 overlaps the main interval and a known case, so the sum is not interpreted as a count of independent observations.</p>'
    )
    rows = []
    for c in report["cases"]:
        name = c["case"]["name"]
        a = arrays[name]
        no_loss = int((a["supported"] & a["core"] & ~loss[name]).sum())
        rows.append(
            [
                link(f"analysis/{name}/evaluation.json", name),
                str(c["tested_frames"]),
                str(c["changed_frames"]),
                str(c["review_worthy_frames"]),
                str(no_loss),
            ]
        )
    parts.append(
        table(
            [
                "Interval / detail JSON",
                "Re-evaluated",
                "Different correspondence",
                "Passed prespecified conditions",
                "Of which no body-part support loss",
            ],
            rows,
        )
    )
    parts.append("""<p>Over the whole 20 s, pectoral-fin moves over 10 mm number 3 for both the original and the candidate. Some candidates improve the local fit, but this is not a result that resolves teleports or ID swaps.</p>
<h2>Scope of the evaluation</h2><p>The 216 combinations that reassign whole fish in each view, and substitutions with unused detections, were examined; starting from the original correspondence and the best whole-fish correspondence, candidates exchanging the mouth, pectoral fin, tail, or mouth + pectoral fin in one view were added.
This is a limited local search and does not cover every simultaneous exchange. Segment length and the available motion before and after were also evaluated, but no biological-ID ground-truth labels were used in candidate selection.</p>
<p>The residual is a consistency measure computed from the same detections. The leave-one-view-out check does not independently guarantee correct ID either. Chaining the proposals as they are can switch correspondence frame by frame, so duration, boundaries and IDs need to be checked.</p>
<p>The image projections compare the saved C34 with the candidate's reconstructed positions. Separately, residuals from re-triangulating both the same way were also checked. For unchanged body parts the recomputation difference is at most 0.0000105 mm, with no change in the number of supporting points.</p>""")
    audit_paths = [root / "support_audit.json", root / "refit_audit.json"]
    note_paths = sorted((root / "review").glob("*visual*notes*.json"))
    sources.extend([p for p in audit_paths + note_paths if p.exists()])
    parts.append(
        "<p>Additional review records: "
        + " · ".join(
            link(p.relative_to(root).as_posix(), p.stem)
            for p in audit_paths + note_paths
            if p.exists()
        )
        + "</p>"
    )
    parts.append(
        "<p>For reproduction: "
        + link("analysis/protocol.json", "Prespecified conditions")
        + " · "
        + link("analysis/summary.json", "All numbers")
        + " · "
        + link("review/manifest.json", "Image, video and graph record")
        + " · "
        + link("analysis/multiview_node_reassessment.provenance.json", "Analysis inputs and code")
        + "</p>"
    )
    parts.append(
        """</main><script>document.getElementById('filter').addEventListener('change',function(){for(const sheet of document.querySelectorAll('.sheet')){sheet.hidden=this.value!=='all'&&!sheet.dataset.types.split(' ').includes(this.value);}});</script></html>"""
    )
    document = "\n".join(parts)
    for path in re.findall(r'(?:href|src)="([^"]+)"', document):
        if not path.startswith("#") and not (root / html.unescape(path)).is_file():
            raise FileNotFoundError(path)
    with provenance("multiview_review_gallery", root) as rec:
        rec.input("builder", Path(__file__), sha256=sha256(Path(__file__)))
        for i, p in enumerate(sources):
            rec.input(f"source_{i}:{p.name}", p, sha256=sha256(p))
        rec.param("local_links_checked", True)
        rec.param("browser_interaction_verified", False)
        rec.param("primary_priority_frames_posthoc", priority_frames)
        target.write_text(document, encoding="utf-8")
        rec.output("gallery", target, sha256=sha256(target))
    print(target)


if __name__ == "__main__":
    main()
