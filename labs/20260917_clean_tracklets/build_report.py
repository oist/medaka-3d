"""Make a source-backed snapshot for the separate clean-fragment report app."""
# Typographic multiplication signs and en dashes in report text are intentional.
# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from medaka3d.aquarium import sha256
from medaka3d.provenance import provenance

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "work/clean_tracklets_v1"
PLUGIN = Path("C:/Users/machi/.codex/plugins/cache/openai-curated-remote/data-analytics/1.0.8")
NODE = Path(
    "C:/Users/machi/AppData/Local/OpenAI/Codex/runtimes/cua_node/1b23c930bdf84ed6/bin/node.exe"
)
REVIEWS = ROOT / "docs/reviews/2026-09-16_multiview_identity_review.json"
CONDITIONS = {
    "saved_C34": "saved C34",
    "previous_custom_C_v3": "previous C v3",
    "initial_clean_v1": "initial F→3D v1",
    "new": "new F→3D v2",
}
CASES = ("cg20s", "known_30735", "known_31479", "known_323819", "control_50000", "holdout20s")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")


def source(label, paths, definition, components):
    paths = [Path(p) for p in paths if Path(p).is_file()]
    return dict(
        label=label,
        classification="observed",
        files=[rel(p) for p in paths],
        hashes={rel(p): sha256(p) for p in paths},
        metricDefinitions=[dict(label=label, definition=definition, componentIds=components)],
        evidenceFlow=[
            dict(
                title="Saved local evidence",
                detail="Read saved JSON and NPZ outputs; no remote queries or manual reassignment.",
            ),
            dict(
                title="Report adapter", detail=rel(__file__) + " SHA256 " + sha256(Path(__file__))
            ),
        ],
    )


def metric_rows(case):
    tracking, geometry, intervals, bridges, flows = [], [], [], [], []
    for arm, views in case["tracking"].items():
        for view, values in views.items():
            tracking.append(
                dict(
                    case=case["name"],
                    condition=arm,
                    view=view,
                    **{k: v for k, v in values.items() if not isinstance(v, dict)},
                    median_fragment_frames=values["fragment_length_frames"]["median"],
                    max_fragment_frames=values["fragment_length_frames"]["maximum"],
                    maximum_step_px=values["step_px"]["maximum"],
                )
            )
    for key, label in CONDITIONS.items():
        if key not in case["comparison"]:
            continue
        result = case["comparison"][key]
        g, f = result["geometry"], result["female"]
        fc = f.get("interval_counts", {})
        geometry.append(
            dict(
                case=case["name"],
                condition=label,
                condition_id=key,
                **{k: v for k, v in g.items() if not isinstance(v, dict)},
                maximum_step_mm=g["step_mm"]["maximum"],
                female_requested_intervals=f.get("requested_intervals"),
                female_preserved=fc.get("preserved"),
                female_swap=fc.get("swap"),
                female_unscored=fc.get("unscored"),
            )
        )
        for interval in f.get("intervals", []):
            intervals.append(dict(case=case["name"], condition=label, **interval))
    for i, item in enumerate(case.get("bridges", [])):
        bridges.append(dict(case=case["name"], proposal=i, **item))
    for view, values in case.get("flow", {}).items():
        flows.append(
            dict(case=case["name"], view=view, metadata=json.dumps(values, ensure_ascii=False))
        )
    return tracking, geometry, intervals, bridges, flows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=BASE)
    parser.add_argument(
        "--analysis", type=Path, help="Versioned inputs; report app stays under --root"
    )
    parser.add_argument("--complete", action="store_true")
    parser.add_argument(
        "--media-manifest", type=Path, help="Verified report-size media derivatives"
    )
    args = parser.parse_args()
    base, app = args.root.resolve(), args.root.resolve() / "report_app"
    analysis = args.analysis.resolve() if args.analysis else base
    paths = [
        analysis / name / "summary.json"
        for name in CASES
        if (analysis / name / "summary.json").exists()
    ]
    cases = [read(p) for p in paths]
    evidence_paths = [*paths, *[analysis / c["name"] / "reconstruction.npz" for c in cases]]
    evidence_paths.extend(
        BASE / c["name"] / "reconstruction.npz"
        for c in cases
        if "initial_clean_v1" in c["comparison"]
    )
    protocol_path = analysis / "protocol.json"
    annotation = read(REVIEWS)
    review_rows = [
        dict(
            frame=e["frame"],
            verbatim=e["verbatim_note"],
            kind=e["kind"],
            reviewed_video=e["video"],
            reviewed_video_sha256=e["video_sha256"],
            reviewed_panel=annotation.get("reviewed_panel", "unspecified"),
            exact_interval=e.get("human_exact_interval"),
        )
        for e in annotation["events"]
    ]
    methods = [
        dict(
            condition="P",
            name="Pectoral-fin point only",
            description=(
                "Per view, uses pectoral-fin motion and forward/backward correspondence "
                "agreement, splitting into short fragments where ambiguous."
            ),
        ),
        dict(
            condition="N",
            name="3-node motion",
            description=(
                "Adds consistency of mouth, pectoral-fin and tail motion. Nodes within a "
                "detection row are not redistributed to another fish."
            ),
        ),
        dict(
            condition="F",
            name="3 nodes + image flow",
            description=(
                "Verifies each node's motion with image flow over 1 native frame before and "
                "after. 3D is reconstructed from the F fragments."
            ),
        ),
    ]
    tracking, geometry, intervals, bridges, flows = [], [], [], [], []
    for c in cases:
        rows = metric_rows(c)
        with np.load(analysis / c["name"] / "reconstruction.npz", allow_pickle=False) as z:
            core = (z["frames"] >= c["start"]) & (z["frames"] < c["stop"])
            present = np.isfinite(z["xyz"][core]).all(-1)
            held = z["identity_review"][core]
            measured_held = int((present & held).sum())
        for row in rows[1]:
            row["measured_and_review_fish_frames"] = (
                measured_held if row["condition_id"] == "new" else None
            )
            if row["condition_id"] == "initial_clean_v1":
                with np.load(BASE / c["name"] / "reconstruction.npz", allow_pickle=False) as z:
                    old_core = (z["frames"] >= c["start"]) & (z["frames"] < c["stop"])
                    old_present = np.isfinite(z["xyz"][old_core]).all(-1)
                    row["measured_and_review_fish_frames"] = int(
                        (old_present & z["identity_review"][old_core]).sum()
                    )
            elif row["condition_id"] != "new":
                row["identity_review_fish_frames"] = None
            elif "initial_clean_v1" not in c["comparison"]:
                row["condition"] = "initial F→3D v1"
        for destination, values in zip(
            (tracking, geometry, intervals, bridges, flows), rows, strict=True
        ):
            destination.extend(values)
    data_source = source(
        "Saved metrics of the frozen experiment",
        evidence_paths,
        (
            "Evaluation range only. 2D incoming edges have their endpoint inside the range and "
            "include entries from the preceding context. 3D steps are only between adjacent "
            "native frames with both ends inside the range. Fish × frame, observed steps and "
            "detection rows are kept as separate denominators. Observed ID hold is the count of "
            "finite(xyz) and identity_review. identity_review as a whole also includes missing "
            "observations. Past methods have no equivalent hold label, so null. "
            "Small moves or holds are not proof of correct ID."
        ),
        [
            "report-summary",
            "tracking-table",
            "geometry-table",
            "all-cases",
            "female-table",
            "metric-fragments",
            "metric-edges",
            "metric-jumps",
            "metric-missing",
            "metric-held",
            "metric-coverage",
        ],
    )

    def query(rows, src):
        return dict(rows=rows, source=src)

    queries = {
        "human_reviews": query(
            review_rows,
            source(
                "8 human reviews of past videos, original text",
                [REVIEWS],
                "Original text preserved. The frame is a marker; left/right panel and exact "
                "boundaries are unspecified. Not reused for ground-truth judgement of the new "
                "method.",
                ["review-cases", "review-notes"],
            ),
        ),
        "methods": query(
            methods,
            source(
                "Experiment specification and frozen settings",
                [Path(__file__).with_name("README.md"), protocol_path],
                "P/N/F are a 2D diagnostic comparison. The new 3D takes only F as input. Past "
                "temporal IDs, 3D trajectories and human annotations are not used for "
                "estimation, only for comparison.",
                ["methods", "limitations"],
            ),
        ),
        "tracking": query(tracking, data_source),
        "geometry": query(geometry, data_source),
        "female_intervals": query(intervals, data_source),
        "bridges": query(
            bridges,
            source(
                "Separate-layer link proposals for 2D fragments",
                paths,
                (
                    "Candidate log including 332 frames of context before and after. accepted "
                    "is a separate-layer 2D completion; the initial clean edges and 3D "
                    "observations are not changed. Overlapping scenes are not summed as "
                    "independent events."
                ),
                ["join-table", "join-summary"],
            ),
        ),
        "flow": {
            **query(
                flows,
                source(
                    "Image-flow input signature and validity diagnostics",
                    paths,
                    (
                        "LK on half-resolution Gaussian pyrDown. "
                        "Forward/backward roundtrip, patch error etc. verified under fixed "
                        "conditions. Processes native ±1frame without using IDs. Not a "
                        "whole-video SHA but size/mtime/SHA of head and tail samples."
                    ),
                    ["flow-provenance"],
                ),
            ),
            "payloadColumns": ["metadata"],
        },
    }
    assets = app / "src/content/assets"
    imported, mappings = [], []
    media_paths = []
    for query_id, manifest_path in (
        (
            "media",
            args.media_manifest.resolve()
            if args.media_manifest
            else analysis / "review/manifest.json",
        ),
        ("event_figures", analysis / "event_figures/manifest.json"),
    ):
        rows = []
        if manifest_path.exists():
            manifest = read(manifest_path)
            entries = list(manifest.get("entries", []))
            for item in manifest.get("assets", []):
                for name, kind in (("video", "video"), ("still", "image")):
                    if item.get(name):
                        entries.append(
                            {
                                **item,
                                "kind": kind,
                                "path": item[name],
                                "sha256": item[name + "_sha256"],
                            }
                        )
            for entry in entries:
                p = Path(entry["path"])
                if not p.is_absolute():
                    p = manifest_path.parent / p
                digest = sha256(p)
                if digest != entry["sha256"]:
                    raise ValueError(f"Media SHA mismatch: {p}")
                assets.mkdir(parents=True, exist_ok=True)
                name = p.stem + "_" + digest[:12] + p.suffix
                shutil.copy2(p, assets / name)
                key = f"m{len(imported)}"
                imported.append(f'import {key} from "../assets/{name}";')
                mappings.append(f"  {json.dumps(rel(p))}: {key}")
                rows.append({**entry, "path": rel(p), "asset_key": rel(p)})
                media_paths.append(p)
            if query_id == "event_figures":
                queries["review_windows"] = query(
                    manifest.get("window_metrics", []),
                    source(
                        "Diagnostics in 8 scenes, within 166 native frames",
                        [manifest_path],
                        "Clip range is center-83:center+83. Moves are adjacent pectoral-fin 3D "
                        "steps with both ends finite inside the range. Holds and missing "
                        "observations are counted separately. Old outputs have no hold label "
                        "of the new method.",
                        ["window-table"],
                    ),
                )
        queries[query_id] = query(
            rows,
            source(
                "Visualisations made from raw images and saved coordinates",
                [manifest_path],
                (
                    "Videos show all 3 views and all 3 fish. Local clips play 166 native frames "
                    "at 30fps; the main 20 s overview is decimated following source time. No "
                    "interpolation. X in figures is the 3D projected prediction, distinct from "
                    "observed points."
                ),
                ["review-media", "overview-video", "event-plot"],
            ),
        )
    stable_id = read(app / "src/data.json").get("id") if app.exists() else None
    complete = args.complete and len(cases) == len(CASES)
    if args.complete and not complete:
        raise ValueError("Cannot complete a report with missing experiment cases")
    snapshot = dict(
        surface="report",
        title="Joining 3D from short 2D fragments: observations and holds compared separately",
        generatedAt=datetime.now(UTC).isoformat(),
        status="reviewed" if complete else "partial",
        buildStatus="complete" if complete else "updating",
        filters=[],
        report={
            "asOf": max(
                datetime.fromtimestamp(p.stat().st_mtime, UTC).date().isoformat()
                for p in [REVIEWS, *paths]
            )
        },
        experiment=dict(
            id="20251225_K_1_2_FCMC",
            fps=166,
            status="complete" if complete else "running",
            analysis_directory=rel(analysis),
        ),
        cases=[
            dict(
                id=c["name"],
                label=f"{c['name']} · f{c['start']}–{c['stop'] - 1}",
                start=c["start"],
                stop=c["stop"],
            )
            for c in cases
        ],
        queries=queries,
        summaryMarkdown=(
            "## Comparison experiment running\n\nUpdated progressively from saved results. "
            "Unmeasured values are not treated as 0."
        ),
        methodNotes=(
            "P/N/F are a 2D comparison. Only F fragments are input to 3D, and observations are "
            "selected from refractive camera geometry, 3D motion and agreement of clean "
            "fragments across independent views. The separate-layer completion J is recorded "
            "with the initial fragments F kept, and is not used to turn missing pixels into "
            "observations or to modify 3D coordinates."
        ),
        limitationsMarkdown=(
            "## How to read the results\n\n"
            "- Alongside any reduction in 3D jumps, check the denominators of missing "
            "observations, holds and observed steps. Smoothness from more holds is not called "
            "an improvement in ID accuracy.\n"
            "- The 8 human reviews are original text on past videos. ID corrections by the new "
            "method are not verified, and there is no ground-truth series for all 3 fish, "
            "including male versus male.\n"
            "- This is one group of the same 3 fish. The other 20 s is also an interval examined "
            "previously, not an independent new holdout. Denominators of the female endpoint "
            "annotation are kept per interval.\n"
            "- All detection rows are kept in 2D. Mixed nodes within a detection are not "
            "retrained or redistributed; the design rejects inconsistent bundles."
        ),
    )
    if stable_id:
        snapshot["id"] = stable_id
    primary = next((c for c in cases if c["name"] == "cg20s"), None)
    if primary:
        comparisons = primary["comparison"]
        old, new = comparisons["saved_C34"]["geometry"], comparisons["new"]["geometry"]
        observed_held = next(
            r["measured_and_review_fish_frames"]
            for r in geometry
            if r["case"] == "cg20s" and r["condition_id"] == "new"
        )
        female = comparisons["new"]["female"].get("interval_counts", {})
        snapshot["summaryMarkdown"] = (
            "## Main 20 s result\n\n"
            "From saved C34 → new F→3D, moves over 10 mm went "
            f"**{old['steps_over_10mm']} → {new['steps_over_10mm']}**, and missing went "
            f"**{old['missing_fish_frames']} → {new['missing_fish_frames']} / "
            f"{new['possible_fish_frames']} fish-frames**. Observed ID holds under the new "
            f"method are **{observed_held} fish-frames** (shown separately from missing).\n\n"
            "Female endpoints under the new method: kept "
            f"**{female.get('preserved', 'not measured')}**, "
            f"changed **{female.get('swap', 'not measured')}**, "
            f"undeterminable **{female.get('unscored', 'not measured')}**. "
            "Fewer moves alone cannot confirm fewer ID swaps."
        )
    narrative_path = base / "report_inputs/final_narrative.json"
    if narrative_path.exists():
        narrative = read(narrative_path)
        for key in (
            "title",
            "summaryMarkdown",
            "methodNotes",
            "limitationsMarkdown",
            "resultNotes",
        ):
            if key in narrative:
                snapshot[key] = narrative[key]
        snapshot["queries"]["narrative"] = query(
            [narrative],
            source(
                "Interpretation after the quantitative audit",
                [narrative_path],
                "Final interpretation saved separately. Numbers refer to the result queries.",
                ["report-summary"],
            ),
        )
    inputs = base / "report_inputs"
    inputs.mkdir(exist_ok=True)
    snapshot_path = inputs / "reviewed_snapshot.json"
    with provenance("clean_tracklets_report_snapshot", inputs, name="snapshot") as rec:
        for p in [REVIEWS, *evidence_paths, *([narrative_path] if narrative_path.exists() else [])]:
            rec.input(rel(p), p, sha256=sha256(p))
        rec.input("adapter", Path(__file__), sha256=sha256(Path(__file__)))
        write(snapshot_path, snapshot)
        if not (app / "AGENTS.md").exists():
            if app.exists():
                # Only media was staged: defer staging until after the preparer.
                staged = inputs / "staged_assets"
                staged.mkdir(exist_ok=True)
                for p in assets.glob("*"):
                    shutil.move(str(p), staged / p.name)
                assets.rmdir()
                assets.parent.rmdir()
                assets.parent.parent.rmdir()
                app.rmdir()
            subprocess.run(
                [
                    str(NODE),
                    str(PLUGIN / "scripts/prepare-data-app.mjs"),
                    "--surface",
                    "report",
                    "--output",
                    str(app),
                    "--snapshot",
                    str(snapshot_path),
                ],
                check=True,
            )
            assets.mkdir(exist_ok=True)
            staged = inputs / "staged_assets"
            if staged.exists():
                for p in staged.glob("*"):
                    shutil.copy2(p, assets / p.name)
            snapshot = read(app / "src/data.json")
            write(snapshot_path, snapshot)
        else:
            write(app / "src/data.json", snapshot)
        (app / "src/content/report/media.js").write_text(
            "\n".join(imported) + "\nexport const media = {\n" + ",\n".join(mappings) + "\n};\n",
            encoding="utf-8",
            newline="\n",
        )
        rec.output("snapshot", snapshot_path, sha256=sha256(snapshot_path))
        rec.output("app_data", app / "src/data.json", sha256=sha256(app / "src/data.json"))
    print(
        json.dumps(
            dict(
                cases=len(cases), assets=len(media_paths), snapshot=rel(snapshot_path), app=rel(app)
            )
        )
    )


if __name__ == "__main__":
    main()
