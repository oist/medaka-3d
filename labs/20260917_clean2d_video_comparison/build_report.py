"""Source-backed, per-view 2D comparison report; never infer identity ground truth."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from medaka3d.aquarium import sha256
from medaka3d.provenance import provenance

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "work/clean2d_video_comparison_v1"
PLUGIN = Path("C:/Users/machi/.codex/plugins/cache/openai-curated-remote/data-analytics/1.0.8")
NODE = Path(
    "C:/Users/machi/AppData/Local/OpenAI/Codex/runtimes/cua_node/1b23c930bdf84ed6/bin/node.exe"
)
REVIEWS = ROOT / "docs/reviews/2026-09-16_multiview_identity_review.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")


def source(label, files, definition, components):
    files = [Path(p) for p in files if Path(p).is_file()]
    return dict(
        label=label,
        classification="observed",
        files=[rel(p) for p in files],
        hashes={rel(p): sha256(p) for p in files},
        metricDefinitions=[dict(label=label, definition=definition, componentIds=components)],
        evidenceFlow=[
            dict(
                title="Saved local evidence",
                detail=(
                    "Read saved JSON and NPZ metrics at their recorded native-frame scope; "
                    "no SQL or live service."
                ),
            ),
            dict(title="Adapter", detail=rel(__file__) + " SHA256 " + sha256(Path(__file__))),
        ],
    )


def normalize(summary, protocol):
    """Preserve exact core scopes, with input absence separated from association."""
    cases = [
        dict(id=c["name"], label=("main 20 s" if c["name"] == "cg20s" else c["name"]), **c)
        for c in protocol.get("cases", [])
    ]
    metrics, observations, sensitivity, reasons = [], [], [], []
    for record in summary.get("records", []):
        common = {k: record[k] for k in ("case", "view", "frame_start", "frame_stop")}
        baseline = record["arms"]["SLEAP"]
        observations.append(
            dict(
                **common,
                same_detection_rows_all_arms=record["same_detection_rows_all_arms"],
                **{
                    k: baseline[k]
                    for k in (
                        "native_frames",
                        "detection_rows",
                        "retained_rows",
                        "frames_without_detections",
                        "frames_below_three",
                        "missing_primary_rows",
                    )
                },
                **{
                    node + "_missing_rows": values["missing_rows"]
                    for node, values in baseline["nodes"].items()
                },
            )
        )
        for arm, values in record["arms"].items():
            flat = {
                **common,
                "condition": arm,
                **values,
                "large_primary_step_count": len(values["large_primary_steps"]),
            }
            for node, counts in values["nodes"].items():
                flat.update({node + "_" + key: value for key, value in counts.items()})
            (metrics if arm in ("SLEAP", "P", "N", "F") else sensitivity).append(flat)
            if arm == "F":
                sensitivity.append(flat)
            if arm == "F":
                reasons.extend(
                    dict(**common, reason=k, count=v)
                    for k, v in values.get("cut_reason_counts_nonexclusive", {}).items()
                )
    parameters = [
        dict(condition="P/N/F", parameter=k, value=v, basis="run settings in the saved protocol")
        for k, v in protocol.get("tracking_config", {}).items()
    ]
    settings = [
        dict(condition=arm, parameter=k, baseline=protocol["tracking_config"][k], value=v)
        for arm, config in protocol.get("sensitivity_configs", {}).items()
        for k, v in config.items()
        if v != protocol["tracking_config"][k]
    ]
    historical = [
        dict(
            case=c["name"],
            frame=int(c["note"]["frame_start"]),
            verbatim=c["note"]["annotator_note"],
            chain=c.get("note_chain"),
            start=c["start"],
            stop=c["stop"],
        )
        for c in protocol.get("cases", [])
        if "note" in c
    ]
    return dict(
        cases=cases,
        metrics=metrics,
        observations=observations,
        parameters=parameters,
        sensitivity=sensitivity,
        reasons=reasons,
        sensitivity_settings=settings,
        historical=historical,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=BASE)
    parser.add_argument("--media-manifest", type=Path)
    parser.add_argument("--complete", action="store_true")
    args = parser.parse_args()
    base = args.root.resolve()
    app, inputs = base / "report_app", base / "report_inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    summary_path = base / "summary.json"
    protocol_path = base / "protocol.json"
    protocol = read(protocol_path) if protocol_path.exists() else {}
    measured_paths = sorted(base.glob("*/metrics_*.json"))
    summary = (
        read(summary_path)
        if summary_path.exists()
        else dict(records=[read(p) for p in measured_paths])
    )
    normalized = normalize(summary, protocol)
    annotation = read(REVIEWS)
    notes = [
        dict(
            frame=e["frame"],
            verbatim=e["verbatim_note"],
            kind=e["kind"],
            original_video=e["video"],
            original_video_sha256=e["video_sha256"],
            panel=annotation.get("reviewed_panel"),
            exact_interval=e.get("human_exact_interval"),
        )
        for e in annotation["events"]
    ]
    data_source = source(
        "Saved 2D tracking metrics on the same detection points",
        [summary_path, protocol_path, *measured_paths],
        "Native range of each case, view and condition only. Separates detection rows, missing "
        "nodes, negative tracking IDs, links and fragments. Not a ground-truth series for "
        "measuring individual-ID accuracy.",
        [
            "summary",
            "metrics-table",
            "observations",
            "metric-fragments",
            "metric-links",
            "sensitivity",
            "reasons",
            "historical",
        ],
    )
    queries = {
        key: dict(rows=normalized[key], source=data_source)
        for key in (
            "metrics",
            "observations",
            "parameters",
            "sensitivity",
            "sensitivity_settings",
            "reasons",
            "historical",
        )
    }
    queries["human_reviews"] = dict(
        rows=notes,
        source=source(
            "Past human review, original text",
            [REVIEWS],
            "Review of an earlier 3D comparison video. 3D slot names differ from SLEAP's "
            "within-view IDs and are not reinterpreted as swap ground truth for 2D tracking.",
            ["review-note", "review-history"],
        ),
    )
    windows_path = base / "review_windows.json"
    window_rows = []
    if windows_path.exists():
        for record in read(windows_path)["records"]:
            for arm, values in record["arms"].items():
                window_rows.append(
                    {
                        "case": record["case"],
                        "view": record["view"],
                        "center_frame": record["anchor"],
                        "condition": arm,
                        "scope": (
                            "centre ±8 (17 frames)"
                            if record["window_kind"] == "anchor_17_native_frames"
                            else "video range (166 frames)"
                        ),
                        "window_kind": record["window_kind"],
                        **values,
                        "large_primary_step_count": len(values["large_primary_steps"]),
                    }
                )
    queries["local_metrics"] = dict(
        rows=window_rows,
        source=source(
            "Recount over the central 17 frames and the 166 video frames",
            [windows_path],
            "Each from frame_start inclusive to frame_stop exclusive. Breaks in the wide known "
            "interval are not confused with breaks at the annotation centre. "
            "Direct 2D metrics of SLEAP/P/N/F/F_all_loose on the same detections.",
            ["local-metrics"],
        ),
    )
    audit_path = base / "parameter_audit.json"
    parameter_audit = read(audit_path) if audit_path.exists() else {}
    if parameter_audit:
        queries["parameters"]["rows"].extend(
            dict(
                condition=f"saved C27 {view}",
                parameter=k,
                value=str(v),
                basis="saved effective_tracker_config",
            )
            for view, config in parameter_audit["saved_c27_effective_config"].items()
            for k, v in config.items()
        )
        requested = parameter_audit["saved_c27_requested_parameters"]
        queries["parameters"]["rows"].extend(
            dict(
                condition="saved C27 all views",
                parameter=k,
                value=json.dumps(requested[k], ensure_ascii=False),
                basis="saved requested parameters",
            )
            for k in ("nodes", "scoring", "window_size", "min_match_points", "whole_recording")
        )
        queries["parameters"]["source"] = source(
            "Run settings of saved C27 and fixed P/N/F",
            [protocol_path, audit_path],
            "C27 from the saved effective config, P/N/F from the run protocol. A synthetic audit "
            "of the currently installed implementation is kept distinct from an exact "
            "reproduction of the original run environment.",
            ["parameters", "limitations"],
        )
    queries["parameter_mapping"] = dict(
        rows=parameter_audit.get("parameter_mapping", []),
        source=source(
            "Audit of tracking method versus parameter correspondence",
            [audit_path],
            "Thresholds with the same name do not make the same experiment if the candidate "
            "set, history or assignment rule differs.",
            ["parameter-mapping", "replay-notes"],
        ),
    )
    queries["parameter_audit"] = dict(
        rows=[parameter_audit] if parameter_audit else [],
        source=source(
            "Audit of the current implementation, saved parameters and local histories on real "
            "data",
            [audit_path, base / "c27_replay/summary.json"],
            "Does not reproduce the internal scores of the past whole-recording run. "
            "The sigma comparison re-scores fixed local histories; it is not a sensitivity "
            "test of full tracking.",
            ["scoring-notes", "software-notes"],
        ),
    )
    c27_path = base / "c27_sensitivity/summary.json"
    c27_rows = []
    if c27_path.exists():
        for record in read(c27_path)["records"]:
            c27_rows.append(
                {
                    "case": record["case"],
                    "view": record["view"],
                    "window": record["window"],
                    **record["metrics"],
                    "large_primary_step_count": len(record["metrics"]["large_primary_steps"]),
                    "npz_sha256": record["npz_sha256"],
                    "readout_different_rows_core": record["readout_different_rows_core"],
                    "maximum_pectoral_step_px": record["metrics"]["nodes"]["pectoral"][
                        "max_adjacent_step_px"
                    ],
                }
            )
    queries["c27_sensitivity"] = dict(
        rows=c27_rows,
        source=source(
            "SLEAP history-length comparison on the same local intervals and environment",
            [c27_path, base / "c27_sensitivity/protocol.json"],
            "Row-faithful output with only window=1/5/10 changed. The history differs from the "
            "saved full-session C27, so it is not treated as an exact reproduction of it.",
            ["c27-sensitivity"],
        ),
    )
    queries["media"] = dict(
        rows=[],
        source=source(
            "Direct comparison video per view",
            [],
            "Compares the 4 conditions SLEAP/P/N/F on the same raw frames and the same "
            "detection points. NEW and UNASSIGNED do not mean the detection disappeared.",
            ["comparison-video"],
        ),
    )
    existing = read(app / "src/data.json") if (app / "src/data.json").exists() else {}
    if args.complete and not normalized["metrics"]:
        raise ValueError("Cannot complete a report without measured data")
    snapshot = dict(
        surface="report",
        title="SLEAP versus custom tracking, per view, on the same 2D detections",
        generatedAt=datetime.now(UTC).isoformat(),
        buildStatus="complete" if args.complete else "updating",
        status="reviewed" if args.complete else "partial",
        filters=[],
        queries=queries,
        cases=normalized["cases"],
        report={"asOf": "2026-09-17"},
        summaryMarkdown="## Direct 2D comparison in preparation\n\nIn each view, SLEAP and P/N/F "
        "are overlaid on the same detections. Missing detections, node dropouts, unassigned "
        "rows and fragment breaks are checked separately.",
        interpretationMarkdown="This report is a 2D comparison. An observation not adopted in "
        "the previous 3D run is not counted as a 2D detection having disappeared.",
    )
    snapshot["protocol"] = protocol
    if parameter_audit:
        snapshot["parameterNotes"] = (
            "Saved C27 uses mouth/pectoral, OKS, Hungarian, a 5-observation history and no "
            "max_tracks. window=5 is not expiry after 5 native frames. P/N/F use bidirectional "
            "agreement between adjacent frames and explicit refusal conditions. They cannot be "
            "compared on numerical strictness alone."
        )
    if normalized["metrics"]:
        snapshot["summaryMarkdown"] = (
            "## Detections remain; links across time are cut\n\n"
            "All 29,386 detections in the main 20 s remain under every condition: SLEAP, P, N "
            "and F. Contiguous 2D fragments: SLEAP 224 versus F 1,324. F does not delete points; "
            "it cuts uncertain links and assigns new fragment numbers.\n\n"
            "743 detection rows lack mouth or pectoral. "
            "Of the 1,129 cases where F cut or changed a SLEAP link, in 720 the refusal reasons "
            "include a missing primary node at an endpoint (overlapping with other reasons). "
            "Even with all thresholds loosened together, F has 1,028 fragments. The required-node "
            "condition was held fixed in this experiment.\n\n"
            "Fewer fragments or smoother motion alone cannot establish that individual IDs are "
            "correct."
        )
    snapshot["limitationsMarkdown"] = (
        "## Comparison conditions and limits\n\nSLEAP is the saved full-session C27 tracking; "
        "P/N/F are processing on the same C01 detection points using 332 native frames of "
        "context before and after. P selects candidates by pectoral and N by mouth + pectoral, "
        "using tail for the consistency check; F additionally uses LK flow. "
        "P/N/F link only on bidirectional agreement between adjacent frames and do not bridge "
        "2D gaps. Input detection points and node groupings are not modified.\n\n"
        "Known intervals of the same 3 fish are re-evaluated. With no 2D ID ground truth for "
        "all individuals, swap accuracy and superiority over the old method are not confirmed. "
        "The past S0/S1/S2 are 3D-side slot names, distinct from the per-view SLEAP numbers."
    )
    if "id" in existing:
        snapshot["id"] = existing["id"]
    findings_path = base / "case_findings.json"
    if findings_path.exists():
        snapshot["queries"]["case_findings"] = dict(
            rows=read(findings_path)["records"],
            source=source(
                "Re-evaluation of 8 known scenes",
                [findings_path, windows_path, base / "inspection/manifest.json", audit_path],
                "Numbers over the 17 native frames at centre ±8; visual check over the 3 frames "
                "at centre ±1. Checked by the assistant; not new human ID ground truth.",
                ["case-findings"],
            ),
        )
    narrative_path = base / "final_narrative.json"
    if not narrative_path.exists():
        narrative_path = inputs / "final_narrative.json"
    if narrative_path.exists():
        narrative = read(narrative_path)
        snapshot.update(
            {
                k: v
                for k, v in narrative.items()
                if k
                in {
                    "title",
                    "summaryMarkdown",
                    "interpretationMarkdown",
                    "parameterNotes",
                    "limitationsMarkdown",
                    "resultNotes",
                    "sensitivityNotes",
                    "replayNotes",
                    "historyNotes",
                    "scoringNotes",
                    "softwareNotes",
                }
            }
        )
        snapshot["queries"]["narrative"] = dict(
            rows=[dict(key=k, text=v) for k, v in narrative.items() if isinstance(v, str)],
            source=source(
                "Interpretation checked against measurements, raw frames and the settings audit",
                [
                    narrative_path,
                    summary_path,
                    windows_path,
                    audit_path,
                    c27_path,
                    base / "audit.json",
                    ROOT / "work/clean_tracklets_v2/summary.json",
                ],
                "Report text checked against the original numbers, definitions and annotations. "
                "Not an addition of ID ground truth.",
                [
                    "summary",
                    "result-notes",
                    "sensitivity-notes",
                    "replay-notes",
                    "limitations",
                    "scoring-notes",
                    "software-notes",
                ],
            ),
        )
    snapshot_path = inputs / "reviewed_snapshot.json"
    write(snapshot_path, snapshot)
    if not (app / "AGENTS.md").exists():
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
        snapshot = read(app / "src/data.json")
    manifest_path = args.media_manifest or base / "review/manifest.json"
    imports, mappings, media_rows = [], [], []
    assets = app / "src/content/assets"
    assets.mkdir(exist_ok=True)
    if manifest_path.exists():
        manifest = read(manifest_path)
        for entry in manifest["assets"]:
            for name, kind in (("video", "video"), ("still", "image")):
                original = Path(entry[name])
                if not original.is_absolute():
                    original = manifest_path.parent / original
                digest = sha256(original)
                if digest != entry[name + "_sha256"]:
                    raise ValueError(f"Media hash mismatch: {original}")
                asset_name = original.stem + "_" + digest[:12] + original.suffix
                shutil.copy2(original, assets / asset_name)
                key = f"m{len(imports)}"
                imports.append(f'import {key} from "../assets/{asset_name}";')
                mappings.append(f"  {json.dumps(rel(original))}: {key}")
                media_rows.append(
                    {
                        **entry,
                        "kind": kind,
                        "path": rel(original),
                        "asset_key": rel(original),
                        "sha256": digest,
                    }
                )
        snapshot["queries"]["media"] = dict(
            rows=media_rows,
            source=source(
                "Direct comparison video per view",
                [manifest_path],
                "Shows every one of the same detections and compares only the per-condition "
                "temporal IDs. Local clips play 166 native frames at 30fps without "
                "interpolation. Where both an original and a report version exist, both SHAs "
                "are recorded.",
                ["comparison-video"],
            ),
        )
    inspection_path = base / "inspection/manifest.json"
    inspection_rows = []
    if inspection_path.exists():
        for entry in read(inspection_path)["assets"]:
            if (entry["anchor"], entry["view"]) not in {(21232, "side2_cam"), (333619, "top_cam")}:
                continue
            original = ROOT / entry["path"]
            digest = sha256(original)
            assert digest == entry["sha256"]
            asset_name = original.stem + "_" + digest[:12] + original.suffix
            shutil.copy2(original, assets / asset_name)
            key = f"m{len(imports)}"
            imports.append(f'import {key} from "../assets/{asset_name}";')
            mappings.append(f"  {json.dumps(rel(original))}: {key}")
            inspection_rows.append(
                {**entry, "kind": "image", "path": rel(original), "asset_key": rel(original)}
            )
    snapshot["queries"]["inspection"] = dict(
        rows=inspection_rows,
        source=source(
            "Check using raw-frame crops over 3 native frames",
            [inspection_path, base / "c27_replay/summary.json"],
            "The same crop is applied to 3 native frames. The display shows measured 2D points "
            "and tracking numbers; no new human ID ground truth is given.",
            ["inspection"],
        ),
    )
    if args.complete:
        assert summary.get("complete") and len(summary["records"]) == 33
        assert read(audit_path)["status"] == "complete"
        assert read(c27_path)["status"] == "complete"
        assert read(base / "audit.json")["status"] == "passed"
        assert read(manifest_path)["status"] in {"complete", "verified"}
        assert len(media_rows) == 84 and len(inspection_rows) == 2
        assert narrative_path.is_file()
    with provenance("clean2d_report_snapshot", inputs, name="snapshot") as rec:
        for path in [
            Path(__file__),
            REVIEWS,
            summary_path,
            protocol_path,
            windows_path,
            audit_path,
            c27_path,
            *measured_paths,
            manifest_path,
            inspection_path,
            findings_path,
            narrative_path,
        ]:
            if path.is_file():
                rec.input(rel(path), path, sha256=sha256(path))
        write(snapshot_path, snapshot)
        write(app / "src/data.json", snapshot)
        (app / "src/content/report/media.js").write_text(
            "\n".join(imports) + "\nexport const media = {\n" + ",\n".join(mappings) + "\n};\n",
            encoding="utf-8",
            newline="\n",
        )
        rec.output("snapshot", snapshot_path, sha256=sha256(snapshot_path))
    print(json.dumps(dict(app=rel(app), metrics=len(normalized["metrics"]), media=len(media_rows))))


if __name__ == "__main__":
    main()
