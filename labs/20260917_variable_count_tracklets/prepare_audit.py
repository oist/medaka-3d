"""Metadata-only temporal selection and preservation snapshot before new results."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from medaka3d.provenance import provenance

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work/variable_count_tracklets_v1"


def sha(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "selection_audit.json"
    if target.exists():
        raise FileExistsError("Selection already frozen")
    exclusions, sources = [], {}

    def source(relative):
        path = ROOT / relative
        sources[relative] = sha(path)
        return path

    for relative in (
        "work/clean2d_video_comparison_v1/protocol.json",
        "work/custom2d_comparison_v1/analysis_v3/protocol.json",
        "work/bidirectional_reconnect_v1/analysis_v2/protocol.json",
        "labs/20260908_generative_tracking/reacquisition_windows.json",
    ):
        doc = json.loads(source(relative).read_text(encoding="utf-8"))
        for case in doc.get("cases", doc.get("windows", [])):
            if "start" in case and "stop" in case:
                exclusions.append(
                    dict(
                        start=case["start"] - 332,
                        stop=case["stop"] + 332,
                        source=relative,
                        kind="prior_window_and_context",
                    )
                )
    relative = "labs/20260909_three_body/long_input_protocol.json"
    doc = json.loads(source(relative).read_text(encoding="utf-8"))
    lo, hi = doc["source_frame_range"]
    exclusions.append(
        dict(start=lo - 332, stop=hi + 332, source=relative, kind="prior_long_target")
    )
    for frame in doc["background_frames"]:
        exclusions.append(
            dict(start=frame - 332, stop=frame + 333, source=relative, kind="background_frame")
        )
    base = "work/generative_tracking_inputs/assets/groundtruth/"
    relative = base + "fish_sex_gt/20251225_K_1_2_FCMC/female_sex_gt.csv"
    with source(relative).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            frame = int(row["frame"])
            exclusions.append(
                dict(
                    start=frame - 332,
                    stop=frame + 333,
                    source=relative,
                    kind="female_annotation_anchor",
                )
            )
    relative = base + "swap_event_gt/20251225_K_1_2_FCMC/swap_events.csv"
    with source(relative).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            exclusions.append(
                dict(
                    start=int(row["frame_start"]) - 332,
                    stop=int(row["frame_end"]) + 333,
                    source=relative,
                    kind="historical_review_anchor",
                )
            )
    chosen = None
    for start in range(332, 348600 - 3652, 1660):
        stop = start + 3320
        if not any(
            start - 332 < item["stop"] and stop + 332 > item["start"] for item in exclusions
        ):
            chosen = dict(
                name=f"validation20s_{start:05}",
                start=start,
                stop=stop,
                category="metadata_selected_post_freeze_same_trio",
                render=False,
                centers=[],
                context_start=start - 332,
                context_stop=stop + 332,
            )
            break
    assert chosen is not None
    audit = dict(
        selected_at=datetime.now(UTC).isoformat(),
        selected_case=chosen,
        selection_rule="First 3320-frame window on start=332+1660*k lattice with full "
        "332-frame context disjoint from metadata exclusions (also padded 332).",
        sources=sources,
        exclusions=exclusions,
        selection_used_tracking_outcomes=False,
        known_limitations=[
            "New for this local comparison under the enumerated metadata audit; not proof "
            "that nobody has ever viewed any frame in the full recording.",
            "Full-session saved C27 and detector outputs already exist for this recording.",
            "One trio; early-recording temporal validation is not independent-group evidence.",
            "No parameter selection or retuning from validation results is allowed.",
        ],
    )
    preserved = {}
    for name in (
        "20260917_clean2d_video_comparison",
        "20260917_clean_tracklets",
        "20260917_custom2d_comparison",
    ):
        for path in (ROOT / "labs" / name).glob("*.py"):
            preserved[path.relative_to(ROOT).as_posix()] = sha(path)
    for name in ("clean2d_video_comparison_v1", "clean_tracklets_v2", "custom2d_comparison_v1"):
        for path in (ROOT / "work" / name).rglob("*.html"):
            preserved[path.relative_to(ROOT).as_posix()] = sha(path)
    for case in json.loads(
        (ROOT / "work/clean2d_video_comparison_v1/protocol.json").read_text(encoding="utf-8")
    )["cases"]:
        for path in (ROOT / "work/clean2d_video_comparison_v1" / case["name"]).glob("*.npz"):
            preserved[path.relative_to(ROOT).as_posix()] = sha(path)
    for relative in (
        ".gitignore",
        "docs/DEVELOPMENT_STRATEGY.md",
        "docs/NEXT_SESSION.md",
        "src/medaka3d/calib.py",
    ):
        preserved[relative] = sha(ROOT / relative)
    with provenance("variable_count_selection", OUT) as record:
        for name, digest in sources.items():
            record.input(name, ROOT / name, sha256=digest)
        record.param("selection_rule", audit["selection_rule"])
        target.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        snapshot = OUT / "preservation_snapshot.json"
        snapshot.write_text(json.dumps(preserved, indent=2), encoding="utf-8")
        record.output("selection", target, sha256=sha(target))
        record.output("preservation", snapshot, sha256=sha(snapshot))
    print(json.dumps(dict(selected_case=chosen, preserved_files=len(preserved))))


if __name__ == "__main__":
    main()
