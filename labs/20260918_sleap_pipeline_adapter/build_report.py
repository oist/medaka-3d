"""Render parameter comparisons and native-frame source-video evidence."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import numpy as np

from medaka3d.provenance import provenance

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "work/sleap_pipeline_adapter_v1"
OLD = ROOT / "work/variable_count_tracklets_v1"
OLD_METHODS = ("C27", "S1", "S2", "P0", "P1")
NEW_METHODS = (
    "SP",
    "G0",
    "G_history3",
    "G_history9",
    "G_match025",
    "G_match10",
    "G_margin02",
    "G_margin10",
    "G_age8",
    "G_age33",
)
VIEWS = ("top_cam", "side1_cam", "side2_cam")
COLORS = {"S2": "#77baff", "SP": "#66e0bd", "G0": "#ffc86f", "P1": "#df8cce"}


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(clean(value), ensure_ascii=False, allow_nan=False, separators=(",", ":")),
        encoding="utf-8",
    )


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(case, view, method):
    path = (OLD if method in OLD_METHODS else OUT) / case / f"tracking_{method}_{view}.npz"
    with np.load(path, allow_pickle=False) as source:
        result = {key: source[key] for key in source.files}
    return result, path


def plot(path, case, view, arms):
    start, stop = case["start"], case["stop"]
    boundaries = np.unique(np.r_[np.arange(start, stop, 66), stop])
    content = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="290" viewBox="0 0 720 290">',
        '<rect width="720" height="290" fill="#101723"/>',
        '<text x="55" y="24" fill="#fff" font-family="sans-serif" font-size="17">'
        f'{view} · {case["name"]}</text>',
    ]
    for y in (0, 0.25, 0.5, 0.75, 1):
        yy = 230 - y * 175
        content.append(f'<path d="M55 {yy}H690" stroke="#344357"/>')
        content.append(
            f'<text x="7" y="{yy + 5}" fill="#b5c3d4" font-size="13">{y * 100:.0f}%</text>'
        )
    for i, (method, color) in enumerate(COLORS.items()):
        arm = arms[method]
        frames, labels = arm["frame_idx"], arm["labels"]
        last, linked = {}, np.zeros(len(frames), dtype=bool)
        for row, (frame, label) in enumerate(zip(frames, labels, strict=True)):
            linked[row] = label >= 0 and last.get(int(label), -99) == frame - 1
            if label >= 0:
                last[int(label)] = int(frame)
        points = []
        for left, right in pairwise(boundaries):
            selected = (frames >= left) & (frames < right) & (frames > start)
            if selected.any():
                rate = float((~linked[selected]).mean())
                xx = 55 + ((left + right) / 2 - start) / (stop - start) * 635
                points.append(f"{xx:.2f},{230 - rate * 175:.2f}")
        joined = " ".join(points)
        content.append(
            f'<polyline points="{joined}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
        content.append(
            f'<text x="{55 + i * 130}" y="46" fill="{color}" font-size="14">{method}</text>'
        )
    content.extend(
        [
            f'<text x="55" y="253" fill="#b5c3d4" font-size="13">native f{start}</text>',
            f'<text x="570" y="253" fill="#b5c3d4" font-size="13">f{stop - 1}</text>',
            '<text x="55" y="277" fill="#b5c3d4" font-size="12">'
            'Unconnected / eligible observations per 66 native frames</text>',
            "</svg>",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content), encoding="utf-8")


def main():
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    metrics = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))
    if not metrics["complete"]:
        raise ValueError("Report requires every frozen condition")
    cases, methods = protocol["cases"], [*OLD_METHODS, *NEW_METHODS]
    from adapter import AdapterConfig
    from run import configurations

    parameters = []
    for name, config in configurations().items():
        if name == "SP":
            continue
        if isinstance(config, dict):
            config = AdapterConfig(**config)
        parameters.append(
            {
                "method": name,
                "history": config.history,
                "match_threshold": config.match_threshold,
                "margin": config.margin,
                "age_frames": config.max_age_frames,
            }
        )
    manifest = {
        "inputs": {},
        "outputs": {},
        "views": 36,
        "media_policy": "Native source video and retained detections; no generated observations",
    }
    with provenance("sleap_adapter_report", OUT) as record:
        for name in ("protocol.json", "metrics.json", "video_source_audit.json"):
            record.input(name, OUT / name, sha256=sha(OUT / name))
        video_audit = json.loads((OUT / "video_source_audit.json").read_text(encoding="utf-8"))
        for source in video_audit["videos"]:
            manifest["inputs"][Path(source["path"]).as_posix()] = source["sha256"]
        for case in cases:
            for view in VIEWS:
                arms = {}
                for method in methods:
                    arms[method], path = load(case["name"], view, method)
                    manifest["inputs"][str(path.relative_to(ROOT))] = sha(path)
                reference = arms["G0"]
                for arm in arms.values():
                    for key in ("frame_idx", "points", "global_row"):
                        if not np.array_equal(arm[key], reference[key], equal_nan=True):
                            raise AssertionError(
                                f"Viewer input differs: {case['name']} {view} {key}"
                            )
                core = (reference["frame_idx"] >= case["start"]) & (
                    reference["frame_idx"] < case["stop"]
                )
                indices = np.flatnonzero(core)
                frame_rows = defaultdict(list)
                for i, frame in enumerate(reference["frame_idx"][core]):
                    frame_rows[int(frame)].append(i)
                evidence_path = OUT / case["name"] / f"evidence_G0_{view}.json.gz"
                manifest["inputs"][str(evidence_path.relative_to(ROOT))] = sha(evidence_path)
                with gzip.open(evidence_path, "rt", encoding="utf-8") as stream:
                    evidence = json.load(stream)
                events = evidence["events"]
                by_row = {event["global_row"]: event["reason"] for event in events}
                cam = {"top_cam": "cam0_top", "side1_cam": "cam1_side", "side2_cam": "cam2_side"}[
                    view
                ]
                viewer = {
                    "view": view,
                    "case": case["name"],
                    "start": case["start"],
                    "stop": case["stop"],
                    "anchor": 75498
                    if case["name"] == "cg20s"
                    else (case.get("centers") or [case["start"]])[0],
                    "points": reference["points"][core],
                    "globalRow": reference["global_row"][core],
                    "frameRows": frame_rows,
                    "labels": {m: a["labels"][core] for m, a in arms.items()},
                    "provisional": reference["provisional_labels"][core],
                    "reasons": [by_row.get(int(reference["global_row"][i]), "") for i in indices],
                    "video": "../three_body/source_cache/medaka_202512/"
                    f"20251225_K_1_2_FCMC/{view}/{cam}.mp4",
                }
                destination = OUT / "viewer" / f"{case['name']}_{view}.json"
                write_json(destination, viewer)
                plot_path = OUT / "plots" / f"{case['name']}_{view}.svg"
                plot(plot_path, case, view, arms)
                for path in (destination, plot_path):
                    manifest["outputs"][str(path.relative_to(OUT))] = sha(path)
        summary = {
            "cases": cases,
            "methods": methods,
            "parameters": parameters,
            "records": metrics["records"],
            "verdict": (
                "All settings were fixed and compared. The separation of provisional histories "
                "from committed fragments, input retention and recoverability were verified; the "
                "individual-correspondence ground truth needed for an adoption decision remains "
                "outstanding. See the results and verification records for details."
            ),
        }
        write_json(OUT / "report_data.json", summary)
        (OUT / "report.html").write_bytes((HERE / "report_template.html").read_bytes())
        for name in ("report_data.json", "report.html"):
            manifest["outputs"][name] = sha(OUT / name)
            record.output(name, OUT / name, sha256=sha(OUT / name))
        record.input("builder", Path(__file__), sha256=sha(Path(__file__)))
        record.input(
            "template", HERE / "report_template.html", sha256=sha(HERE / "report_template.html")
        )
        write_json(OUT / "report_manifest.json", manifest)
        record.output(
            "manifest", OUT / "report_manifest.json", sha256=sha(OUT / "report_manifest.json")
        )
    print("Built HTML, 36 native source viewers, and 36 timeline graphs")


if __name__ == "__main__":
    main()
