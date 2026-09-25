"""Export source-pixel proposal comparisons and scientific figures for human review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from run import CALIBRATION, write_json
from visuals import plot_case, render_comparison_sheet, render_comparison_video

from medaka3d.aquarium import sha256
from medaka3d.calib import load_rig
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS


def focus_at(arrays, frame):
    t = int(np.searchsorted(arrays["frames"], frame))
    changed = np.any(arrays["candidate_source"][t] != arrays["original_source"][t], axis=(1, 2))
    slots = np.flatnonzero(changed).tolist()
    return slots if len(slots) >= 2 else [0, 1, 2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("work/multiview_node_review_v1"))
    parser.add_argument("--source-cache", type=Path, default=Path("work/three_body/source_cache"))
    parser.add_argument("--input-root", type=Path, default=Path("work/generative_tracking_inputs"))
    args = parser.parse_args()
    output = args.root / "review"
    output.mkdir(parents=True, exist_ok=False)
    analysis = args.root / "analysis"
    report = json.loads((analysis / "summary.json").read_text(encoding="utf-8"))
    cache_manifest = args.source_cache / "source_cache_manifest.json"
    cached = json.loads(cache_manifest.read_text(encoding="utf-8"))
    if cached["status"] != "verified":
        raise ValueError("Source video cache unverified")
    raw_paths = {
        v: args.source_cache / "medaka_202512" / cached["experiment"] / v / filename
        for v, filename in zip(
            VIEWS, ("cam0_top.mp4", "cam1_side.mp4", "cam2_side.mp4"), strict=True
        )
    }
    rig = load_rig(args.input_root / CALIBRATION)
    all_entries = []
    arrays_by_case = {}
    reports_by_case = {}
    with provenance("multiview_node_visuals", output) as rec:
        rec.params_from(args)
        for path in (
            Path(__file__),
            Path(__file__).with_name("visuals.py"),
            analysis / "summary.json",
            cache_manifest,
        ):
            rec.input(path.name, path, sha256=sha256(path))
        for view, path in raw_paths.items():
            known = cached["files"][path.relative_to(args.source_cache).as_posix()]
            if (
                path.stat().st_size != known["bytes"]
                or known["local_full_sha256"] != known["source_stream_sha256"]
            ):
                raise ValueError("Source cache mismatch")
            rec.input(view, path, sha256=known["local_full_sha256"])
        rec.param("raw_sha_origin", "previously verified source cache; bytes checked this run")

        def register(kind, name, case, center, metadata, **extra):
            path = Path(metadata["output"])
            entry = dict(
                kind=kind,
                name=name,
                case=case,
                center_frame=int(center),
                path=path.relative_to(output.resolve()).as_posix(),
                metadata=metadata,
                **extra,
            )
            all_entries.append(entry)
            rec.output(name, path, sha256=sha256(path))
            write_json(output / "progress.json", dict(completed=len(all_entries), last=name))
            print(f"{kind}: {name}", flush=True)

        for item in report["cases"]:
            name = item["case"]["name"]
            directory = analysis / name
            with np.load(directory / "evaluation.npz", allow_pickle=False) as a:
                arrays = {k: a[k] for k in a.files}
            detail = json.loads((directory / "evaluation.json").read_text(encoding="utf-8"))
            arrays_by_case[name] = arrays
            reports_by_case[name] = detail
            for filename in ("evaluation.npz", "evaluation.json"):
                path = directory / filename
                rec.input(f"{name}/{filename}", path, sha256=sha256(path))
            center = int(item["case"].get("note", {}).get("frame_start", item["case"]["start"]))
            meta = plot_case(
                arrays,
                center,
                output / f"{name}_diagnostics.png",
                core=(item["case"]["start"], item["case"]["stop"]),
            )
            register("plot", f"{name}_diagnostics", name, center, meta, scope="full_case")
            centers = {}
            if name == "cg20s":
                for g in detail["parent_episodes"]:
                    relevant = [
                        d
                        for d in detail["frames"]
                        if g["start_frame"] <= d["frame"] < g["stop_frame_exclusive"]
                    ]
                    candidates = [d for d in relevant if d["gain"] is not None]
                    chosen = (
                        max(candidates, key=lambda d: d["gain"])["frame"]
                        if candidates
                        else g["first_edge_frame"]
                    )
                    centers.setdefault(chosen, []).append(f"episode_{g['index']}")
                    for edge in g["edges"]:
                        if edge["bidirectional_only"]:
                            centers.setdefault(edge["frame"], []).append(
                                "additional_backward_event"
                            )
            elif name.startswith("known_"):
                centers[center] = ["historical_center"]
            for frame, roles in sorted(centers.items()):
                selected = np.unique(
                    np.clip(
                        np.array([-8, -1, 0, 1, 8]) + frame,
                        arrays["frames"][0],
                        arrays["frames"][-1],
                    )
                )
                title = f"{name}_f{frame}"
                slots = focus_at(arrays, frame)
                meta = render_comparison_sheet(
                    raw_paths,
                    rig,
                    arrays,
                    selected,
                    output / f"{title}_comparison.png",
                    focus_slots=slots,
                )
                register(
                    "image",
                    f"{title}_comparison",
                    name,
                    frame,
                    meta,
                    roles=roles,
                    focus_slots=slots,
                )

        video_centers = [
            ("known_30735", 30735),
            ("known_31479", 31479),
            ("cg20s", 76235),
            ("cg20s", 76930),
            ("cg20s", 76988),
            ("cg20s", 77415),
            ("known_323819", 323819),
        ]
        for name, center in video_centers:
            arrays = arrays_by_case[name]
            first, stop = center - 83, center + 83
            slots = focus_at(arrays, center)
            title = f"{name}_f{center}"
            meta = render_comparison_video(
                raw_paths,
                rig,
                arrays,
                first,
                stop,
                output / f"{title}_native_slow.mp4",
                focus_slots=slots,
            )
            register("video", f"{title}_native_slow", name, center, meta, focus_slots=slots)
            meta = plot_case(
                arrays,
                center,
                output / f"{title}_local_diagnostics.png",
                core=(first, stop),
                focus_slots=slots,
            )
            register(
                "plot",
                f"{title}_local_diagnostics",
                name,
                center,
                meta,
                scope="local_video",
                focus_slots=slots,
            )
        manifest = dict(
            accepted_identity_corrections=0,
            entries=all_entries,
            counts={
                kind: sum(e["kind"] == kind for e in all_entries)
                for kind in ("image", "plot", "video")
            },
            selection=(
                "Every primary episode: frame with greatest saved objective gain "
                "(not human labels); "
                "all14 backward-only events and12 historical centers;7 fixed mechanism videos."
            ),
        )
        write_json(output / "manifest.json", manifest)
        rec.output("manifest", output / "manifest.json", sha256=sha256(output / "manifest.json"))


if __name__ == "__main__":
    main()
