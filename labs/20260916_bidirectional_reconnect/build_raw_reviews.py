"""Build source-camera evidence for the fixed bidirectional development cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from review_video import render_contact_sheet, render_review_clip

from medaka3d.calib import load_rig
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS

REPO = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_float(value):
    return float(value) if np.isfinite(value) else None


def load_case(root, name, record):
    directory = root / name
    for filename in ("poses.npz", "analysis.json"):
        path = directory / filename
        record.input(f"{name}/{filename}", path, sha256=sha256(path))
    with np.load(directory / "poses.npz") as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    report = json.loads((directory / "analysis.json").read_text(encoding="utf-8"))
    return arrays, report


def selection(arrays, wanted):
    frames = arrays["frames"]
    wanted = np.unique(np.clip(np.asarray(wanted, dtype=np.int64), frames[0], frames[-1]))
    indices = np.searchsorted(frames, wanted)
    if not np.array_equal(frames[indices], wanted):
        raise ValueError("selected review frame absent from native pose source")
    return wanted, arrays["pose"][indices]


def method_labels(arrays, first, stop):
    """Retain alerts for 0.1 s either side so real-time sampling can show them."""
    flags = {}
    for t, frame in enumerate(arrays["diag_frames"]):
        if not first <= frame < stop:
            continue
        for slot in range(3):
            method = None
            if arrays["diag_distance"][t, slot]:
                method = "distance"
            elif arrays["diag_forward"][t, slot]:
                method = "forward"
            elif arrays["diag_bidirectional"][t, slot]:
                method = "bidirectional-only"
            if method:
                for shown_frame in range(max(first, int(frame) - 17), min(stop, int(frame) + 18)):
                    flags.setdefault(shown_frame, set()).add(f"S{slot}:{method}")
    return {frame: " | ".join(sorted(values)) for frame, values in flags.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis", type=Path, default=REPO / "work/bidirectional_reconnect_v1/analysis"
    )
    parser.add_argument(
        "--output", type=Path, default=REPO / "work/bidirectional_reconnect_v1/review"
    )
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output directory must be new or empty; preserve existing evidence")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    cache = REPO / "work/three_body/source_cache"
    cache_path = cache / "source_cache_manifest.json"
    cached = json.loads(cache_path.read_text(encoding="utf-8"))
    if cached["status"] != "verified":
        raise ValueError("source cache does not have verified status")
    raw_paths = {
        view: cache / "medaka_202512" / cached["experiment"] / view / filename
        for view, filename in zip(
            VIEWS, ("cam0_top.mp4", "cam1_side.mp4", "cam2_side.mp4"), strict=True
        )
    }
    calibration = REPO / (
        "work/generative_tracking_inputs/calibration/fixed_refractive_192x192x186/"
        "phase1-calibration-preserve-20260802T180200JST-60e13a7746d1/calibration"
    )
    rig = load_rig(calibration)
    manifest = {
        "schema": "bidirectional_raw_review_v1",
        "purpose": "source-pixel review of extra direction events and historical failure locations",
        "identity_corrections_applied": 0,
        "identity_ground_truth": False,
        "source_cache_rehashed_this_run": False,
        "source_cache_sha_origin": "previously verified source_cache_manifest.json",
        "movie_label_padding_native_frames": 17,
        "movies": [],
        "contact_sheets": [],
        "source": {},
    }
    with provenance("bidirectional_raw_review", args.output) as record:
        record.params_from(args)
        code_paths = [
            Path(__file__),
            Path(__file__).with_name("review_video.py"),
            REPO / "src/medaka3d/aquarium_render.py",
            REPO / "src/medaka3d/aquarium.py",
            REPO / "src/medaka3d/calib.py",
        ]
        for path in code_paths:
            record.input(f"code:{path.name}", path, sha256=sha256(path))
        record.input("source_cache_manifest", cache_path, sha256=sha256(cache_path))
        for path in calibration.glob("*.json"):
            record.input(f"calibration:{path.name}", path, sha256=sha256(path))
        for view, path in raw_paths.items():
            entry = cached["files"][path.relative_to(cache).as_posix()]
            info = path.stat()
            if info.st_size != entry["bytes"]:
                raise ValueError(f"source cache byte count changed: {view}")
            if entry["local_full_sha256"] != entry["source_stream_sha256"]:
                raise ValueError(f"source cache verification inconsistent: {view}")
            record.input(view, path, sha256=entry["local_full_sha256"])
            manifest["source"][view] = {
                "path": str(path),
                "bytes": info.st_size,
                "mtime_ns": info.st_mtime_ns,
                "previously_verified_sha256": entry["local_full_sha256"],
            }

        def register(kind, role, name, metadata, **extra):
            path = Path(metadata["output"])
            digest = sha256(path)
            record.output(name, path, sha256=digest)
            item = {
                "role": role,
                "name": name,
                "path": path.relative_to(args.output).as_posix(),
                "sha256": digest,
                "bytes": path.stat().st_size,
                "metadata": metadata,
                **extra,
            }
            manifest[kind].append(item)
            (args.output / "progress.json").write_text(
                json.dumps(
                    {
                        "elapsed_seconds": time.monotonic() - started,
                        "contact_sheets": len(manifest["contact_sheets"]),
                        "movies": len(manifest["movies"]),
                        "last_output": item["path"],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"{time.monotonic() - started:.1f}s {name}: {path.name}", flush=True)

        core, core_report = load_case(args.analysis, "cg20s", record)
        first, stop = core_report["case"]["start"], core_report["case"]["stop"]
        extra_mask = core["diag_bidirectional"] & ~core["diag_forward"]
        extra_mask &= ((core["diag_frames"] >= first) & (core["diag_frames"] < stop))[:, None]
        expected = int(core_report["bidirectional_extra_slot_steps"])
        if np.count_nonzero(extra_mask) != expected:
            raise ValueError("extra event count disagrees with saved core analysis")
        for edge, slot in np.argwhere(extra_mask):
            frame, slot = int(core["diag_frames"][edge]), int(slot)
            position = core["xyz"][np.searchsorted(core["frames"], frame)]
            distances = np.linalg.norm(position - position[slot], axis=1)
            focus = sorted(set([slot, *np.flatnonzero(distances <= 30).tolist()]))
            frames, pose = selection(core, np.array([-8, -2, -1, 0, 8]) + frame)
            name = f"bidirectional_only_f{frame}_slot{slot}"
            metadata = render_contact_sheet(
                raw_paths, rig, frames, pose, args.output / f"{name}.png", focus_slots=focus
            )
            register(
                "contact_sheets",
                "bidirectional_only",
                name,
                metadata,
                center_frame=frame,
                diagnostic_slot=slot,
                step_mm=finite_float(core["diag_step_mm"][edge, slot]),
                past_angle_deg=finite_float(core["diag_past_angle_deg"][edge, slot]),
                future_angle_deg=finite_float(core["diag_future_angle_deg"][edge, slot]),
                focus_reason="flagged slot plus slots within 30 mm at the event frame",
            )

        protocol_path = args.analysis / "protocol.json"
        record.input("analysis_protocol", protocol_path, sha256=sha256(protocol_path))
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        known = [case for case in protocol["cases"] if case["name"].startswith("known_")]
        if len(known) != 12:
            raise ValueError("expected the fixed twelve historical failure neighborhoods")
        for case in known:
            arrays, report = load_case(args.analysis, case["name"], record)
            center = int(case["name"].split("_")[-1])
            nearby = np.abs(arrays["diag_frames"] - center) <= 2
            jumps = np.where(
                np.isfinite(arrays["diag_step_mm"][nearby]), arrays["diag_step_mm"][nearby], -np.inf
            )
            scores = jumps.max(axis=0)
            focus = sorted(np.argsort(-scores, kind="stable")[:2].tolist())
            frames, pose = selection(arrays, np.array([-8, -2, 0, 2, 8]) + center)
            name = f"{case['name']}_native_contact"
            metadata = render_contact_sheet(
                raw_paths, rig, frames, pose, args.output / f"{name}.png", focus_slots=focus
            )
            register(
                "contact_sheets",
                "historical_failure_neighborhood",
                name,
                metadata,
                center_frame=center,
                historical_note=case.get("note"),
                historical_note_chain=case.get("note_chain"),
                current_c34_detection=report.get("historical_note_proximity"),
                focus_reason="two slots with largest step within two native frames of center",
                focus_step_scores_mm=[finite_float(value) for value in scores],
            )

        labels = method_labels(core, first, stop)
        frames, pose = selection(core, np.arange(first, stop))
        print("Rendering main 20-second raw/CG movie", flush=True)
        metadata = render_review_clip(
            raw_paths,
            rig,
            frames,
            pose,
            args.output / "cg20s_raw_and_measured_cg.mp4",
            labels=labels,
        )
        register("movies", "development20s", "cg20s_raw_and_measured_cg", metadata)
        for center in (76235, 76930):
            frames, pose = selection(core, np.arange(center - 166, center + 166))
            name = f"native_slow_f{center}"
            print(f"Rendering two native seconds around {center}", flush=True)
            metadata = render_review_clip(
                raw_paths,
                rig,
                frames,
                pose,
                args.output / f"{name}.mp4",
                labels=labels,
                preserve_native=True,
            )
            register("movies", "native_slow_event", name, metadata, center_frame=center)
        manifest["elapsed_seconds"] = time.monotonic() - started
        manifest["expected_bidirectional_only_contact_sheets"] = expected
        manifest["expected_historical_contact_sheets"] = len(known)
        manifest_path = args.output / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
        record.output("manifest", manifest_path, sha256=sha256(manifest_path))
        record.param("raw_video_sha256_recomputed", False)
        record.param("raw_video_sha256_source", str(cache_path))
        record.param("elapsed_seconds", manifest["elapsed_seconds"])
        print(f"Completed: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
