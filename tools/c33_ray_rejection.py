"""C33 - drop the one offending ray, only where the doubt channel points.

=============================================================================
A BOUNDED REVIVAL OF A DEFERRED IDEA
=============================================================================

The 2026-08-24 ray-level plan proposed dropping the offending ray instead of
refusing the whole animal-frame. The 2026-08-26 followup deferred it: after
the uncapping there were 871 contaminated three-camera animal-frames left,
0.084% of the recording, in a reconstruction that commits no swaps - nothing
to prove an improvement against. What changed is the target: the
experimenter's 2026-09-01 review diagnosed the surviving single-frame spikes
(30299, 30307, 30841: a false positive or wall reflection roping in as a
third ray while the real animal is occluded), and `c31` measured that
``residual > 3 mm on a 3-ray slot`` fires on **0.35%** of frames and captures
**93.7%** of them. That is the whole domain of this tool.

The prior evidence for the mechanism: of those 871 frames, **93.1%** were
fixed by dropping exactly one ray and the culprit was sharply identified on
72.1% (`NEXT_SESSION_uncapped_followup.md`).

=============================================================================
THE RULE - fixed in decisions/2026-09-01-doubt-channel.md before the run
=============================================================================

For every (frame, slot) with **3 rays and residual > 3.0 mm**: triangulate
leave-one-out for each view, the same view dropped across the cost nodes
(pectoral, mouth - the tail never takes part in anything that decides).
Adopt the best drop **only if** its residual is <= **1.5 mm** and the
runner-up drop is at least **1.0 mm worse** - a culprit that is not sharp is
not a culprit, and an ambiguous frame keeps its honest fat residual rather
than a guessed thin one.

Adoption changes geometry, never labels: ``xyz``, ``residual_mm``,
``slot_rays`` for that (frame, slot), and the dropped view's ``source`` entry
becomes UNRESOLVED so every downstream reader (the render, `c09`'s anchor
matching) sees the ray as unused. The correspondence's choice of which
detection is which animal is untouched. The gated tracker and per-view tracks
are re-run on the repaired geometry, because geometry moves its breaks.

Run::

    set PYTHONIOENCODING=utf-8
    .venv\\Scripts\\python.exe tools/c33_ray_rejection.py --arm-dir <dir> --arm-prefix C32-gated
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import c01_s16_predictions as c01
from c02_tracking_probe import CALIBRATION
from c19_viterbi_correspondence import coverage, derive_gate
from c31_doubt_channel import RESIDUAL_SPIKE_MM

from medaka3d import load_paths
from medaka3d.calib import load_rig, triangulate
from medaka3d.provenance import provenance
from medaka3d.reconstruct import Reconstruction
from medaka3d.track import METHODS, propagate_to_views
from medaka3d.types import VIEWS

RUN = "bottomup"
GATE_MM = 4.0
GAP_FRAMES = 73

#: Cost nodes, as everywhere since `c13`: the tail decides nothing.
NODES = ("pectoral", "mouth")
ANCHOR = "pectoral"

#: Adoption thresholds, pre-registered. A repaired point must look like a
#: correct triangulation (calibration rms 0.24 mm, tail p99 ~1.3 mm), and the
#: culprit must be sharp.
ADOPT_MM = 1.5
SHARP_MM = 1.0

UNRESOLVED = -1


def repair(arm: dict, detections: dict, rig) -> tuple[dict, dict]:
    """Leave-one-out repair of flagged (frame, slot) pairs. Pure on its inputs."""
    xyz = arm["xyz"].astype(np.float64).copy()
    residual = arm["residual_mm"].astype(np.float64).copy()
    slot_rays = arm["slot_rays"].copy()
    source = arm["source"].copy()

    flagged = np.argwhere((slot_rays == 3) & (residual > RESIDUAL_SPIKE_MM))
    node_index = {view: [detections[view].nodes.index(n) for n in NODES]
                  for view in VIEWS}
    starts = {view: detections[view].start for view in VIEWS}

    stats = {"flagged": int(flagged.shape[0]), "adopted": 0,
             "not_sharp": 0, "not_clean": 0, "no_geometry": 0,
             "dropped_view": dict.fromkeys(VIEWS, 0)}
    if not flagged.size:
        return {"xyz": xyz, "residual_mm": residual, "slot_rays": slot_rays,
                "source": source}, stats

    # Rays for every flagged (frame, slot): (n, views, nodes, 3).
    n = flagged.shape[0]
    origins = np.full((n, len(VIEWS), len(NODES), 3), np.nan)
    directions = np.full_like(origins, np.nan)
    for column, view in enumerate(VIEWS):
        rows = source[flagged[:, 0], flagged[:, 1], column]
        held = rows >= 0
        flat = starts[view][flagged[:, 0]] + np.where(held, rows, 0)
        uv = detections[view].points[flat][:, node_index[view], :]
        uv[~held] = np.nan
        ray_o, ray_d = rig.camera(view).rays(uv.reshape(-1, 2))
        origins[:, column] = ray_o.reshape(n, len(NODES), 3)
        directions[:, column] = ray_d.reshape(n, len(NODES), 3)
    present = np.isfinite(origins).all(axis=-1) & np.isfinite(directions).all(axis=-1)

    anchor = NODES.index(ANCHOR)
    for i, (frame, slot) in enumerate(flagged):
        candidates = []
        for drop in range(len(VIEWS)):
            keep = present[i].copy()
            keep[drop, :] = False
            if keep[:, anchor].sum() < 2:
                continue
            point, rms = triangulate(
                origins[i].transpose(1, 0, 2), directions[i].transpose(1, 0, 2),
                valid=keep.T)                      # per node: (nodes,), rays=views
            live = keep.T.sum(axis=1) >= 2
            live &= np.isfinite(rms)
            if not live[anchor]:
                continue
            candidates.append((float(rms[live].mean()), drop, point[anchor]))
        if len(candidates) < 2:
            stats["no_geometry"] += 1
            continue
        candidates.sort()
        best, runner = candidates[0], candidates[1]
        if best[0] > ADOPT_MM:
            stats["not_clean"] += 1
            continue
        if runner[0] - best[0] < SHARP_MM:
            stats["not_sharp"] += 1
            continue
        xyz[frame, slot] = best[2]
        residual[frame, slot] = best[0]
        slot_rays[frame, slot] = 2
        source[frame, slot, best[1]] = UNRESOLVED
        stats["adopted"] += 1
        stats["dropped_view"][VIEWS[best[1]]] += 1

    return {"xyz": xyz, "residual_mm": residual, "slot_rays": slot_rays,
            "source": source}, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("=====")[0].strip())
    parser.add_argument("--experiment", default="20251225_K_1_2_FCMC")
    parser.add_argument("--arm-dir", type=Path, required=True,
                        help="directory holding the arms to repair")
    parser.add_argument("--arm-prefix", required=True,
                        help="e.g. C32-gated or C28-gated - named explicitly so "
                             "the output says what it repaired")
    parser.add_argument("--lambdas", nargs="+", type=float,
                        default=[0.0, 0.1, 0.25, 0.5, 0.767, 2.0, 8.0])
    parser.add_argument("--tag", default=None,
                        help="output leaf suffix when the prefix alone is "
                             "ambiguous - two chains can both hand over "
                             "C32-gated arms")
    args = parser.parse_args()

    paths = load_paths()
    experiment = args.experiment
    out_dir = (paths.work_dir / "c33_ray_rejection"
               / f"{experiment}__{args.tag or args.arm_prefix}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rig = load_rig(paths.asset_root / CALIBRATION)

    detections, starts, counts = {}, {}, {}
    for view in VIEWS:
        detections[view] = c01.load(paths.work_dir, experiment, view, RUN)
        starts[view] = detections[view].start
        counts[view] = int(detections[view].frame_idx.size)

    with provenance("ray_rejection", out_dir, name="c33") as record:
        record.params_from(args)
        record.param("n_groups", 1)
        record.param("n_experiments", 1)
        record.param("flag_residual_mm", RESIDUAL_SPIKE_MM)
        record.param("adopt_mm", ADOPT_MM)
        record.param("sharp_mm", SHARP_MM)
        record.param("gate_mm", GATE_MM)
        record.param("gate_is_post_hoc", True)
        record.param("lambda_is_chosen", False)

        arms = {}
        for lam in args.lambdas:
            arm_path = (args.arm_dir
                        / f"{args.arm_prefix}__{experiment}__lam{lam:g}.npz")
            if not arm_path.is_file():
                raise SystemExit(f"no arm at {arm_path}")
            arm = np.load(arm_path)
            record.input(f"arm_lam{lam:g}", arm_path)
            started = time.time()

            fixed, stats = repair(arm, detections, rig)
            out = Reconstruction(
                experiment=experiment, views=tuple(VIEWS),
                xyz=fixed["xyz"].astype(np.float32),
                source=fixed["source"],
                residual_mm=fixed["residual_mm"].astype(np.float32),
                margin_mm=arm["margin_mm"],
                n_views_used=fixed["slot_rays"].min(axis=1).astype(np.int8))
            derived = derive_gate(out)
            config = dataclasses.replace(derived, max_step_mm=GATE_MM,
                                         max_gap_frames=GAP_FRAMES)
            slot_track = METHODS["gated"](out, config)
            per_view = propagate_to_views(out, slot_track, starts, counts)

            path = out_dir / f"C33-gated__{experiment}__lam{lam:g}.npz"
            payload = {k: arm[k] for k in arm.files
                       if k not in ("xyz", "residual_mm", "slot_rays", "source",
                                    "slot_track", "n_views_used")
                       and not k.startswith("track__")}
            np.savez_compressed(
                path, xyz=out.xyz, residual_mm=out.residual_mm,
                slot_rays=fixed["slot_rays"], source=out.source,
                n_views_used=out.n_views_used, slot_track=slot_track,
                **payload, **{f"track__{v}": per_view[v] for v in VIEWS})
            record.output(f"C33-gated__lam{lam:g}", path)

            arms[f"{lam:g}"] = {
                "lambda_mm": float(lam), "path": str(path), "input": str(arm_path),
                **stats,
                "coverage_gated": coverage(slot_track, out.xyz.shape[0]),
                "seconds": round(time.time() - started, 1)}
            print(f"  lambda {lam:<6g} flagged {stats['flagged']:5d}  adopted "
                  f"{stats['adopted']:5d}  (not sharp {stats['not_sharp']}, "
                  f"not clean {stats['not_clean']}, no geometry "
                  f"{stats['no_geometry']})   dropped "
                  + " ".join(f"{v[:5]}:{stats['dropped_view'][v]}" for v in VIEWS)
                  + f"   {arms[f'{lam:g}']['seconds']:.0f}s", flush=True)

        record.param("arms", arms)
        (out_dir / "c33_summary.json").write_text(
            json.dumps({"experiment": experiment, "n_groups": 1,
                        "n_experiments": 1, "input_prefix": args.arm_prefix,
                        "input_dir": str(args.arm_dir),
                        "flag_residual_mm": RESIDUAL_SPIKE_MM,
                        "adopt_mm": ADOPT_MM, "sharp_mm": SHARP_MM,
                        "arms": arms}, indent=2, default=float),
            encoding="utf-8")
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
