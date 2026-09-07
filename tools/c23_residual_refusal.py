"""C23 — the triangulation residual as a refusal, absolute and in time.

=============================================================================
THE RULE, FIXED BEFORE THE ARMS WERE RUN
=============================================================================

Minimising over 36 correspondence states returns a winner even when every state
is wrong. Nothing downstream checks whether the winner is any good, and the
evidence that it is not has been computed and stored all along: `residual_mm`,
the per-animal point-to-ray RMS. On this rig a correct pairing is
**sub-millimetre** — median 0.466 mm over 992,116 three-camera animal-frames.

K_1_2 frame 113,382, reviewed by eye by the experimenter, is what this is for.
`side1_cam` reports **four** detections; `c16` hands out exactly three labels
(``max_tracks=3``, no reject option), and for **ten** frames — 113,374 to
113,383, corrected by `c24` on 2026-08-25; this tool reports 113,378 onward only
because that is where its own flag fires — label 0 sits on a **reflection in the
tank wall** while the real third fish goes unlabelled. The residual for that
animal reads 33.75, 33.90, 33.99, 34.19, 34.35 mm and then 0.49 mm the moment the
label lands on the fish. The correspondence state never
changes across it, so the sequence model cannot fix it: the state is defined on
the labels, and it is the labels that are wrong.

**Two flags, because neither covers the other.**

``absolute``  an animal-frame whose residual exceeds ``--abs-mm``. Catches
              *sustained* wrong pairings — the reflection above holds a label
              for five frames, so its own local median is 34 mm too and no
              relative test can see it.
``relative``  an animal-frame whose residual exceeds ``--rel-x`` times the
              median residual of **its own track** over +/- 25 frames. Catches
              *sudden* ones — a slot that normally sits at 0.4 mm and spikes to
              3 mm is plainly wrong and never reaches any absolute threshold.

Both thresholds are read off the data, not chosen to fix a scene:

* **``--abs-mm`` 4.0** is the trough of the residual histogram. Counts per
  0.5 mm fall 22,835 → 2,711 → 1,439 → 821 → 443 → **286** across 1.0-4.0 mm and
  then flatten at 100-300 per bin all the way to 100 mm. Below the trough is one
  population and above it is another.
* **``--rel-x`` 5.19** is the 99.9th percentile of the ratio, the same
  percentile `c03.derive_gate` uses for `max_step_mm`.

**What a flag does.** The animal's 3D position is withheld from the tracker for
that frame — set to NaN, which stage C's linker already reads as "this animal
has no position here" and bridges with the existing gap tolerance. It is *not*
deleted from the reconstruction and it does not change the correspondence. The
claim under test is narrow: **a confident wrong point is worse than no point.**

=============================================================================
FOUR ARMS
=============================================================================

``none``      the incumbent, `c21`'s `gated-partial-unbounded`, unchanged.
``absolute``  absolute flag only.
``relative``  relative flag only.
``both``      the union.

One gate for all four — `c19`'s 4.0 mm / 73 frames — for the reason `c20` gave:
a derived gate is not comparable between runs. `gated` and `partial`, since
`c21` disqualified `strict` and found `bounded` indistinguishable.

**What would count as success**, stated here so it cannot be chosen afterwards:
swaps not worse on the `c09` unit, and identities that hold at least a second
not materially more numerous. Withholding 0.8% of animal-frames must not cost
more continuity than it buys.

**n_groups = 1, n_experiments = 1, female only** for `c09`; the reflection scene
is one frame range judged by eye. Neither can rank anything finely.

**SUPERSEDED IN PART, 2026-08-25.** `c24` removed `c16`'s track cap, which is the
cause this tool only detects. On `c24`'s labels the reflection scene reads
0.77-0.89 mm throughout and the three-camera animal-frames above 4 mm fall from
7,513 to 871. The relative criterion is still the better of the two flags; the
population it acts on is now an eighth of what is measured above. See
`docs/decisions/2026-08-25-uncapped-tracklets.md`.

Run::

    set PYTHONIOENCODING=utf-8
    .venv\\Scripts\\python.exe tools/c23_residual_refusal.py
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import c01_s16_predictions as c01
from c02_tracking_probe import CALIBRATION
from c13_pectoral_anchor import report
from c19_viterbi_correspondence import coverage
from c20_uncapped_correspondence import cross_view_agreement
from c21_link_refusal import HELD_GAP_FRAMES, HELD_GATE_MM, VELOCITY_FRAMES

from medaka3d import load_paths
from medaka3d.calib import load_rig
from medaka3d.cohort import ExperimentId
from medaka3d.provenance import provenance
from medaka3d.reconstruct import Reconstruction
from medaka3d.track import METHODS, TrackConfig, propagate_to_views
from medaka3d.types import FPS, VIEWS

EXPERIMENT = "20251225_K_1_2_FCMC"
RUN = "bottomup"
STAGE_C = (Path("c20_uncapped_correspondence") / "mouth-pectoral_anchor-pectoral_gate-4"
           / f"C-gated__{EXPERIMENT}.npz")
INCUMBENT = Path("c21_link_refusal") / f"gated-partial-unbounded__{EXPERIMENT}.npz"

#: Half-window for the local median, in frames. 25 either side is 0.3 s, long
#: enough that one bad frame cannot move the median and short enough that a
#: genuine change of posture is not averaged away.
LOCAL_HALF = 25

#: The reflection scene, checked by name so a change that stops catching it is
#: visible rather than silent.
CASE = (113_377, 113_383)


def local_median(residual: np.ndarray, slot_track: np.ndarray, half: int) -> np.ndarray:
    """(n_frames, n_animals) median residual of **each animal's own track** nearby.

    Grouped by track ID rather than by slot, because a slot is arbitrary within a
    frame and comparing an animal against whatever else held its slot index would
    be comparing it to a different fish.
    """
    out = np.full(residual.shape, np.nan)
    order = np.argsort(slot_track, axis=None, kind="stable")
    track = slot_track.ravel()[order]
    value = residual.ravel()[order]
    edges = np.r_[0, np.flatnonzero(np.diff(track)) + 1, track.size]
    for a, b in pairwise(edges):
        if track[a] < 0 or b - a < 3:
            continue
        series = value[a:b]
        if np.isfinite(series).sum() < 3:
            continue
        out.ravel()[order[a:b]] = [
            np.nanmedian(series[max(i - half, 0): i + half + 1]) for i in range(series.size)
        ]
    return out


def flags(run: dict, abs_mm: float, rel_x: float, half: int) -> dict[str, np.ndarray]:
    """The two refusals, and their union, as (n_frames, n_animals) masks."""
    residual = run["residual_mm"].astype(float)
    here = np.isfinite(run["xyz"]).all(axis=2) & (run["slot_rays"] >= 2)
    absolute = here & np.isfinite(residual) & (residual > abs_mm)
    median = local_median(residual, run["slot_track"], half)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = residual / np.maximum(median, 0.05)
    relative = here & np.isfinite(ratio) & (ratio > rel_x)
    return {"none": np.zeros(residual.shape, bool), "absolute": absolute,
            "relative": relative, "both": absolute | relative}


def withhold(run: dict, mask: np.ndarray, views: tuple[str, ...]) -> Reconstruction:
    """The same reconstruction with the flagged animals' positions removed."""
    xyz = run["xyz"].astype(float).copy()
    xyz[mask] = np.nan
    here = np.isfinite(xyz).all(axis=2)
    used = np.where(here, run["slot_rays"], 0).max(axis=1).astype(np.int8)
    return Reconstruction(
        experiment=EXPERIMENT, views=views, xyz=xyz, source=run["source"],
        residual_mm=run["residual_mm"], margin_mm=run["margin_mm"],
        n_views_used=np.where(here.any(axis=1), used, 0).astype(np.int8))


def identities(slot_track: np.ndarray, min_frames: int) -> dict:
    live, counts = np.unique(slot_track[slot_track >= 0], return_counts=True)
    long = counts >= min_frames
    return {"tracks": int(live.size), "held_1s": int(long.sum()),
            "frames_in_1s": float(counts[long].sum() / slot_track.size),
            "median_lifetime_s": float(np.median(counts[long]) / FPS) if long.any() else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("=====")[0].strip())
    parser.add_argument("--abs-mm", type=float, default=4.0,
                        help="trough of the residual histogram; not tuned")
    parser.add_argument("--rel-x", type=float, default=5.19,
                        help="p99.9 of residual / local median; not tuned")
    parser.add_argument("--local-half", type=int, default=LOCAL_HALF)
    parser.add_argument("--gate-mm", type=float, default=HELD_GATE_MM)
    parser.add_argument("--max-gap-frames", type=int, default=HELD_GAP_FRAMES)
    args = parser.parse_args()

    paths = load_paths()
    rig = load_rig(CALIBRATION)
    source_npz = paths.work_dir / STAGE_C
    if not source_npz.is_file():
        raise SystemExit(f"run c20 first; no stage-C output at {source_npz}")
    out_dir = paths.work_dir / "c23_residual_refusal"
    out_dir.mkdir(parents=True, exist_ok=True)

    with provenance("residual_refusal", out_dir, name="c23") as record:
        record.params_from(args)
        record.input("stage_c", source_npz)
        record.param("n_groups", 1)
        record.param("n_experiments", 1)
        print(f"stage C {source_npz}")
        print(f"absolute > {args.abs_mm} mm   relative > {args.rel_x}x local median "
              f"(+/-{args.local_half} frames)")
        print("n_groups = 1, n_experiments = 1 — K_1_2, female only\n")

        run = dict(np.load(source_npz))
        starts, counts = {}, {}
        for view in VIEWS:
            detections = c01.load(paths.work_dir, EXPERIMENT, view, RUN)
            starts[view] = detections.start
            counts[view] = int(detections.frame_idx.shape[0])

        masks = flags(run, args.abs_mm, args.rel_x, args.local_half)
        here = np.isfinite(run["xyz"]).all(axis=2) & (run["slot_rays"] >= 2)
        print("animal-frames withheld, of the "
              f"{int(here.sum()):,} that have a position")
        for name, mask in masks.items():
            print(f"  {name:<9} {int(mask.sum()):>7,}  ({mask.sum()/here.sum():7.4%})")
        record.param("withheld", {k: int(v.sum()) for k, v in masks.items()})

        lo, hi = CASE
        print(f"\nthe reflection scene {lo:,}-{hi:,}, per animal-frame flagged:")
        for name in ("absolute", "relative", "both"):
            caught = [int(f) for f in range(lo, hi + 1) if masks[name][f].any()]
            print(f"  {name:<9} {caught}")
        record.param("case_frames_caught",
                     {k: [int(f) for f in range(lo, hi + 1) if masks[k][f].any()]
                      for k in ("absolute", "relative", "both")})

        config = TrackConfig(max_step_mm=args.gate_mm, max_gap_frames=args.max_gap_frames,
                             velocity_frames=VELOCITY_FRAMES)
        agreement = cross_view_agreement(rig, run)
        incumbent = np.load(paths.work_dir / INCUMBENT, allow_pickle=False)["slot_track"]

        summary = {}
        for name, mask in masks.items():
            out = withhold(run, mask, rig.views)
            slot_track = METHODS["gated"](out, config)
            if name == "none":
                same = bool((slot_track == incumbent).all())
                print(f"\n  the `none` arm reproduces c21's tracks exactly: {same}")
                record.param("reproduces_c21", same)
                if not same:
                    raise SystemExit(
                        "`none` must be c21's own arm; it is not, so nothing below "
                        "is comparable to it.")
            per_view = propagate_to_views(out, slot_track, starts, counts)
            path = out_dir / f"{name}__{EXPERIMENT}.npz"
            np.savez_compressed(
                path, slot_track=slot_track, xyz=out.xyz.astype(np.float32),
                residual_mm=out.residual_mm, margin_mm=out.margin_mm,
                n_views_used=out.n_views_used, source=out.source,
                slot_rays=run["slot_rays"], state=run["state"], withheld=mask,
                **{f"track__{v}": per_view[v] for v in VIEWS})
            record.output(name, path)
            stored = dict(np.load(path))
            summary[name] = {
                "path": str(path), "withheld": int(mask.sum()),
                "reconstruction": report(stored, name),
                "coverage": coverage(slot_track, c01.N_FRAMES),
                "identities": identities(slot_track, round(FPS)),
                "cross_view_agreement": agreement,
            }

        print(f"\n{'arm':<10} {'withheld':>9} {'resolved':>9} {'ids>=1s':>8} "
              f"{'frames in them':>15} {'median life':>12} {'tracks':>8}")
        for name, cell in summary.items():
            ident = cell["identities"]
            print(f"{name:<10} {cell['withheld']:>9,} "
                  f"{cell['reconstruction']['resolved']:>9.2%} "
                  f"{ident['held_1s']:>8,} {ident['frames_in_1s']:>15.2%} "
                  f"{ident['median_lifetime_s']:>11.1f}s {ident['tracks']:>8,}")
        record.param("arms", summary)

        (out_dir / "c23_summary.json").write_text(
            json.dumps({"experiment": EXPERIMENT, "n_groups": 1, "n_experiments": 1,
                        "groups": [ExperimentId.parse(EXPERIMENT).group],
                        "abs_mm": args.abs_mm, "rel_x": args.rel_x,
                        "local_half": args.local_half,
                        "gate_mm": config.max_step_mm,
                        "max_gap_frames": config.max_gap_frames,
                        "arms": summary}, indent=2, default=float),
            encoding="utf-8")

    print(f"\nwrote {out_dir}\nnow score each arm, then restore the baseline:")
    for name in summary:
        print(f"  tools/c09_gt_swap_validation.py --npz {out_dir / f'{name}__{EXPERIMENT}.npz'}")
    print("  tools/c09_gt_swap_validation.py        # no --npz: restores the baseline")


if __name__ == "__main__":
    main()
