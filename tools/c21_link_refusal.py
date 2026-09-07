"""C21 — two repairs for the one swap stage C did not fix, measured separately
and together.

=============================================================================
THE CASE THIS EXISTS FOR
=============================================================================

`c20` removed the "exactly three detections per view" rule. Of `c19`'s two
measured female swaps, one became a break; **the other survived and is the only
swap left** at a held gate: the interval 82,121 -> 83,265, the crossing the
experimenter reviewed in `SWAP_i82121_x80554_p12.mp4`.

Read frame by frame, K_1_2 at 82,715-82,717:

* `top_cam` label 0 — the fish that turns and darts — leaves **side1 (as its
  label 2) and side2 (as its label 0) on the same frame**, 82,716. Under the
  chosen state those are the same animal, so one animal drops to a single camera
  while the other two keep all three.
* The female is `top_cam` label 2. **She is visible in all three cameras on
  every frame of the window.** Her 3D position at 82,716 exists in stage C's
  output — three cameras, 0.22 mm residual — where `c19` had nothing.
* `n_views_used` is the **minimum** over the three animals, so the frame is
  still marked unresolved and the linker never sees her position. Removing the
  per-view cardinality rule left a per-frame one standing.
* `gated` therefore links 82,715 straight to 82,717, and the assignment problem
  there has **four** live tracks, not three::

        track   age   ceiling      ->s0     ->s1     ->s2
          612     2       8.0     12.48    39.00    11.00   the animal that left
          627     2       8.0      2.52    41.65     2.07   the female
          632     2       8.0     40.83     0.04    40.63   the still fish
          636    56     224.0     16.15    27.57    15.07   a 56-frame-old fragment

  The ceiling is ``max_step_mm x frames absent``, so the fragment carries
  **224 mm in a 192 mm tank** and can claim any animal in the volume. Whichever
  order the ceiling and the assignment are applied in, the cheapest total puts
  the female on slot 0.

So three rules are candidates — one in the reconstruction, two in the linker —
and they are independent. This tool crosses them.

=============================================================================
THE TWO BY TWO
=============================================================================

**Which frames the linker is offered**

``whole``    a frame counts only when every animal has a 3D position — what
             `c20` produced, and what every run before it did.
``partial``  a frame offers the animals it has. `n_views_used` becomes the
             camera count behind the best-supported animal, so `report`'s
             ``views_2`` / ``views_3`` buckets describe **that** animal on a
             partial frame; nothing else in the file reads them.

**How the linker refuses**

``gated``    Hungarian on raw distance, ceiling applied afterwards. Unchanged.
``strict``   ceiling applied first, so a track with nothing in reach goes
             unmatched instead of being parked on someone else's animal.

**How far a stale track may reach**

``unbounded``  ``max_step_mm x frames absent``, with no limit. What every run
               before this one used.
``bounded``    the same, stopped at ``VELOCITY_FRAMES x max_step_mm`` = 12.0 mm.
               The horizon is not chosen to fix this frame: it is the one the
               velocity estimate already uses, and past it the bound has stopped
               describing how far a fish could swim and started describing the
               tank.

Eight arms, **one gate for all of them** — `c19`'s 4.0 mm / 73 frames — because
`c20` showed `derive_gate` is contaminated by correspondence changes and a
comparison across derived gates is a comparison of gates. That makes the gate a
post-hoc choice and it is labelled as one; the derived value is printed beside
it for every arm.

Nothing is promoted. `c03`, `c13`, `c16`, `c18`, `c19` and `c20` are not
modified. Clause 1 needs `c09` run against these outputs, which overwrites
`c09`'s fixed output path, so re-run `c09` with no ``--npz`` afterwards.

**n_groups = 1, n_experiments = 1, female only. One swap against two is not a
difference at this sample size. What this tool can establish is whether a named
mechanism removes a named failure, not whether it is better on average.**

=============================================================================
WHAT IT FOUND (2026-08-19) — THE REPAIR DOES NOT PAY
=============================================================================

Only `strict` **and** `bounded` together remove the reviewed swap, which is what
the four-track table predicts, and they remove it by turning it into a **break**
rather than by carrying her identity through. The price is other intervals:
swaps go 1 -> 3 (bounded) or 1 -> 5 (unbounded), and clause 1 reads "an increase
is disqualifying regardless of every other number". `gated` holds at 1 swap in
every variant. A mechanism can be real, reproducible in a unit test with the
measured numbers, and still a net loss across 298 intervals.

The one thing that survived its own measurement is `partial`: **resolved
98.55% -> 99.94%**, run coverage 93.13% -> 94.31%, swaps and breaks unchanged.
It is the half of stage C `c20` left behind — the per-frame all-or-nothing rule
that outlived the per-view one. See the decision document's addendum for what it
needs before promotion.

Run::

    set PYTHONIOENCODING=utf-8
    .venv\\Scripts\\python.exe tools/c21_link_refusal.py
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
from c13_pectoral_anchor import report
from c19_viterbi_correspondence import coverage, derive_gate
from c20_uncapped_correspondence import cross_view_agreement

from medaka3d import load_paths
from medaka3d.calib import load_rig
from medaka3d.cohort import ExperimentId
from medaka3d.provenance import provenance
from medaka3d.reconstruct import Reconstruction
from medaka3d.track import METHODS, TrackConfig, propagate_to_views
from medaka3d.types import VIEWS

EXPERIMENT = "20251225_K_1_2_FCMC"
RUN = "bottomup"
#: `c20`'s gate-held run. Its reconstruction is the same solve as the
#: pre-registered one — only the gate differs — and its `C-gated` arm is
#: exactly this tool's `gated x whole`, which is what makes the assertion below
#: an assertion rather than a coincidence.
STAGE_C = (Path("c20_uncapped_correspondence")
           / "mouth-pectoral_anchor-pectoral_gate-4")

#: `c19`'s gate, held across every arm. Post-hoc, and the reason is in the
#: module docstring: a derived gate is not comparable between reconstructions.
HELD_GATE_MM, HELD_GAP_FRAMES, VELOCITY_FRAMES = 4.0, 73, 3

#: The frame the whole tool is about, and the interval `c09` scores it inside.
CASE_FRAME, CASE_INTERVAL = 82_717, (82_121, 83_265)


def reconstructions(run: dict, views: tuple[str, ...]) -> dict[str, Reconstruction]:
    """``whole`` and ``partial``, from one stage-C solve.

    The geometry is identical in both; only ``n_views_used`` differs, and it is
    the single thing that decides which frames the linker is offered.
    """
    xyz = run["xyz"].astype(float)
    here = np.isfinite(xyz).all(axis=2)  # (n_frames, n_animals) who has a position
    common = dict(experiment=EXPERIMENT, views=views, xyz=xyz, source=run["source"],
                  residual_mm=run["residual_mm"], margin_mm=run["margin_mm"])
    best = np.where(here, run["slot_rays"], 0).max(axis=1).astype(np.int8)
    return {
        "whole": Reconstruction(**common, n_views_used=run["n_views_used"]),
        "partial": Reconstruction(**common, n_views_used=np.where(here.any(axis=1), best, 0)),
    }


def case_report(slot_track: np.ndarray, run: dict) -> dict:
    """What each arm does at the reviewed crossing, frame by frame."""
    return {
        "frames": {int(f): slot_track[f].tolist()
                   for f in range(CASE_FRAME - 2, CASE_FRAME + 2)},
        "female_slot_82715": 2,
        "track_at_82715": int(slot_track[82_715, 2]),
        "track_at_82717": int(slot_track[82_717, 2]),
        "kept_her_slot": bool(slot_track[82_717, 2] == slot_track[82_715, 2]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("=====")[0].strip())
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument("--gate-mm", type=float, default=HELD_GATE_MM)
    parser.add_argument("--max-gap-frames", type=int, default=HELD_GAP_FRAMES)
    args = parser.parse_args()

    paths = load_paths()
    rig = load_rig(CALIBRATION)
    source_npz = paths.work_dir / STAGE_C / f"C-gated__{args.experiment}.npz"
    if not source_npz.is_file():
        raise SystemExit(f"run c20 first; no stage-C output at {source_npz}")
    out_dir = paths.work_dir / "c21_link_refusal"
    out_dir.mkdir(parents=True, exist_ok=True)

    with provenance("link_refusal", out_dir, name="c21") as record:
        record.params_from(args)
        record.input("stage_c", source_npz)
        record.param("n_groups", 1)
        record.param("n_experiments", 1)
        print(f"stage C   {source_npz}")
        print("n_groups = 1, n_experiments = 1 — K_1_2, female only\n")

        run = dict(np.load(source_npz))
        starts, counts = {}, {}
        for view in VIEWS:
            detections = c01.load(paths.work_dir, args.experiment, view, RUN)
            starts[view] = detections.start
            counts[view] = int(detections.frame_idx.shape[0])

        built = reconstructions(run, rig.views)
        config = TrackConfig(max_step_mm=args.gate_mm, max_gap_frames=args.max_gap_frames,
                             velocity_frames=VELOCITY_FRAMES)
        print(f"gate held at {config.max_step_mm} mm / {config.max_gap_frames} frames "
              f"for every arm — POST-HOC; derived values printed below")
        for name, out in built.items():
            print(f"  {name:<8} resolved {out.resolved.mean():7.2%}   "
                  f"animals with a position {np.isfinite(out.xyz).all(axis=2).mean():7.2%}   "
                  f"derive_gate would say {derive_gate(out).max_step_mm} mm")
            record.param(f"derived_gate.{name}", derive_gate(out).max_step_mm)
            record.param(f"resolved.{name}", float(out.resolved.mean()))

        # --- the assertion that this is c20's reconstruction, not a new one ---
        rebuilt = METHODS["gated"](built["whole"], config)  # unbounded: c20's own arm
        same = bool((rebuilt == run["slot_track"]).all())
        print(f"\n  gated x whole reproduces c20's stored tracks exactly: {same}")
        record.param("reproduces_c20", same)
        if not same:
            raise SystemExit(
                "gated x whole must be c20's own arm; it is not, so the reconstruction "
                "was rebuilt wrongly and nothing below can be compared to c20.")

        # The cross-view pairing depends on `source` and the state alone, both of
        # which are the same solve in every arm, so it is computed once rather
        # than eight times over identical inputs.
        agreement = cross_view_agreement(rig, run)

        summary = {}
        ceilings = {"unbounded": float("inf"),
                    "bounded": VELOCITY_FRAMES * config.max_step_mm}
        for offer, out in built.items():
            for method in ("gated", "strict"):
              for reach, cap in ceilings.items():
                name = f"{method}-{offer}-{reach}"
                started = time.time()
                slot_track = METHODS[method](
                    out, dataclasses.replace(config, max_ceiling_mm=cap))
                per_view = propagate_to_views(out, slot_track, starts, counts)
                path = out_dir / f"{name}__{args.experiment}.npz"
                np.savez_compressed(
                    path, slot_track=slot_track,
                    xyz=out.xyz.astype(np.float32),
                    residual_mm=out.residual_mm.astype(np.float32),
                    margin_mm=out.margin_mm.astype(np.float32),
                    n_views_used=out.n_views_used, source=out.source,
                    slot_rays=run["slot_rays"], state=run["state"],
                    **{f"track__{v}": per_view[v] for v in VIEWS},
                )
                record.output(name, path)
                stored = dict(np.load(path))
                summary[name] = {
                    "path": str(path), "method": method, "offer": offer,
                    "reach": reach, "max_ceiling_mm": cap,
                    "reconstruction": report(stored, name),
                    "coverage": coverage(slot_track, c01.N_FRAMES),
                    "cross_view_agreement": agreement,
                    "case": case_report(slot_track, stored),
                    "seconds": round(time.time() - started, 1),
                }

        print(f"\n{'arm':<26} {'resolved':>9} {'tracks':>7} {'top3 track':>11} "
              f"{'top3 runs>=1s':>14}  the reviewed crossing at 82,717")
        for name, cell in summary.items():
            case = cell["case"]
            print(f"{name:<26} {cell['reconstruction']['resolved']:>9.2%} "
                  f"{cell['coverage']['tracks']:>7,} "
                  f"{cell['coverage']['top3_track_coverage']:>11.2%} "
                  f"{cell['coverage']['top3_run_coverage']:>14.2%}  "
                  f"female {'STAYS' if case['kept_her_slot'] else 'MOVES'} "
                  f"({case['track_at_82715']} -> {case['track_at_82717']})")
        tail = np.mean([agreement[k]["agreement_rate"] for k in agreement
                        if k.startswith("tail")])
        print(f"\ntail-derived cross-view agreement, identical for every arm: {tail:.4f}")
        record.param("arms", summary)

        (out_dir / "c21_summary.json").write_text(
            json.dumps({
                "experiment": args.experiment, "n_groups": 1, "n_experiments": 1,
                "groups": [ExperimentId.parse(args.experiment).group],
                "stage_c": str(source_npz),
                "gate_mm": config.max_step_mm, "max_gap_frames": config.max_gap_frames,
                "gate_is_post_hoc": True,
                "case_frame": CASE_FRAME, "case_interval": list(CASE_INTERVAL),
                "arms": summary,
            }, indent=2, default=float),
            encoding="utf-8")

    print(f"\nwrote {out_dir}\nnow score every arm, then restore the baseline:")
    for name in summary:
        print(f"  tools/c09_gt_swap_validation.py --npz "
              f"{out_dir / f'{name}__{args.experiment}.npz'}")
    print("  tools/c09_gt_swap_validation.py        # no --npz: restores the JSON "
          "c11/c14/c15 read")


if __name__ == "__main__":
    main()
