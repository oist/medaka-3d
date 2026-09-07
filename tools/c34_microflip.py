"""C34 - repair out-and-back slot flips: the class the stitch deliberately ignores.

=============================================================================
THE CLASS, AND WHY IT SURVIVES EVERYTHING UPSTREAM
=============================================================================

`c30`'s join rule compares 8-frame medians precisely so that a one-frame event
cannot move them - that is what makes it immune to triangulation spikes, and
it is also why a **one-to-three-frame slot flip that immediately reverts**
passes straight through: the medians on both sides agree, the excursion is
invisible to them by construction. The experimenter's 2026-09-03 annotation
holds three such events (21232, 67142, 333619) and their mechanism is known -
the correspondence transiently re-pairs the views (typically when `top_cam`
drops or garbles one animal for a moment) and re-pairs back, leaving two
ghost-triangulated frames and a ~30-115 mm there-and-back teleport in 3D.

The experimenter asked whether what a single-frame 3D jump distance can
identify can be repaired in a final pass. Measured before this tool was
written (production arm, lambda 0.767): **65 windows** (55 one-frame, 9
two-frame), fixing all three annotated events (steps 32.6 -> 3.1,
61.2 -> 3.7, 45.7 -> 1.5 mm), adjacent steps > 30 mm 334 -> 193, and `c09` on
the repaired labelling **identical** - 293 / 0 / 5. Zero measured cost.

=============================================================================
THE RULE - and why it cannot chase a real exchange
=============================================================================

Scan adjacent resolved frames (gaps <= the gate's 73). A window is repaired
only when THREE things hold:

1. at its entry, a non-identity assignment ``q`` beats identity by more than
   **5 mm** (single-frame - this detector runs where the stitch's does not);
2. within **<= 3 resolved frames**, the exact inverse ``q^-1`` beats identity
   by more than 5 mm at the exit;
3. no *other* flip intervenes.

Then ``q`` is applied to the frames **inside the window only** - bounded, no
downstream effect, which is what separates this from `c30`: the stitch decides
identity going forward, this repairs a hiccup and touches nothing else. A real
exchange has no decisive inverse two frames later, so rule 2 cannot fire on
it; a triangulation spike keeps identity as its best assignment, so rule 1
cannot. `tests/test_microflip.py` pins all three behaviours.

The gated tracker and per-view tracks are re-run on the repaired geometry
(both the gated and the label arms are rewritten), because 65 removed
teleports remove gated breaks. Nothing is promoted; `c03` still owns the
fixed paths; `n_groups = 1`.

Run::

    set PYTHONIOENCODING=utf-8
    .venv\\Scripts\\python.exe tools/c34_microflip.py --arm-dir <c30 output dir> --tag <chain>
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from itertools import permutations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import c01_s16_predictions as c01
from c19_viterbi_correspondence import coverage, derive_gate

from medaka3d import load_paths
from medaka3d.provenance import provenance
from medaka3d.reconstruct import N_ANIMALS, Reconstruction
from medaka3d.track import METHODS, UNTRACKED, propagate_to_views
from medaka3d.types import VIEWS

RUN = "bottomup"
GATE_MM = 4.0
GAP_FRAMES = 73

#: Entry and exit must each be this decisive, in mm summed over slots - the
#: same decisiveness the stitch demands of a join.
MARGIN_MM = 5.0

#: Longest repairable window, in resolved frames. The three annotated events
#: are 1-2 frames; anything longer is the stitch's jurisdiction, not this.
MAX_WINDOW = 3

PERMS = tuple(permutations(range(N_ANIMALS)))
IDENTITY = PERMS[0]
INVERSE = {q: tuple(int(i) for i in np.argsort(q)) for q in PERMS}


def best_assignment(a: np.ndarray, b: np.ndarray) -> tuple[tuple[int, ...], float]:
    """Cheapest slot assignment between two frames, and its lead over identity."""
    d = np.linalg.norm(a[:, None] - b[None, :], axis=-1)
    costs = [float(np.nan_to_num(d[(0, 1, 2), q], nan=0.0).sum()) for q in PERMS]
    best = int(np.argmin(costs))
    return PERMS[best], costs[0] - costs[best]


def find_windows(xyz: np.ndarray, ok_idx: np.ndarray,
                 margin_mm: float = MARGIN_MM, max_window: int = MAX_WINDOW,
                 max_gap: int = GAP_FRAMES) -> list[dict]:
    """Out-and-back flip windows, per the pre-registered rule. Pure."""
    windows = []
    i = 1
    while i < ok_idx.size:
        f0, f1 = int(ok_idx[i - 1]), int(ok_idx[i])
        if f1 - f0 > max_gap:
            i += 1
            continue
        q, lead = best_assignment(xyz[f0], xyz[f1])
        if q == IDENTITY or lead <= margin_mm:
            i += 1
            continue
        length = None
        for k in range(1, max_window + 1):
            if i + k >= ok_idx.size:
                break
            fa, fb = int(ok_idx[i + k - 1]), int(ok_idx[i + k])
            if fb - fa > max_gap:
                break
            q2, lead2 = best_assignment(xyz[fa], xyz[fb])
            if q2 == INVERSE[q] and lead2 > margin_mm:
                length = k
                break
            if q2 != IDENTITY:
                break                      # a different flip - not out-and-back
        if length is None:
            i += 1
            continue
        windows.append({"first_frame": int(ok_idx[i]),
                        "last_frame": int(ok_idx[i + length - 1]),
                        "permutation": list(map(int, q)),
                        "entry_margin_mm": round(lead, 1)})
        i += length + 1
    return windows


def apply_windows(arrays: dict, windows: list[dict]) -> dict:
    out = {k: v.copy() for k, v in arrays.items()}
    for w in windows:
        lo, hi, q = w["first_frame"], w["last_frame"] + 1, w["permutation"]
        for v in out.values():
            v[lo:hi] = v[lo:hi][:, q]
    return out


def step_census(xyz: np.ndarray, ok: np.ndarray) -> dict:
    okb = ok[:-1] & ok[1:]
    step = np.linalg.norm(np.diff(xyz, axis=0), axis=-1)[okb]
    step = step[np.isfinite(step)]
    return {"over_10mm": int((step > 10).sum()), "over_30mm": int((step > 30).sum()),
            "max_mm": float(step.max())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("=====")[0].strip())
    parser.add_argument("--experiment", default="20251225_K_1_2_FCMC")
    parser.add_argument("--arm-dir", type=Path, required=True,
                        help="a c30 output directory holding C30-gated and "
                             "C30-label arms")
    parser.add_argument("--tag", required=True,
                        help="chain name for the output leaf - required, so no "
                             "two chains can ever land in one place")
    parser.add_argument("--lambdas", nargs="+", type=float,
                        default=[0.0, 0.1, 0.25, 0.5, 0.767, 2.0, 8.0])
    args = parser.parse_args()

    paths = load_paths()
    experiment = args.experiment
    out_dir = paths.work_dir / "c34_microflip" / f"{experiment}__{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    starts, counts = {}, {}
    for view in VIEWS:
        detections = c01.load(paths.work_dir, experiment, view, RUN)
        starts[view], counts[view] = detections.start, int(detections.frame_idx.size)

    with provenance("microflip", out_dir, name="c34") as record:
        record.params_from(args)
        record.param("n_groups", 1)
        record.param("n_experiments", 1)
        record.param("margin_mm", MARGIN_MM)
        record.param("max_window_frames", MAX_WINDOW)
        record.param("gate_mm", GATE_MM)
        record.param("gate_is_post_hoc", True)
        record.param("lambda_is_chosen", False)

        arms = {}
        for lam in args.lambdas:
            gated_path = args.arm_dir / f"C30-gated__{experiment}__lam{lam:g}.npz"
            label_path = args.arm_dir / f"C30-label__{experiment}__lam{lam:g}.npz"
            for path in (gated_path, label_path):
                if not path.is_file():
                    raise SystemExit(f"no arm at {path}")
            gated = np.load(gated_path)
            record.input(f"gated_lam{lam:g}", gated_path)
            started = time.time()

            xyz = gated["xyz"].astype(np.float64)
            ok = ((gated["n_views_used"] >= 2)
                  & np.isfinite(xyz).all(axis=(1, 2)))
            before = step_census(xyz, ok)
            windows = find_windows(xyz, np.flatnonzero(ok))

            fixed = apply_windows(
                {"xyz": xyz, "residual_mm": gated["residual_mm"],
                 "slot_rays": gated["slot_rays"], "source": gated["source"]},
                windows)
            after = step_census(fixed["xyz"], ok)

            out = Reconstruction(
                experiment=experiment, views=tuple(VIEWS),
                xyz=fixed["xyz"].astype(np.float32), source=fixed["source"],
                residual_mm=fixed["residual_mm"], margin_mm=gated["margin_mm"],
                n_views_used=gated["n_views_used"])
            derived = derive_gate(out)
            config = dataclasses.replace(derived, max_step_mm=GATE_MM,
                                         max_gap_frames=GAP_FRAMES)
            slot_track = METHODS["gated"](out, config)
            per_view = propagate_to_views(out, slot_track, starts, counts)
            path = out_dir / f"C34-gated__{experiment}__lam{lam:g}.npz"
            np.savez_compressed(
                path, slot_track=slot_track, xyz=out.xyz,
                residual_mm=out.residual_mm.astype(np.float32),
                margin_mm=gated["margin_mm"],
                n_views_used=gated["n_views_used"], source=out.source,
                slot_rays=fixed["slot_rays"],
                **({"slot_perm": gated["slot_perm"]}
                   if "slot_perm" in gated.files else {}),
                **{f"track__{v}": per_view[v] for v in VIEWS})
            record.output(f"C34-gated__lam{lam:g}", path)

            label_track = np.where(
                out.resolved[:, None],
                np.arange(N_ANIMALS, dtype=np.int32)[None, :],
                UNTRACKED).astype(np.int32)
            per_view_label = propagate_to_views(out, label_track, starts, counts)
            lpath = out_dir / f"C34-label__{experiment}__lam{lam:g}.npz"
            np.savez_compressed(
                lpath, slot_track=label_track, xyz=out.xyz,
                residual_mm=out.residual_mm.astype(np.float32),
                margin_mm=gated["margin_mm"],
                n_views_used=gated["n_views_used"], source=out.source,
                **{f"track__{v}": per_view_label[v] for v in VIEWS})
            record.output(f"C34-label__lam{lam:g}", lpath)

            lengths = [w["last_frame"] - w["first_frame"] + 1 for w in windows]
            arms[f"{lam:g}"] = {
                "lambda_mm": float(lam), "path": str(path),
                "label_path": str(lpath), "windows": len(windows),
                "window_lengths": {str(k): lengths.count(k)
                                   for k in range(1, MAX_WINDOW + 1)},
                "steps_before": before, "steps_after": after,
                "coverage_gated": coverage(slot_track, xyz.shape[0]),
                "window_list": windows,
                "seconds": round(time.time() - started, 1)}
            a = arms[f"{lam:g}"]
            print(f"  lambda {lam:<6g} windows {len(windows):3d} "
                  f"{a['window_lengths']}   steps>10mm "
                  f"{before['over_10mm']:,} -> {after['over_10mm']:,}   >30mm "
                  f"{before['over_30mm']:,} -> {after['over_30mm']:,}   gated "
                  f"tracks {a['coverage_gated']['tracks']}   "
                  f"{a['seconds']:.0f}s", flush=True)

        record.param("arms", {k: {kk: vv for kk, vv in v.items()
                                  if kk != "window_list"}
                              for k, v in arms.items()})
        (out_dir / "c34_summary.json").write_text(
            json.dumps({"experiment": experiment, "n_groups": 1,
                        "n_experiments": 1, "input_dir": str(args.arm_dir),
                        "tag": args.tag, "margin_mm": MARGIN_MM,
                        "max_window_frames": MAX_WINDOW,
                        "lambda_is_chosen": False, "arms": arms},
                       indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
