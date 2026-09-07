"""C04 — score the three trackers and apply a decision rule written before they ran.

=============================================================================
SPECIFICATION — fixed in code before the first score was read
=============================================================================

**There is no ground truth and there will be none this session.** No human track
exists for M1 or M2 anywhere in this project; ground truth labels the female
alone. Every number below is therefore a property of the prediction, measurable
without a human, in the same spirit as `s10`. They can prove a tracker wrong.
**They cannot prove one right**, and a tracker that swaps two animals mid-contact
and swaps them back is invisible to all of them.

**Metrics.**

1. **switch-candidate rate** (primary). Between adjacent frames, the tracker
   asserts a correspondence by re-using track IDs; geometry asserts one by
   minimum displacement. Where they disagree, either the tracker swapped or the
   animals genuinely exchanged places within 6 ms — indistinguishable without a
   human, hence *candidates*. Computed **on the 3D slots in mm**, one test per
   frame transition. It is deliberately not pooled over the three cameras: a 3D
   tracker assigns one ID across all three at once, so pooling would count the
   same decision three times and shrink its own p-values.

2. **fragmentation**. Tracks per animal, track lengths, coverage. This is the
   metric that stops a tracker winning by refusing to link — a fresh ID every
   frame has a perfect switch rate by construction, and `test_tracking_metrics`
   pins that.

3. **encounter identity preservation**. `medaka3d.eval.encounters` segments the
   reconstructed 3D trajectories into windows bounded on both sides by sustained
   separation, and each tracker is asked whether the two animals that entered
   kept their IDs to the exit. Necessary, not sufficient: unbroken IDs through a
   crossing is what being right looks like, but it is also what a confident swap
   looks like.

4. **three-view consistency**. Diagnostic, and it must be read as one. A 3D
   tracker assigns IDs jointly, so its three cameras agree **by construction**
   and a rate of 1.000 there measures nothing. What *is* informative is computed
   on the **tail** node — the trackers link on mouths, so a tail-derived
   cross-view pairing is independent evidence — plus the per-view track-jump
   rate, which is the only way a wrong per-frame cross-view correspondence
   becomes visible at all: the 3D track stays smooth while the 2D instance it
   owns hops between animals.

**Thresholds are derived, per view, and printed.** The 2D gate is `s10`'s half
median mouth-tail span from `labels.round8.<view>.train` — 112.8 px -> 60 px top,
94.0 px -> 50 px side. The 3D gate is half the median *reconstructed* body length
in mm. Nothing here is hardcoded.

**Decision rule, fixed before the run and applied as written.**

    A beats B  iff  A's switch-candidate rate is at least 20% lower in relative
                    terms,  AND  a two-proportion z-test gives p < 0.05,
                    AND  A's fragmentation (tracks per animal) is not worse.

    If A wins on switches and loses on fragmentation, that is reported as a
    **trade-off and is not called a win.**

    If no method separates from another under this rule, the answer is
    **"indistinguishable"** and no ranking of the noise is offered.

**What the rule could not do, recorded here after the fact.** The
switch-candidate rate compares a tracker against the per-frame minimum-
displacement assignment, and all three methods are *built on* that assignment:
`nearest` is it, `gated` differs only by refusing links (which removes the
transition from scoring rather than creating a disagreement), and `predictive` is
the only one that can genuinely deviate. The rule accordingly separated
`predictive` from the other two and left those two tied at exactly zero, and it
declined to select — correctly. The method was chosen on the **track-jump** rate,
which compares a tracker against physics rather than against another algorithm
and is the only metric here that is not constructional. See
`docs/decisions/2026-08-13-tracking-3d.md`.

**n_groups = 1.** Group K — `K_1_1`, `K_1_2`, `K_1_3` are one trio of fish on
three consecutive days, repeated measures rather than three replicates. The
corpus holds 14 groups and 13 are unmeasured here. Whatever this chooses, it
chooses on one trio.

Run::

    set PYTHONIOENCODING=utf-8
    .venv\\Scripts\\python.exe tools/c04_score_tracking.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import c01_s16_predictions as c01
from c02_tracking_probe import CALIBRATION
from s10_shared_node_20k import median_span_px, tail_jump_threshold_px

from medaka3d import load_catalog, load_paths
from medaka3d.calib import load_rig, triangulate
from medaka3d.cohort import ExperimentId
from medaka3d.eval import EncounterConfig, find_encounters
from medaka3d.eval.tracking import (
    UNTRACKED,
    ViewTracks,
    fragmentation,
    identity_preservation,
    modal_agreement,
    switch_candidates,
    track_jumps,
    two_proportion_z,
)
from medaka3d.provenance import provenance
from medaka3d.reconstruct import N_ANIMALS
from medaka3d.track import METHODS
from medaka3d.types import VIEWS

#: The pre-registered margin a winner must clear, and the significance level.
MIN_RELATIVE_REDUCTION = 0.20
ALPHA = 0.05


def load_run(work_dir: Path, method: str, experiment: str, run: str) -> dict:
    path = work_dir / "c03_track_3d" / run / f"{method}__{experiment}.npz"
    if not path.is_file():
        raise SystemExit(
            f"no tracking output at {path}. Run tools/c03_track_3d.py first."
        )
    return dict(np.load(path, allow_pickle=False))


def slot_tracks_3d(run: dict, experiment: str) -> ViewTracks:
    """The 3D slots as a flat `ViewTracks` in mm — the primary metric's input."""
    slot_track = run["slot_track"]
    xyz = run["xyz"].astype(float)
    resolved = slot_track.max(axis=1) >= 0
    frames = np.flatnonzero(resolved)
    return ViewTracks(
        experiment=experiment,
        view="reconstruction",
        frame_idx=np.repeat(frames, N_ANIMALS),
        mouth=xyz[frames].reshape(-1, 3),
        track=slot_track[frames].reshape(-1).astype(np.int32),
        n_frames=slot_track.shape[0],
    )


def view_tracks(run: dict, detections: c01.Detections, view: str) -> ViewTracks:
    """One camera's detections carrying the 3D track IDs propagated back onto them."""
    return ViewTracks(
        experiment=detections.experiment,
        view=view,
        frame_idx=detections.frame_idx.astype(np.int64),
        mouth=detections.node("mouth").astype(float),
        track=run[f"track__{view}"],
        n_frames=detections.n_frames,
    )


def node_pairing(rig, run: dict, node: str, reference: str, other: str) -> np.ndarray:
    """(n_frames, 3) per-slot agreement between a node's cross-view pairing and the tracker's.

    The trackers link on **mouths** and the correspondence was solved on mouths,
    so re-deriving the cross-view pairing from a *different* node is evidence
    they did not generate. For each frame the node's three points in each view
    are triangulated under all six pairings and the cheapest is taken; the answer
    is expressed in track IDs, so it is comparable across frames. A row is 1
    where the two agree.

    **Use the pectoral for this.** The tail is missing in 4.3-4.8% of otherwise
    complete frames, and a missing point forces some pairing to be cheapest
    regardless of the geometry — which reads as disagreement and is not. Frames
    where either view lacks the node are marked unresolved rather than scored;
    the difference between the two is large enough to change the headline (tail:
    90.6% raw, 97.0% among frames where the tail was actually present).
    """
    from medaka3d.reconstruct import PERMUTATIONS

    slot_track = run["slot_track"]
    resolved = slot_track.max(axis=1) >= 0
    frames = np.flatnonzero(resolved)
    out = np.full((slot_track.shape[0], N_ANIMALS), -1, dtype=np.int64)
    if frames.size == 0:
        return out

    columns = {v: i for i, v in enumerate(rig.views)}
    source = run["source"]
    both = frames[
        (source[frames, :, columns[reference]] >= 0).all(axis=1)
        & (source[frames, :, columns[other]] >= 0).all(axis=1)
    ]
    if both.size == 0:
        return out

    xy_ref = run[f"{node}__{reference}"][both].astype(float)
    xy_other = run[f"{node}__{other}"][both].astype(float)
    present = np.isfinite(xy_ref).all(axis=(1, 2)) & np.isfinite(xy_other).all(axis=(1, 2))
    both, xy_ref, xy_other = both[present], xy_ref[present], xy_other[present]
    if both.size == 0:
        return out
    o_ref, d_ref = rig.camera(reference).rays(xy_ref)
    o_oth, d_oth = rig.camera(other).rays(xy_other)

    costs = []
    for perm in PERMUTATIONS:
        origins = np.stack([o_ref, o_oth[:, perm, :]], axis=-2)
        directions = np.stack([d_ref, d_oth[:, perm, :]], axis=-2)
        _, rms = triangulate(origins, directions)
        costs.append(rms.sum(axis=-1))
    winner = np.argmin(np.where(np.isfinite(costs), costs, np.inf), axis=0)
    perms = np.asarray(PERMUTATIONS)[winner]

    # Row r of a view holds the animal whose slot carries that row in `source`,
    # so argsort turns "slot -> row" into "row -> slot" and lets a track ID be
    # attached to each detector row.
    ref_ids = np.take_along_axis(
        slot_track[both], np.argsort(source[both, :, columns[reference]], axis=1), axis=1
    )
    oth_ids = np.take_along_axis(
        slot_track[both], np.argsort(source[both, :, columns[other]], axis=1), axis=1
    )
    # 1 where the tail-derived pairing names the same animal the mouth-derived
    # one did. A frame of [1, 1, 1] is agreement; anything else is not.
    out[both] = (np.take_along_axis(oth_ids, perms, axis=1) == ref_ids).astype(np.int64)
    return out


def reference_chain(xyz: np.ndarray, resolved: np.ndarray) -> np.ndarray:
    """(n_frames, 3) slot index per canonical column — the geometric reference chain.

    **Not a tracker and not a competitor.** It exists so that encounters are
    defined once, identically for every method: the three animals' proximity is a
    property of the shared reconstruction, and only the *label* of which pair came
    close needs a consistent column. The chain follows the forced assignment
    across adjacent resolved frames and, across a gap, matches on last known
    position with no gate, so a column never dies and the window is never split.

    Its limitation is the same one the switch-candidate rate has and cannot
    escape without a human: where the chain itself follows an animal through a
    true crossing incorrectly, a method that disagreed with it reads as broken.
    """
    from medaka3d.reconstruct import PERMUTATIONS

    n_frames = xyz.shape[0]
    chain = np.full((n_frames, N_ANIMALS), -1, dtype=np.int8)
    frames = np.flatnonzero(resolved)
    if frames.size == 0:
        return chain
    chain[frames[0]] = np.arange(N_ANIMALS)
    previous = xyz[frames[0]]

    permutations_array = np.asarray(PERMUTATIONS)
    for frame in frames[1:]:
        points = xyz[frame]
        cost = np.linalg.norm(previous[:, None, :] - points[None, :, :], axis=-1)
        totals = [cost[[0, 1, 2], p].sum() for p in PERMUTATIONS]
        chain[frame] = permutations_array[int(np.argmin(totals))]
        previous = points[chain[frame]]
    return chain


def judge(challenger: dict, baseline: dict) -> dict:
    """The pre-registered rule. Nothing here was decided after a number was seen."""
    a, b = challenger["switch"], baseline["switch"]
    rate_a, rate_b = a["switch_candidate_rate"], b["switch_candidate_rate"]
    relative = (rate_b - rate_a) / rate_b if rate_b > 0 else float("nan")
    z, p = two_proportion_z(
        a["switch_candidates"], a["scored"], b["switch_candidates"], b["scored"]
    )
    frag_a = challenger["fragmentation"]["tracks_per_animal"]
    frag_b = baseline["fragmentation"]["tracks_per_animal"]
    switch_win = bool(relative >= MIN_RELATIVE_REDUCTION and p < ALPHA)
    fragmentation_ok = bool(frag_a <= frag_b)
    if switch_win and fragmentation_ok:
        verdict = "challenger wins"
    elif switch_win:
        verdict = "TRADE-OFF - not a win"
    else:
        verdict = "no win"
    return {
        "challenger_switch_rate": rate_a,
        "baseline_switch_rate": rate_b,
        "relative_reduction": relative,
        "z": z,
        "p": p,
        "challenger_tracks_per_animal": frag_a,
        "baseline_tracks_per_animal": frag_b,
        "switch_criterion_met": switch_win,
        "fragmentation_not_worse": fragmentation_ok,
        "verdict": verdict,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("=====")[0].strip())
    parser.add_argument("--experiments", nargs="*", default=list(c01.EXPERIMENTS))
    parser.add_argument("--methods", nargs="*", default=sorted(METHODS))
    parser.add_argument("--run", choices=sorted(c01.RUNS), default=c01.DEFAULT_RUN,
                        help="Which detection run's tracks to score.")
    parser.add_argument("--calibration", type=Path, default=CALIBRATION)
    parser.add_argument("--close-mm", type=float, default=EncounterConfig().close_mm)
    args = parser.parse_args()

    paths = load_paths()
    out_dir = paths.work_dir / "c04_score_tracking" / args.run
    rig = load_rig(args.calibration)
    catalog = load_catalog(paths.asset_root)

    with provenance("score_tracking", out_dir, name=f"c04_{args.run}") as record:
        record.params_from(args)
        record.input("calibration", rig.manifest)
        groups = sorted({ExperimentId.parse(e).group for e in args.experiments})
        record.param("n_groups", len(groups))
        record.param("run", args.run)

        # --- thresholds, derived and printed ------------------------------
        gate_px = {}
        print("gates — half a median body span, per view, from the labels")
        for view in VIEWS:
            key = f"labels.round8.{'top' if view == 'top_cam' else 'side'}.train"
            label_path = catalog.get(key)
            record.input(key, label_path)
            span, n_span = median_span_px(label_path)
            gate_px[view] = tail_jump_threshold_px(span)
            print(f"  {view:<12} median span {span:7.2f} px ({n_span:>5} instances) "
                  f"-> {gate_px[view]:.0f} px")

        detections = {
            (e, v): c01.load(paths.work_dir, e, v, args.run)
            for e in args.experiments
            for v in VIEWS
        }

        # 3D gate: half the median reconstructed mouth-tail length, measured
        # from the reconstructions themselves rather than converted from pixels.
        body = []
        for experiment in args.experiments:
            run = load_run(paths.work_dir, args.methods[0], experiment, args.run)
            resolved = run["slot_track"].max(axis=1) >= 0
            frames = np.flatnonzero(resolved)
            source = run["source"][frames]
            take = (source >= 0).all(axis=2).all(axis=1)
            frames, source = frames[take], source[take]
            origins, directions = [], []
            for column, view in enumerate(rig.views):
                o, d = rig.camera(view).rays(run[f"tail__{view}"][frames].astype(float))
                rows = np.arange(frames.size)[:, None]
                origins.append(o[rows, source[:, :, column]])
                directions.append(d[rows, source[:, :, column]])
            tail_xyz, _ = triangulate(
                np.stack(origins, axis=-2), np.stack(directions, axis=-2)
            )
            body.append(np.linalg.norm(tail_xyz - run["xyz"][frames], axis=-1).ravel())
        lengths = np.concatenate(body)
        lengths = lengths[np.isfinite(lengths)]
        gate_mm = float(np.median(lengths)) / 2.0
        print(f"  {'3D':<12} median body length {2 * gate_mm:7.2f} mm "
              f"({lengths.size:,} instances) -> {gate_mm:.2f} mm")
        record.param("gate_px", gate_px)
        record.param("gate_mm", gate_mm)
        print(f"\nn_groups = {len(groups)} {groups} — one trio, three consecutive days\n")

        # --- the reference chain, built once for every method ---------------
        # Encounters must be the same windows for all three methods or the
        # preserved rates are not comparable. Proximity is a property of the
        # shared reconstruction; only the column labelling needs a chain.
        chains = {}
        for experiment in args.experiments:
            run = load_run(paths.work_dir, args.methods[0], experiment, args.run)
            chains[experiment] = reference_chain(
                run["xyz"].astype(float), run["slot_track"].max(axis=1) >= 0
            )

        # --- score ---------------------------------------------------------
        results: dict[str, dict] = {}
        encounter_config = EncounterConfig(close_mm=args.close_mm)
        for method in args.methods:
            pooled = {
                "switch": Counter(),
                "jumps": {v: Counter() for v in VIEWS},
                "encounters": Counter(),
            }
            per_experiment = {}
            frag_lengths: Counter = Counter()
            consistency = {}
            for experiment in args.experiments:
                run = load_run(paths.work_dir, method, experiment, args.run)
                slots = slot_tracks_3d(run, experiment)
                switch = switch_candidates(slots, gate=gate_mm)
                frag = fragmentation(slots)

                chain = chains[experiment]
                rows = np.arange(chain.shape[0])[:, None]
                canonical_xyz = np.where(
                    (chain >= 0)[:, :, None],
                    run["xyz"][rows, np.maximum(chain, 0)],
                    np.nan,
                )
                encounters = find_encounters(
                    canonical_xyz, experiment=experiment, config=encounter_config
                )
                canonical_track = np.where(
                    chain >= 0, run["slot_track"][rows, np.maximum(chain, 0)], UNTRACKED
                )
                preserved = identity_preservation(encounters, canonical_track)

                jumps = {}
                for view in VIEWS:
                    jumps[view] = track_jumps(
                        view_tracks(run, detections[(experiment, view)], view),
                        gate=gate_px[view],
                    )
                    pooled["jumps"][view].update(
                        {k: v for k, v in jumps[view].items() if isinstance(v, int)}
                    )
                per_experiment[experiment] = {
                    "switch": switch,
                    "fragmentation": frag,
                    "encounters": preserved,
                    "jumps": jumps,
                }
                pooled["switch"].update({k: v for k, v in switch.items() if isinstance(v, int)})
                pooled["encounters"].update(
                    {k: v for k, v in preserved.items() if isinstance(v, int)}
                )
                # Track IDs restart at 0 in every experiment, so the pooled
                # length distribution is keyed by (experiment, id) — merging two
                # experiments' id 0 would fabricate one long track from two.
                labels = run["slot_track"][run["slot_track"] >= 0]
                frag_lengths.update((experiment, int(i)) for i in labels)

                consistency[experiment] = {
                    f"{node}:{VIEWS[0]}~{other}": modal_agreement(
                        node_pairing(rig, run, node, VIEWS[0], other)
                    )
                    for node in ("pectoral", "tail")
                    for other in VIEWS[1:]
                }

            pooled_lengths = np.asarray(sorted(frag_lengths.values(), reverse=True))
            results[method] = {
                "switch": {
                    **dict(pooled["switch"]),
                    "switch_candidate_rate": (
                        pooled["switch"]["switch_candidates"] / pooled["switch"]["scored"]
                        if pooled["switch"]["scored"]
                        else float("nan")
                    ),
                },
                "fragmentation": {
                    "tracks": int(pooled_lengths.size),
                    "tracks_per_animal": float(
                        pooled_lengths.size / (N_ANIMALS * len(args.experiments))
                    ),
                    "median_track_frames": float(np.median(pooled_lengths)),
                    "longest_track_frames": int(pooled_lengths.max()),
                    # The honest fragmentation number. `tracks_per_animal` counts
                    # a 2-frame speck the same as a 100,000-frame track, and on
                    # this data 78% of tracks are debris of <=10 frames, so the
                    # ratio overstates the problem by an order of magnitude.
                    "tracks_to_90": int(
                        np.searchsorted(np.cumsum(pooled_lengths) / pooled_lengths.sum(), 0.90) + 1
                    ),
                    "tracks_to_99": int(
                        np.searchsorted(np.cumsum(pooled_lengths) / pooled_lengths.sum(), 0.99) + 1
                    ),
                    "tracks_under_10_frames": int((pooled_lengths <= 10).sum()),
                    "frames_covered_by_top9": float(
                        pooled_lengths[: N_ANIMALS * len(args.experiments)].sum()
                        / pooled_lengths.sum()
                    ),
                },
                "encounters": {
                    **dict(pooled["encounters"]),
                    "preserved_rate": (
                        pooled["encounters"]["preserved"]
                        / (pooled["encounters"]["preserved"] + pooled["encounters"]["broken"])
                        if pooled["encounters"]["preserved"] + pooled["encounters"]["broken"]
                        else float("nan")
                    ),
                },
                "jumps": {
                    v: {
                        **dict(pooled["jumps"][v]),
                        "jump_rate": (
                            pooled["jumps"][v]["jumps"] / pooled["jumps"][v]["adjacent_pairs"]
                            if pooled["jumps"][v]["adjacent_pairs"]
                            else float("nan")
                        ),
                    }
                    for v in VIEWS
                },
                "per_experiment": per_experiment,
                "consistency": consistency,
            }

        # --- report ---------------------------------------------------------
        header = (
            f"{'method':<12} {'switch%':>9} {'scored':>11} {'tracks':>8} {'per animal':>11} "
            f"{'median len':>11} {'enc kept%':>10} {'enc n':>7} "
            f"{'jump% top':>10} {'side1':>8} {'side2':>8}"
        )
        print("=" * len(header))
        print(f"POOLED over {len(args.experiments)} experiments, n_groups = {len(groups)}")
        print("=" * len(header))
        print(header)
        for method in args.methods:
            r = results[method]
            print(
                f"{method:<12} {100 * r['switch']['switch_candidate_rate']:>9.4f} "
                f"{r['switch']['scored']:>11,} {r['fragmentation']['tracks']:>8,} "
                f"{r['fragmentation']['tracks_per_animal']:>11.1f} "
                f"{r['fragmentation']['median_track_frames']:>11.0f} "
                f"{100 * r['encounters']['preserved_rate']:>10.2f} "
                f"{r['encounters']['scored']:>7,} "
                + " ".join(
                    f"{100 * r['jumps'][v]['jump_rate']:>{w}.4f}"
                    for v, w in zip(VIEWS, (10, 8, 8), strict=True)
                )
            )

        print(
            "\nthree-view consistency (diagnostic) — pairing re-derived from a node the "
            "trackers do not link on"
        )
        for method in args.methods:
            for name in sorted(next(iter(results[method]["consistency"].values()))):
                agree = sum(
                    p[name]["agree"] for p in results[method]["consistency"].values()
                )
                resolved = sum(
                    p[name]["resolved"] for p in results[method]["consistency"].values()
                )
                print(
                    f"  {method:<11} {name:<28} resolved {resolved:>9,}  "
                    f"agree {100 * agree / resolved:6.2f}%"
                )
        print(
            "  A 3D tracker assigns one ID across all three cameras at once, so a\n"
            "  mouth-derived version of this would read ~100% by construction and mean\n"
            "  nothing. The pectoral is the honest check: it is never missing in these\n"
            "  predictions, where the tail is absent in 4.3-4.8% of complete frames and\n"
            "  a missing point makes some pairing cheapest for no geometric reason.\n"
            "  The per-view jump% above is the other half of the same check."
        )

        # --- the pre-registered decision -------------------------------------
        print("\n" + "=" * len(header))
        print("DECISION RULE — relative switch reduction >= 20%, p < 0.05, "
              "fragmentation not worse")
        print("=" * len(header))
        verdicts = {}
        for challenger in args.methods:
            for baseline in args.methods:
                if challenger == baseline:
                    continue
                outcome = judge(results[challenger], results[baseline])
                verdicts[f"{challenger} vs {baseline}"] = outcome
                print(
                    f"  {challenger:>11} vs {baseline:<11} "
                    f"{100 * outcome['challenger_switch_rate']:.4f}% vs "
                    f"{100 * outcome['baseline_switch_rate']:.4f}%  "
                    f"rel {100 * outcome['relative_reduction']:>+7.1f}%  p = {outcome['p']:.3g}  "
                    f"frag {outcome['challenger_tracks_per_animal']:.1f} vs "
                    f"{outcome['baseline_tracks_per_animal']:.1f}  "
                    f"-> {outcome['verdict']}"
                )
        winners = {
            c for pair, o in verdicts.items() if o["verdict"] == "challenger wins"
            for c in [pair.split(" vs ")[0]]
        }
        beaten = {
            pair.split(" vs ")[1]
            for pair, o in verdicts.items()
            if o["verdict"] == "challenger wins"
        }
        undefeated = sorted(winners - beaten)
        if not verdicts:
            selected = None
        elif not winners:
            selected = None
            print("\n  No method beats another under the rule: INDISTINGUISHABLE.")
        elif len(undefeated) == 1:
            selected = undefeated[0]
            print(f"\n  -> ADOPT {selected}")
        else:
            selected = None
            print(f"\n  No single undefeated method ({undefeated}); reporting, not ranking.")
        print(f"     n_groups = {len(groups)}. One trio of the corpus's 14 groups.")

        record.param("results", {m: {k: v for k, v in r.items() if k != "per_experiment"}
                                 for m, r in results.items()})
        record.param("verdicts", verdicts)
        record.param("selected", selected)
        out = out_dir / "c04_scores.json"
        out.write_text(
            json.dumps(
                {
                    "n_groups": len(groups),
                    "groups": groups,
                    "experiments": list(args.experiments),
                    "gate_px": gate_px,
                    "gate_mm": gate_mm,
                    "results": results,
                    "verdicts": verdicts,
                    "selected": selected,
                },
                indent=2,
                default=float,
            ),
            encoding="utf-8",
        )
        record.output("scores", out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
