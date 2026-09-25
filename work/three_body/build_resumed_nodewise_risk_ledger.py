"""Apply preregistered multi-cue review selection to frozen five-minute results."""

from __future__ import annotations

import csv
import itertools
from collections import Counter
from pathlib import Path

import numpy as np
from sample_resumed_nodewise_appearance import OUT, PLAN_SHA, ROOT, WORK, read, sha, write

from medaka3d.provenance import provenance

START, STOP, FPS = 66400, 116200, 166
BEAM = WORK / "resumed_nodewise_five_minute_identity_beam256_v1"
SPARSE = WORK / "sparse_instance_five_minute_continuation_v1"


def runs(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    return list(
        zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist(), strict=True)
    )


def candidates(root):
    with np.load(root / "candidate.npz", allow_pickle=False) as z:
        frames = z["frame_idx"]
        keep = (frames >= START) & (frames < STOP)
        assert np.array_equal(frames[keep], np.arange(START, STOP))
        return {
            k: z[k][keep]
            for k in ("frame_idx", "pose_xyz", "observed", "score_margin", "identity_review")
        }


def scene_windows(events, plan):
    context = plan["high_risk_event_context_frames"]
    windows = sorted(
        (max(START, e["start_frame"] - context), min(STOP, e["stop_frame_exclusive"] + context))
        for e in events
        if e["high_risk"]
    )
    merged = []
    for lo, hi in windows:
        if merged and lo <= merged[-1][1] + plan["merge_gap_frames"]:
            merged[-1][1] = max(hi, merged[-1][1])
        else:
            merged.append([lo, hi])
    result = []
    for lo, hi in merged:
        while hi - lo > plan["maximum_scene_frames"]:
            end = lo + plan["maximum_scene_frames"]
            result.append((lo, end))
            lo = end - plan["long_scene_split_overlap_frames"]
        result.append((lo, hi))
    return result


def run():
    assert sha(OUT / "selection_plan.json") == PLAN_SHA
    assert not (OUT / "scenes.json").exists(), "fresh risk outputs required"
    plan = read(OUT / "selection_plan.json")
    with provenance("resumed_nodewise_risk_ledger_v1", OUT, name="risk") as rec:
        bindings = {}

        def bind(name, path, expected=None):
            digest = sha(path)
            if expected is not None and digest != expected:
                raise ValueError(f"SHA mismatch: {path}")
            bindings[str(path)] = digest
            rec.input(name, path, sha256=digest)
            return path

        bind("source", Path(__file__))
        bind("helper", WORK / "sample_resumed_nodewise_appearance.py")
        bind("plan", OUT / "selection_plan.json", PLAN_SHA)
        for name, root in (("nodewise", ROOT), ("sparse", SPARSE)):
            seal = read(bind(name + "_seal", root / "tracking.provenance.json"))
            assert seal["status"] == "ok" and seal["stage"] == root.name
            bind(name, root / "candidate.npz", seal["outputs"]["candidate.npz"]["sha256"])
        beam_seals = [p for p in BEAM.glob("*.provenance.json") if read(p).get("status") == "ok"]
        assert len(beam_seals) == 1, "one successful beam seal required"
        seal = read(bind("beam_seal", beam_seals[0]))
        assert seal["stage"] == BEAM.name
        beam_summary = read(
            bind("beam_summary", BEAM / "summary.json", seal["outputs"]["summary.json"]["sha256"])
        )
        reference = beam_summary["raw_candidate_reference"]["candidate"]
        assert Path(reference["path"]).resolve() == (ROOT / "candidate.npz").resolve()
        assert reference["sha256"] == bindings[str(ROOT / "candidate.npz")]
        mapping_path = BEAM / "identity_mapping_and_ambiguity.npz"
        bind("mapping", mapping_path, seal["outputs"][mapping_path.name]["sha256"])
        appearance_seal = read(bind("appearance_seal", OUT / "appearance.provenance.json"))
        assert appearance_seal["status"] == "ok"
        assert appearance_seal["stage"] == "resumed_nodewise_appearance_samples_v1"
        assert appearance_seal["inputs"]["plan"]["sha256"] == PLAN_SHA
        assert (
            appearance_seal["inputs"]["candidate.npz"]["sha256"]
            == bindings[str(ROOT / "candidate.npz")]
        )
        assert appearance_seal["params"]["configuration"] == plan["appearance"]
        bind(
            "appearance",
            OUT / "appearance_samples.npz",
            appearance_seal["outputs"]["appearance_samples.npz"]["sha256"],
        )
        raw, sparse = candidates(ROOT), candidates(SPARSE)
        with np.load(mapping_path, allow_pickle=False) as z:
            assert np.array_equal(z["frame_idx"], raw["frame_idx"])
            mapping = z["permutation_by_id"]
            gaps = z["conditional_identity_alternative_gap"]
            statuses = z["conditional_identity_alternative_status"]
        assert np.array_equal(np.sort(mapping, axis=1), np.tile(np.arange(3), (STOP - START, 1)))
        assert np.array_equal(mapping[0], [0, 1, 2])
        events = []

        def add(family, lo, hi, high, **detail):
            events.append(
                dict(
                    event_id=f"event_{len(events) + 1:05d}",
                    family=family,
                    start_frame=int(lo),
                    stop_frame_exclusive=int(hi),
                    high_risk=bool(high),
                    **detail,
                )
            )

        close_by_method = {}
        for method, data in (("nodewise", raw), ("sparse", sparse)):
            chest = data["pose_xyz"][:, :, 1]
            distances = np.linalg.norm(chest[:, :, None] - chest[:, None, :], axis=-1)
            distances[:, np.arange(3), np.arange(3)] = np.inf
            close = distances.min(axis=(1, 2)) < plan["close_chest_distance_mm"]
            close_by_method[method] = close
            for a, b in itertools.combinations(range(3), 2):
                for lo, hi in runs(distances[:, a, b] < plan["close_chest_distance_mm"]):
                    add(
                        "close_3d_approach",
                        START + lo,
                        START + hi,
                        False,
                        method=method,
                        raw_slots=[a, b],
                        minimum_distance_mm=float(distances[lo:hi, a, b].min()),
                    )
            pose, obs = data["pose_xyz"], data["observed"]
            displacement = np.linalg.norm(np.diff(pose, axis=0), axis=-1)
            direct = obs[:-1] & obs[1:]
            for ti, fish, node in np.argwhere(
                (displacement > plan["direct_adjacent_displacement_mm"]) & direct
            ):
                add(
                    "direct_motion_jump",
                    START + ti,
                    START + ti + 2,
                    True,
                    method=method,
                    raw_slot=int(fish),
                    node=int(node),
                    displacement_mm=float(displacement[ti, fish, node]),
                )
            acceleration = np.linalg.norm(np.diff(pose, n=2, axis=0), axis=-1)
            direct3 = obs[:-2] & obs[1:-1] & obs[2:]
            for ti, fish, node in np.argwhere(
                (acceleration > plan["direct_adjacent_acceleration_displacement_mm"]) & direct3
            ):
                add(
                    "direct_acceleration_jump",
                    START + ti,
                    START + ti + 3,
                    True,
                    method=method,
                    raw_slot=int(fish),
                    node=int(node),
                    second_difference_mm=float(acceleration[ti, fish, node]),
                )
            for fish in range(3):
                for node in (0, 1):
                    for lo, hi in runs(~obs[:, fish, node]):
                        if hi - lo < plan["support_gap_frames"]:
                            continue
                        is_high = close[lo:hi].any()
                        add(
                            "mouth_chest_support_gap",
                            START + lo,
                            START + hi,
                            False,
                            method=method,
                            raw_slot=fish,
                            node=node,
                            close_during_gap=bool(is_high),
                        )
                        if is_high:
                            add(
                                "support_gap_entry",
                                START + lo,
                                START + lo + 1,
                                True,
                                method=method,
                                raw_slot=fish,
                                node=node,
                            )
                            if hi < STOP - START:
                                add(
                                    "support_gap_recovery",
                                    START + hi,
                                    START + hi + 1,
                                    True,
                                    method=method,
                                    raw_slot=fish,
                                    node=node,
                                )
            finite_low = np.isfinite(data["score_margin"]) & (
                data["score_margin"] <= plan["low_raw_margin"]
            )
            for lo, hi in runs(finite_low & close):
                add("low_raw_margin_with_close", START + lo, START + hi, True, method=method)

        changed = np.r_[False, np.any(np.diff(mapping, axis=0) != 0, axis=1)]
        for ti in np.flatnonzero(changed):
            add(
                "raw_beam_permutation_transition",
                START + ti,
                START + ti + 1,
                True,
                before=mapping[ti - 1].tolist(),
                after=mapping[ti].tolist(),
            )
        disagreement = np.any(mapping != np.arange(3), axis=1)
        unaccepted_spans = []
        for lo, hi in runs(disagreement):
            span = dict(start_frame=START + lo, stop_frame_exclusive=START + hi, accepted=False)
            unaccepted_spans.append(span)
            add("raw_beam_disagreement_span", START + lo, START + hi, False, accepted=False)
        low_beam = np.any(np.isfinite(gaps) & (gaps <= plan["low_beam_identity_gap"]), axis=1)
        low_beam |= np.any(
            (statuses != "finite_gap") & (statuses != "fixed_initial_constraint"), axis=1
        )
        for lo, hi in runs(low_beam & close_by_method["nodewise"]):
            add("low_conditional_beam_gap_with_close", START + lo, START + hi, True)

        perms = np.array(list(itertools.permutations(range(3))))
        costs = np.stack(
            [
                np.sqrt(
                    np.mean(
                        np.sum((raw["pose_xyz"] - sparse["pose_xyz"][:, p]) ** 2, axis=3),
                        axis=2,
                    )
                ).mean(axis=1)
                for p in perms
            ],
            axis=1,
        )
        order = np.argsort(costs, axis=1)
        best = order[:, 0]
        first_cost = np.take_along_axis(costs, order[:, :1], axis=1)[:, 0]
        second_cost = np.take_along_axis(costs, order[:, 1:2], axis=1)[:, 0]
        confident = (first_cost <= plan["method_match_max_mm"]) & (
            second_cost - first_cost >= plan["method_match_margin_mm"]
        )
        stable = []
        for pi in range(6):
            stable.extend(
                (lo, hi, pi)
                for lo, hi in runs((best == pi) & confident)
                if hi - lo >= plan["method_stable_frames"]
            )
        stable.sort()
        for previous, current in itertools.pairwise(stable):
            if previous[2] != current[2]:
                # Review the entire uncertain bridge, preserving endpoint context.
                add(
                    "method_correspondence_transition",
                    START + previous[1] - 1,
                    START + current[0] + 1,
                    True,
                    before=perms[previous[2]].tolist(),
                    after=perms[current[2]].tolist(),
                )
        for lo, hi in runs(~confident):
            add("method_correspondence_uncertain", START + lo, START + hi, False)

        with np.load(OUT / "appearance_samples.npz", allow_pickle=False) as z:
            appearance_frames = z["frame_idx"]
            discontinuity = z["discontinuity"]
        assert np.array_equal(
            appearance_frames,
            np.arange(START, STOP, plan["appearance"]["sample_stride_frames"]),
        )
        assert discontinuity.shape == (len(appearance_frames) - 1, 3)
        for ti, fish in np.argwhere(discontinuity):
            lo, hi = int(appearance_frames[ti]), int(appearance_frames[ti + 1]) + 1
            high = close_by_method["nodewise"][lo - START : hi - START].any()
            add("two_view_appearance_discontinuity", lo, hi, high, raw_slot=int(fish))

        windows = scene_windows(events, plan)
        scenes = []
        for lo, hi in windows:
            overlapping = [
                e
                for e in events
                if e["high_risk"] and e["start_frame"] < hi and e["stop_frame_exclusive"] > lo
            ]
            scenes.append(
                dict(
                    start_frame=lo,
                    stop_frame_exclusive=hi,
                    anchor_frames=[lo, (lo + hi - 1) // 2, hi - 1],
                    selection_group="high_risk",
                    event_ids=[e["event_id"] for e in overlapping],
                    risk_families=sorted({e["family"] for e in overlapping}),
                )
            )
        low = plan["random_low_risk"]
        pool = []
        for lo in range(START, STOP - low["scene_frames"] + 1, low["scene_frames"]):
            hi = lo + low["scene_frames"]
            if (
                not any(a < hi and b > lo for a, b in windows)
                and not disagreement[lo - START : hi - START].any()
            ):
                pool.append((lo, hi))
        assert len(pool) >= low["count"], (
            "insufficient predefined low-risk pool; no silent substitute"
        )
        chosen = np.random.default_rng(low["seed"]).choice(len(pool), low["count"], replace=False)
        for i in chosen:
            lo, hi = pool[int(i)]
            scenes.append(
                dict(
                    start_frame=lo,
                    stop_frame_exclusive=hi,
                    anchor_frames=[lo, (lo + hi - 1) // 2, hi - 1],
                    selection_group="random_low_risk",
                    event_ids=[],
                    risk_families=[],
                )
            )
        scenes.sort(key=lambda row: row["start_frame"])
        for i, scene in enumerate(scenes):
            scene["scene_id"] = f"scene_{i + 1:03d}"
        high_mask = np.zeros(STOP - START, bool)
        for lo, hi in windows:
            high_mask[lo - START : hi - START] = True
        for e in events:
            if e["high_risk"]:
                assert high_mask[
                    e["start_frame"] - START : e["stop_frame_exclusive"] - START
                ].all(), "high-risk coverage gap"
        summary = dict(
            schema="resumed_nodewise_risk_ledger_v1",
            status="ok",
            interval=[START, STOP],
            fps=FPS,
            selection_plan_sha256=PLAN_SHA,
            event_count=len(events),
            event_counts=dict(Counter(e["family"] for e in events)),
            high_risk_event_counts=dict(Counter(e["family"] for e in events if e["high_risk"])),
            high_risk_scene_count=len(windows),
            random_low_risk_scene_count=low["count"],
            high_risk_unique_seconds=float(high_mask.sum() / FPS),
            total_clip_seconds=sum(
                (s["stop_frame_exclusive"] - s["start_frame"]) / FPS for s in scenes
            ),
            all_high_risk_events_covered=True,
            raw_beam_disagreement_frames=int(disagreement.sum()),
            raw_beam_disagreement_spans=unaccepted_spans,
            beam_status_counts=dict(Counter(map(str, statuses.ravel()))),
            raw_nonfinite_score_margin_fraction={
                n: float((~np.isfinite(d["score_margin"])).mean())
                for n, d in (("nodewise", raw), ("sparse", sparse))
            },
            human_review_completed=False,
            new_review_unscored_fraction=1.0,
            known_male_identity_truth=False,
            candidate_or_review_modified=False,
            limitations=[
                "A risk queue is not an identity score. "
                "All three-fish review edges remain unscored.",
                "Method correspondence is a geometric cue; a change can reflect geometry failure.",
                "Conditional beam gaps are not probabilities or global guarantees.",
                "Appearance proxy is sampled, lighting-sensitive and may lack support.",
                "Persistent raw/beam disagreements remain explicitly unaccepted, "
                "even outside transition clips.",
            ],
        )
        write(OUT / "risk_ledger.json", dict(**summary, events=events))
        write(
            OUT / "scenes.json",
            dict(
                schema="resumed_nodewise_blinded_scene_selection_v1",
                interval=[START, STOP],
                fps=FPS,
                selection_plan_sha256=PLAN_SHA,
                scenes=scenes,
            ),
        )
        write(
            OUT / "random_low_risk_registration.json",
            dict(**low, eligible_pool=pool, selected_pool_indices=chosen.tolist()),
        )
        write(OUT / "risk_summary.json", summary)
        with (OUT / "scenes.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "scene_id",
                    "start_frame",
                    "stop_frame_exclusive",
                    "selection_group",
                    "risk_families",
                ],
            )
            writer.writeheader()
            writer.writerows(
                {k: (";".join(s[k]) if k == "risk_families" else s[k]) for k in writer.fieldnames}
                for s in scenes
            )
        for path, digest in bindings.items():
            assert sha(path) == digest, f"input changed: {path}"
        for name in (
            "risk_ledger.json",
            "scenes.json",
            "random_low_risk_registration.json",
            "risk_summary.json",
            "scenes.csv",
        ):
            rec.output(name, OUT / name, sha256=sha(OUT / name))
        print(
            {
                k: summary[k]
                for k in (
                    "high_risk_scene_count",
                    "random_low_risk_scene_count",
                    "total_clip_seconds",
                    "raw_beam_disagreement_frames",
                    "high_risk_event_counts",
                )
            },
            flush=True,
        )


if __name__ == "__main__":
    run()
