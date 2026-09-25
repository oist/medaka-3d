"""Disabled CPU evaluation of 57 existing female anchors on frozen five-minute results.

No fit, training, identity solve, video/mask/checkpoint read or new human labels.
Only two SLEAP mouth caches and five candidate fields are needed. Geometry
support is inherited from the successful comparison, not recomputed here.
"""

from __future__ import annotations

import argparse
import csv
import io
import itertools
from dataclasses import asdict
from pathlib import Path

import numpy as np
import run_fast_tail_five_minute_identity_beam as control

from medaka3d.eval import generative

proof, require = control.proof, control.require
WORK, REPO = control.WORK, control.REPO
START, STOP = 66400, 116200
STAGE = "female_57_five_minute_endpoint_evaluation_v1"
EXECUTION_READY = True
CONTROL_SHA = "9bf62d84cf2eff9cb2dc19ed2dd6beab901633b8221ca6491ab5f0b2bc3a55e1"
GENERATIVE_SHA = "a0e66679588053ba81de4f2333f00a2d906b768b8cfe1db5ba2a196101a82157"
GT_ROOT = WORK / "long_female_endpoint_160s_v1"
GT_SEAL_SHA = "e5894da372b2f1b577e8c0ac2d6930140231a6674db90594501dda58d1fba5b6"
GT_SHA = "ab0a173ac737a63311aec834c5832fc19475d4f7d1892b8193f8465bba67519f"
BEAM_ROOT = WORK / control.STAGE
BEAM_SEAL_SHA = "4f2c676bbaddace1aa5d4a0696767e5ecdd042787dbbb7208b20d3d70403d2db"
FRAMES = (
    66889,
    68186,
    69363,
    70234,
    70835,
    71288,
    72371,
    73027,
    73953,
    74786,
    75987,
    76698,
    77285,
    78083,
    78799,
    79547,
    80215,
    80657,
    81410,
    82121,
    83265,
    83853,
    84700,
    85625,
    86350,
    87520,
    88147,
    89709,
    90200,
    90687,
    91941,
    92899,
    94029,
    94891,
    95495,
    96132,
    96612,
    97802,
    98387,
    99156,
    99870,
    100343,
    101107,
    102328,
    103035,
    103409,
    104237,
    105304,
    106021,
    107124,
    107706,
    108343,
    109222,
    110803,
    111518,
    113682,
    115672,
)
FIELDS = ("frame_idx", "pose_xyz", "sources", "observed", "identity_review")
SIDES = ("side1_cam", "side2_cam")
PERMS = np.array(list(itertools.permutations(range(3))), dtype=np.int64)


def selected_records(data, expected=FRAMES):
    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
    selected = [
        r for r in rows if START <= int(r["frame"]) < STOP and r["gt_status"] == "confirmed"
    ]
    require(all(r["sex_label"] == "F" for r in selected), "confirmed non-female row")
    selected.sort(key=lambda r: int(r["frame"]))
    require(
        tuple(int(r["frame"]) for r in selected) == tuple(expected),
        "fixed female frame denominator differs or contains duplicate",
    )
    return selected


def match_records(records, mouth_at):
    """Use frozen matching unchanged; diagnostics do not select additional detections."""
    anchors = generative.match_mouth_anchors(records, mouth_at, START, STOP, match_px=5.0)
    diagnostics = []
    for row, anchor in zip(records, anchors, strict=True):
        require(int(row["frame"]) == anchor.frame, "annotation order differs")
        views = {}
        for view in SIDES:
            prefix = view.split("_")[0]
            reason, error = "not_selected", None
            if row[f"{prefix}_selection_status"] == "sleap_linked":
                try:
                    target = np.array(
                        [row[f"{prefix}_mouth_x"], row[f"{prefix}_mouth_y"]], dtype=float
                    )
                except (TypeError, ValueError):
                    target = np.full(2, np.nan)
                reason = "annotation_mouth_unavailable"
                if np.isfinite(target).all():
                    points = np.asarray(mouth_at(view, anchor.frame))
                    distance = np.linalg.norm(points - target, axis=-1)
                    distance = np.sort(np.where(np.isfinite(distance), distance, np.inf))
                    if not len(distance) or not np.isfinite(distance[0]):
                        reason = "no_finite_SL_mouth"
                    else:
                        error = float(distance[0])
                        reason = (
                            "outside_5px"
                            if error > 5.0
                            else (
                                "ambiguous_nearest_tie"
                                if len(distance) > 1 and np.isclose(distance[0], distance[1])
                                else "matched"
                            )
                        )
            require(
                (reason == "matched") == (view in anchor.detection_rows),
                "diagnostic and frozen matching disagree",
            )
            views[view] = dict(
                status=reason, nearest_error_px=error, SL_row=anchor.detection_rows.get(view)
            )
        diagnostics.append(dict(frame=anchor.frame, views=views))
    return anchors, diagnostics


def score_rows(rows, anchors):
    """Generic endpoint scorer reusable for later frozen -6/-12 trajectories."""
    require(set(rows) == set(FIELDS), "exact five row fields required")
    frames = np.array([a.frame for a in anchors], dtype=np.int64)
    n = len(frames)
    require(np.array_equal(rows["frame_idx"], frames), "requested anchor rows differ")
    require(
        rows["frame_idx"].dtype.kind in "iu"
        and rows["sources"].dtype.kind in "iu"
        and rows["pose_xyz"].dtype.kind == "f"
        and rows["observed"].dtype.kind == "b"
        and rows["identity_review"].dtype.kind == "b",
        "anchor dtypes differ",
    )
    for key, shape in (
        ("pose_xyz", (n, 3, 3, 3)),
        ("sources", (n, 3, 3, 3)),
        ("observed", (n, 3, 3)),
        ("identity_review", (n, 3)),
    ):
        require(rows[key].shape == shape, "anchor shape differs: " + key)
    before = {k: proof.signature(v) for k, v in rows.items()}
    raw = generative.score_female_identity(
        frames, rows["sources"][:, :, :, 0], rows["observed"][:, :, 0], anchors
    )
    accepted = generative.score_female_identity(
        frames,
        rows["sources"][:, :, :, 0],
        rows["observed"][:, :, 0],
        anchors,
        review=rows["identity_review"],
    )
    details = []
    for t, anchor in enumerate(anchors):
        slots, duplicate = set(), False
        for view, detection in anchor.detection_rows.items():
            matches = np.flatnonzero(
                rows["sources"][t, :, generative.VIEWS.index(view), 0] == detection
            )
            slots.update(map(int, matches))
            duplicate |= len(matches) > 1
        unique = next(iter(slots)) if len(slots) == 1 and not duplicate else None
        details.append(
            dict(
                frame=anchor.frame,
                matched_SL_views=len(anchor.detection_rows),
                source_matching_slots=sorted(slots),
                duplicated_source_row=duplicate,
                pre_support_unique_slot=unique,
                mouth_state_by_slot=[
                    "direct"
                    if rows["observed"][t, s, 0]
                    else (
                        "latent_finite"
                        if np.isfinite(rows["pose_xyz"][t, s, 0]).all()
                        else "nonfinite"
                    )
                    for s in range(3)
                ],
                original_review_by_slot=rows["identity_review"][t].tolist(),
                raw_result=raw["anchors"][t],
                with_original_review=accepted["anchors"][t],
            )
        )
    require(before == {k: proof.signature(v) for k, v in rows.items()}, "anchor rows mutated")
    return dict(raw=raw, with_original_review=accepted, endpoint_details=details)


def reorder_rows(rows, permutations):
    require(
        permutations.dtype.kind in "iu"
        and permutations.shape == (len(rows["frame_idx"]), 3)
        and np.array_equal(
            np.sort(permutations, axis=1), np.tile([0, 1, 2], (len(permutations), 1))
        ),
        "whole-fish permutation rows must be bijective",
    )
    return {
        k: (v.copy() if k == "frame_idx" else v[np.arange(len(v))[:, None], permutations].copy())
        for k, v in rows.items()
    }


def bound_json(root, seal, name, evidence):
    return control.document(root, seal, name, evidence)


def load_rows(reference, evidence):
    path = Path(reference["path"]).resolve()
    require(path.is_relative_to(WORK), "candidate outside workspace")
    evidence.add(path, reference["sha256"])
    selected = {}
    with np.load(path, allow_pickle=False) as z:
        require(
            len(z.files) == len(set(z.files)) == 21 and set(z.files) == set(reference["arrays"]),
            "candidate21 archive schema",
        )
        for key in FIELDS:
            value = z[key]
            require(proof.signature(value) == reference["arrays"][key], "field signature: " + key)
            if key == "frame_idx":
                require(
                    np.array_equal(value, np.arange(control.FIRST, STOP)),
                    "full retained native frame range differs",
                )
            selected[key] = value[np.asarray(FRAMES) - control.FIRST].copy()
    for value in selected.values():
        value.flags.writeable = False
    return selected


def load_mouths(shared, comparison_seal, annotation_seal, evidence):
    """Two side caches once; original within-frame row ordering is preserved."""
    pools = {}
    for view in SIDES:
        paths = [Path(p) for p in shared if Path(p).name == f"20251225_K_1_2_FCMC__{view}.npz"]
        require(len(paths) == 1, "one registered SLEAP cache per side")
        path = paths[0]
        control.edge(comparison_seal, path, shared[str(path)])
        require(
            annotation_seal["inputs"][str(path)]["sha256"] == shared[str(path)],
            "annotation matching used a different SLEAP version",
        )
        evidence.add(path, shared[str(path)])
        with np.load(path, allow_pickle=False) as z:
            require(
                str(z["experiment"]) == "20251225_K_1_2_FCMC"
                and str(z["view"]) == view
                and tuple(z["nodes"]) == ("mouth", "pectoral", "tail"),
                "SL context",
            )
            index, points = z["frame_idx"], z["points"]
            require(
                index.ndim == 1
                and index.dtype.kind in "iu"
                and np.all(np.diff(index) >= 0)
                and np.all((index >= 0) & (index < 348600))
                and points.shape == (len(index), 3, 2)
                and points.dtype.kind == "f"
                and not np.isinf(points).any(),
                "SL numeric schema",
            )
            for frame in FRAMES:
                lo, hi = np.searchsorted(index, [frame, frame + 1])
                pools[view, frame] = points[lo:hi, 0].copy()
                pools[view, frame].flags.writeable = False
    return pools


def validate_mapping(arrays, n=STOP - START, tolerance=0.0):
    require(np.array_equal(arrays["frame_idx"], np.arange(START, START + n)), "ID native gap")
    p, state = arrays["permutation_by_id"], arrays["path_state"]
    require(
        state.shape == (n,)
        and state.dtype.kind in "iu"
        and np.all((state >= 0) & (state < 6))
        and p.dtype.kind in "iu"
        and p.shape == (n, 3)
        and np.array_equal(p, PERMS[state])
        and p[0].tolist() == [0, 1, 2],
        "ID permutation",
    )
    status, gap = (
        arrays["conditional_identity_alternative_status"],
        arrays["conditional_identity_alternative_gap"],
    )
    require(
        status.shape == gap.shape == (n, 3)
        and gap.dtype.kind == "f"
        and not np.isnan(gap).any()
        and np.all(gap >= 0),
        "ID ambiguity shape/numeric",
    )
    require(
        np.all(status[0] == "fixed_initial_constraint")
        and np.isposinf(gap[0]).all()
        and np.array_equal(status[1:] == "not_retained_unknown", np.isposinf(gap[1:])),
        "conditional unknown or initial scope differs",
    )
    expected = np.full(gap.shape, "finite_gap", dtype="U32")
    expected[np.isfinite(gap) & (gap <= tolerance)] = "tie_within_tolerance"
    expected[np.isposinf(gap)] = "not_retained_unknown"
    expected[0] = "fixed_initial_constraint"
    require(np.array_equal(status, expected), "conditional status does not match retained gap")


def load_beam(reference, evidence):
    seal_path = BEAM_ROOT / "identity.provenance.json"
    evidence.add(seal_path, BEAM_SEAL_SHA)
    seal = proof.read_json(seal_path)
    require(seal["stage"] == control.STAGE and seal["status"] == "ok", "completed beam required")
    control.edge(seal, Path(control.__file__), CONTROL_SHA)
    control.edge(
        seal, control.COMPARISON_ROOT / "comparison.provenance.json", control.COMPARISON_SEAL_SHA
    )
    control.edge(seal, Path(reference["path"]), reference["sha256"])
    summary = bound_json(BEAM_ROOT, seal, "summary.json", evidence)
    plan = bound_json(BEAM_ROOT, seal, "plan.json", evidence)
    scalars = bound_json(BEAM_ROOT, seal, "beam_scalars.json", evidence)
    for doc in (plan, summary):
        require(
            doc["native_interval"] == [START, STOP]
            and doc["comparison_candidate_reference"] == reference
            and doc["comparison_seal_sha256"] == control.COMPARISON_SEAL_SHA
            and doc["milestone_summary_sha256"] == control.MILESTONE_SHA
            and doc["gaps_conditional_on_retained_graph"] is True
            and doc["physical_ID_truth"] is False
            and doc["global_optimality_certified"] is False,
            "beam binding/scope differs",
        )
    path = control.comparison.custom_member(
        seal, BEAM_ROOT, "identity_mapping_and_ambiguity.npz", evidence
    )
    keys = (
        "frame_idx",
        "path_state",
        "permutation_by_id",
        "conditional_identity_alternative_gap",
        "conditional_identity_alternative_status",
    )
    with np.load(path, allow_pickle=False) as z:
        require(
            len(z.files) == len(set(z.files)) == 14
            and set(z.files) == set(control.frozen.MAPPING_FIELDS),
            "beam mapping schema",
        )
        arrays = {k: z[k] for k in keys}
    validate_mapping(arrays, tolerance=scalars["numerical_tie_tolerance"])
    return {k: v[np.asarray(FRAMES) - START].copy() for k, v in arrays.items()}


def run(output):
    require(EXECUTION_READY, "source-only: ROOT quick review required before actual scoring")
    control.cpu_only()
    output = output.resolve()
    require(output.is_relative_to(WORK) and not output.exists(), "fresh workspace output required")
    out, evidence, started = proof.Output(output), proof.Evidence(), proof.utc()
    try:
        code = control.source_code()
        require(code[str(Path(control.__file__).resolve())] == CONTROL_SHA, "control source pin")
        require(
            code.get(str(Path(generative.__file__).resolve())) == GENERATIVE_SHA,
            "frozen female scorer changed",
        )
        require(str(Path(__file__).resolve()) in code, "new source absent from actual closure")
        for path, digest in code.items():
            evidence.add(path, digest)
        control.comparison_headers(control.COMPARISON_SEAL_SHA, evidence)
        comp_seal = proof.read_json(control.COMPARISON_ROOT / "comparison.provenance.json")
        control.edge(comp_seal, Path(generative.__file__), GENERATIVE_SHA)
        plan = bound_json(control.COMPARISON_ROOT, comp_seal, "plan.json", evidence)
        comparison_summary = bound_json(
            control.COMPARISON_ROOT, comp_seal, "summary.json", evidence
        )
        references = plan["candidate_references"]
        evidence.add(GT_ROOT / "evaluation.provenance.json", GT_SEAL_SHA)
        annotation_seal = proof.read_json(GT_ROOT / "evaluation.provenance.json")
        require(
            annotation_seal["status"] == "ok"
            and annotation_seal["stage"] == "long_female_endpoint_evaluation_160s",
            "registered annotation provenance",
        )
        gt = GT_ROOT / "original_female_sex_gt.csv"
        descriptor = annotation_seal["outputs"]["original_annotations"]
        require(
            Path(descriptor["path"]).resolve() == gt.resolve()
            and descriptor["sha256"] == GT_SHA
            and gt.stat().st_size == descriptor["size"],
            "original annotation copy binding",
        )
        evidence.add(gt, GT_SHA)
        data = gt.read_bytes()
        records = selected_records(data)
        out.write("original_female_sex_gt.csv", data)
        out.write(
            "fixed_female_anchor_set.json",
            dict(
                frames=list(FRAMES),
                CSV_sha256=GT_SHA,
                source_interval=[START, STOP],
                selection="confirmed F; keep all 57 even if a view or endpoint is unavailable",
                prior_first_minute=11,
                newly_scored_later_four_minutes=46,
                labels_created=False,
                later46_used_for_new_training_or_tuning=False,
                historical_SLEAP_training_independence_verified=False,
            ),
        )
        pools = load_mouths(
            plan["shared_SL_calibration_sha256"], comp_seal, annotation_seal, evidence
        )
        anchors, observations = match_records(records, lambda v, f: pools[v, f])
        rows = {arm: load_rows(ref, evidence) for arm, ref in references.items()}
        results = {arm: score_rows(value, anchors) for arm, value in rows.items()}
        for arm, value in rows.items():
            first = score_rows({k: v[:11] for k, v in value.items()}, anchors[:11])
            saved = comparison_summary["methods"][arm]["female_first_minute"]
            require(
                all(first[k] == saved[k] for k in ("raw", "with_original_review")),
                "same-horizon first11 control score differs",
            )
        mapping = load_beam(references["fast_tail"], evidence)
        reordered = reorder_rows(rows["fast_tail"], mapping["permutation_by_id"])
        results["fast_tail_beam_ID"] = score_rows(reordered, anchors)
        for t, item in enumerate(results["fast_tail_beam_ID"]["endpoint_details"]):
            identity = item["raw_result"]["identity"]
            item.update(
                raw_slot_by_ID=mapping["permutation_by_id"][t].tolist(),
                conditional_ID_alternative_status=mapping[
                    "conditional_identity_alternative_status"
                ][t].tolist(),
                conditional_ID_alternative_gap=control.explicit_json(
                    mapping["conditional_identity_alternative_gap"][t]
                ),
                same_as_initial_numeric_ID1=None if identity is None else identity == 1,
                initial_ID1_sex_certified=False,
            )
        out.write(
            "anchor_observations.json",
            dict(matches=[asdict(a) for a in anchors], diagnostics=observations),
        )
        out.write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                native_interval=[START, STOP],
                requested_anchors=57,
                requested_adjacent_intervals=56,
                results=results,
                candidate_references=references,
                comparison_seal_sha256=control.COMPARISON_SEAL_SHA,
                beam_seal_sha256=BEAM_SEAL_SHA,
                annotation_CSV_sha256=GT_SHA,
                geometry_support_inherited_from_comparison=True,
                support_gates=dict(ray_mm=2.0, native_pixel=12.0, minimum_views=2),
                female_endpoint_evaluation_only=True,
                whole_three_physical_ID_validation=False,
                male_ID_truth=False,
                dense_swap_zero_certified=False,
                reviews_cleared=False,
                geometry_modified=False,
                conditional_gaps_are_probabilities=False,
                original_first11_reused_development=True,
                later46_previously_scored_here=False,
                same_horizon_first11_control_exact=True,
                new_training_or_threshold_tuning=False,
                checkpoint39_reads=0,
                video_mask_reads=0,
                SLEAP_archives_decoded=2,
                candidate_archives_decoded=2,
                candidate_fields_decoded=5,
                identity_graph_reads=0,
                identity_solves=0,
                GPU_used=False,
                source_sha256=code,
            ),
        )
        control.cpu_only()
        require(control.source_code() == code, "source changed")
        evidence.final_guard()
        out.final_guard()
        status = dict(status="ok")
    except BaseException as error:
        status = dict(status="failed", error_type=type(error).__name__, error=str(error))
        out.write("failure.json", status)
        raise
    finally:
        with (output / "evaluation.provenance.json").open("xb") as stream:
            stream.write(
                proof.json_bytes(
                    dict(
                        stage=STAGE,
                        started_at=started,
                        finished_at=proof.utc(),
                        inputs=evidence.paths,
                        outputs=out.records,
                        **status,
                    )
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
