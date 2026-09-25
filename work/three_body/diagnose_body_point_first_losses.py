"""Saved-only first three Original-relative loss times, after independent seven-arm evaluation.

Read three candidate21 files to fix times, then at most three frame trees per
arm. Recompute frozen support/selection arithmetic only; no raw image decode,
tracker initialization, rendering, fit or model. Pool alternatives are anonymous
slot observations under their own parents/plans, never biological-ID truth.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import body_point_transport as points
import chest_candidate_diagnostics as selection
import evaluate_body_point200 as evaluation
import numpy as np
import render_body_point200 as contracts

producer, previous = evaluation.producer, evaluation.previous
base, prepared, inputs = evaluation.base, evaluation.prepared, evaluation.inputs
WORK, REPO, FIRST, STOP = evaluation.WORK, evaluation.REPO, evaluation.FIRST, evaluation.STOP
ARMS = ("original", "ct_gap", producer.ARM)
ROOT = WORK / "body_point200_tracking_v1"
TRACKING_SHA = "f5cc08f200a6eb7d3029110f876ce449d9875edbe9ea8e42217b00efd08c63df"
STAGE = "body_point200_first_losses_diagnosis_v1"
require = base.require
PINS = {
    **evaluation.SOURCE_PINS,
    **producer.SOURCE_PINS,
    "work/three_body/run_body_point200.py": evaluation.PRODUCER_SHA,
    "work/three_body/evaluate_body_point200.py": contracts.EVALUATOR_SHA,
    "work/three_body/chest_candidate_diagnostics.py": (
        "a2a8bdc8196328290b8bb8a4f1549d0e43da73c81dde9e872eda691d32ae8f6d"
    ),
    "work/three_body/render_body_point200.py": (
        "edfda2cb48e225e5762bf5c8a6b343b92ad26d577fa45ff3c1b38c6b638f98b7"
    ),
}


class Consumed:
    """Existing seal/reader primitives, hashing only consumed payloads."""

    def __init__(self, evidence):
        self.evidence = evidence

    def seal(self, root, name, digest, stage):
        path = root / name
        self.evidence.add(path, digest)
        seal = base.read_json(path)
        require(seal["status"] == "ok" and seal["stage"] == stage, "successful stage required")
        return seal, previous.member_index(seal, root)

    def file(self, root, members, relative):
        path = prepared.bound_path(root, members, relative)
        self.evidence.add(path, members[str(path)]["sha256"])
        return path

    def json(self, root, members, relative):
        self.file(root, members, relative)
        return base.bound_json(root, members, relative)

    def arrays(self, root, members, relative):
        self.file(root, members, relative)
        return previous.read_arrays(root, members, relative)

    def tree(self, root, members, prefix):
        for suffix in (".json", ".npz"):
            self.file(root, members, prefix + suffix)
        return evaluation.old_evaluation.load_tree(root, members, prefix)


def first_loss_times(candidates):
    for candidate in candidates.values():
        previous.same(candidate["frame_idx"], np.arange(FIRST, STOP, dtype=np.int64), "native200")
        require(
            candidate["observed"].shape == (200, 3, 3) and candidate["observed"].dtype == bool,
            "all200 direct support cells required",
        )
    lost = candidates["original"]["observed"] & ~candidates[producer.ARM]["observed"]
    return np.flatnonzero(lost.any(axis=(1, 2)))[:3], lost


def retention(fit, beam_size):
    selected, order = selection.selected_indices(
        fit["pose"], fit["path_cost"], fit["parent"], fit["permutation"], beam_size
    )
    previous.same(selected, fit["selected"], "saved stable beam selection")
    previous.same(order, fit["order"], "saved full cost order")
    reasons = np.full(len(order), "after_beam_capacity_not_examined", dtype="U48")
    duplicate = np.full(len(order), -1, np.int64)
    kept = []
    for index in order:
        if len(kept) == beam_size:
            break
        matches = [
            j
            for j in kept
            if fit["parent"][index] == fit["parent"][j]
            and selection.identical(fit["permutation"][index], fit["permutation"][j])
            and np.max(np.linalg.norm(fit["pose"][index] - fit["pose"][j], axis=-1)) < 0.5
        ]
        if matches:
            reasons[index], duplicate[index] = "same_parent_permutation_pose_dedup", matches[0]
        else:
            reasons[index] = "retained"
            kept.append(int(index))
    previous.same(np.asarray(kept, np.int64), selected, "selection classification")
    return reasons, duplicate


def trace_matches(candidate, offset, records):
    fields = set(previous.RECORD_FIELDS) - {"parent"}
    matched = []
    for rank, record in enumerate(records):
        equal = True
        for key in fields:
            a, b = np.asarray(candidate[key][offset]), np.asarray(record[key])
            equal &= (
                (a.shape == b.shape and np.array_equal(a, b))
                if a.dtype.kind == "U"
                else selection.identical(a, b)
            )
        if equal:
            matched.append(rank)
    require(bool(matched), "final same-horizon trace has no exact retained record")
    return np.asarray(matched, np.int64)


def pool_arrays(tree, candidate, offset, rays, rig, beam_size):
    fit = tree["fit"]
    require(fit["native"] == FIRST + offset, "fit native differs")
    selected, order = previous.tracking.check_retained_fit(tree["records"], fit, beam_size)
    reasons, duplicate = retention(fit, beam_size)
    matches = trace_matches(candidate, offset, tree["records"])
    post = selection.support_snapshot(fit["pose"], fit["proposal_plan"], rays, rig)
    pre = selection.support_snapshot(fit["proposal_pose"], fit["proposal_plan"], rays, rig)
    for rank, index in enumerate(selected):
        for key in ("sources", "observed"):
            previous.same(
                post["strict_" + key][index],
                np.asarray(tree["records"][rank][key]),
                "retained recomputed " + key,
            )
    diagnostics = fit["owner"]
    coefficient = diagnostics["coefficient"]
    gap = diagnostics.get("gap_loss", np.zeros_like(diagnostics["base_score"]))
    terms = np.stack(
        [
            diagnostics["base_score"],
            coefficient * diagnostics["owner_loss"],
            coefficient * gap,
            fit["parent_cost"],
        ],
        axis=1,
    )
    require(
        np.allclose(terms[:, :3].sum(1), fit["cost"], atol=1e-10, rtol=0), "local decomposition"
    )
    previous.same(fit["cost"] + fit["parent_cost"], fit["path_cost"], "saved parent cost addition")
    result = {
        k: fit[k].copy()
        for k in (
            "pose",
            "proposal_pose",
            "proposal_predicted",
            "proposal_plan",
            "parent",
            "permutation",
            "kind",
            "candidate_anchor",
            "cost",
            "path_cost",
            "parent_cost",
            "parts",
            "support",
        )
    }
    result.update(
        selected=selected,
        order=order,
        retention_reason=reasons,
        dedup_partner=duplicate,
        trace_matching_retained_ranks=matches,
        trace_fit_indices=selected[matches],
        base_owner_gap_parent=terms,
        best_iteration=diagnostics["best_iteration"],
        **{"pre_" + k: v for k, v in pre.items()},
        **{"post_" + k: v for k, v in post.items()},
    )
    return result


def alternatives(pool, body, node):
    supported = np.flatnonzero(pool["post_strict_observed"][:, body, node])
    references = np.unique(np.r_[pool["selected"][0], pool["trace_fit_indices"]])
    rows = []
    for j in supported:
        for ref in references:
            delta = pool["base_owner_gap_parent"][j] - pool["base_owner_gap_parent"][ref]
            path_delta = float(pool["path_cost"][j] - pool["path_cost"][ref])
            require(
                np.isclose(delta.sum(), path_delta, atol=1e-10, rtol=0),
                "paired score decomposition",
            )
            rows.append(
                dict(
                    candidate=int(j),
                    reference_candidate=int(ref),
                    reference_is_rank0=bool(ref == pool["selected"][0]),
                    reference_on_final_trace=bool(ref in pool["trace_fit_indices"]),
                    parent=int(pool["parent"][j]),
                    anchor=int(pool["candidate_anchor"][j]),
                    kind=str(pool["kind"][j]),
                    retention=str(pool["retention_reason"][j]),
                    dedup_partner=int(pool["dedup_partner"][j]),
                    delta_base_owner_gap_parent=delta.tolist(),
                    delta_path=path_delta,
                    alternative_base_lower=bool(delta[0] < 0),
                    alternative_local_lower=bool(pool["cost"][j] < pool["cost"][ref]),
                    alternative_parent_cost_lower=bool(delta[3] < 0),
                )
            )
    return dict(
        prefit_supported_indices=np.flatnonzero(
            pool["pre_strict_observed"][:, body, node]
        ).tolist(),
        fitted_supported_indices=supported.tolist(),
        retained_supported_indices=[int(j) for j in pool["selected"] if j in supported],
        comparisons=rows,
    )


def point_terms(packet):
    p = points.validate_packet(packet)
    quotient = np.full_like(p.old_mass, np.nan)
    np.divide(p.old_mass, p.single_count, out=quotient, where=p.single_count > 0)
    return {
        k: getattr(p, k).copy()
        for k in (
            "anchor_ids",
            "old_mass",
            "single_count",
            "valid_single_count",
            "effective_mass",
            "point_selected",
            "point_weight",
            "query_counts",
            "ray_valid",
            "native_inside",
        )
    } | dict(
        m_over_S=quotient,
        body_mask_row=np.asarray(
            [[a.body_mask_row for a in by_view] for by_view in p.whole_mask_packet.assignments]
        ),
        body_weight=np.asarray(
            [[a.body_weight for a in by_view] for by_view in p.whole_mask_packet.assignments]
        ),
    )


def run(evaluation_root, evaluation_sha, output_root):
    require(base.valid_sha(evaluation_sha), "completed independent evaluation SHA required")
    eroot, destination = Path(evaluation_root).resolve(), Path(output_root).resolve()
    require(
        eroot.is_relative_to(WORK)
        and destination.is_relative_to(WORK)
        and not destination.exists(),
        "workspace input and new output required",
    )
    evidence, out, captured = base.Evidence(), base.Output(destination), {}
    reader, started = Consumed(evidence), base.utc()
    try:
        evidence.add(Path(__file__).resolve())
        for name, digest in PINS.items():
            evidence.add(REPO / name, digest)
        eseal, emembers = reader.seal(
            eroot, "evaluation.provenance.json", evaluation_sha, evaluation.STAGE
        )
        eplan, esummary = (reader.json(eroot, emembers, k) for k in ("plan.json", "summary.json"))
        contracts.validate_evaluation(eplan, esummary, eseal, ROOT, TRACKING_SHA)
        references = {}
        for root, sha, stage, arms in (
            (producer.PREVIOUS, producer.PREVIOUS_SHA, previous.STAGE, ("original",)),
            (
                contracts.OLD_GAP,
                contracts.OLD_GAP_SHA,
                "mask_transport_gap200_tracking_v1",
                ("ct_gap",),
            ),
            (ROOT, TRACKING_SHA, producer.STAGE, (producer.ARM,)),
        ):
            evaluation.input_member(
                eseal, root / "tracking.provenance.json", sha, "evaluated producer"
            )
            _, members = reader.seal(root, "tracking.provenance.json", sha, stage)
            for arm in arms:
                row = next(r for r in eplan["candidate_references"] if r["arm"] == arm)
                path = prepared.bound_path(root, members, f"{arm}/candidate.npz", row["sha256"])
                require(str(path) == row["path"], "evaluated candidate path differs")
                evaluation.input_member(eseal, path, row["sha256"], "evaluated candidate")
                references[arm] = (root, members)
        # Gate above completes before any candidate or fit NPZ is decoded.
        candidates = {a: reader.arrays(*references[a], f"{a}/candidate.npz") for a in ARMS}
        offsets, lost = first_loss_times(candidates)
        out.write(
            "selection.npz",
            dict(
                frame_idx=np.arange(FIRST, STOP, dtype=np.int64),
                original_relative_lost=lost,
                selected_offsets=offsets,
            ),
        )
        pseal, pmembers = reader.seal(
            inputs.PREPARATION,
            "preparation.provenance.json",
            inputs.PREPARATION_SHA,
            "semantic_node200_preparation_v2",
        )
        evaluation.input_member(
            eseal,
            inputs.PREPARATION / "preparation.provenance.json",
            inputs.PREPARATION_SHA,
            "shared preparation",
        )
        pplan, context, natives = (
            reader.json(inputs.PREPARATION, pmembers, k)
            for k in ("plan.json", "original_context.json", "native_evidence.json")
        )
        calibration = {
            Path(p): sha
            for p, sha in pplan["shared_SL_calibration_sha256"].items()
            if Path(p).suffix == ".json"
        }
        require(
            len(calibration) == 4 and len({p.parent for p in calibration}) == 1, "same calibration4"
        )
        for path, sha in calibration.items():
            evaluation.input_member(pseal, path, sha, "prepared calibration")
            evaluation.input_member(eseal, path, sha, "evaluated calibration")
            evidence.add(path, sha)
        rig = inputs.load_rig(next(iter(calibration)).parent)
        require(
            context["cameras"]
            == json.loads(inputs._canonical([dataclasses.asdict(c) for c in rig.cameras]))
            and context["views"] == list(rig.views) == list(points.VIEWS),
            "same saved refractive rig",
        )
        observations = {(r["frame_idx"], r["view"]): r for r in natives}
        rows = []
        for offset in offsets:
            native = FIRST + int(offset)
            detections = {}
            for view in points.VIEWS:
                obs = observations[native, view]
                arrays = reader.arrays(inputs.PREPARATION, pmembers, obs["observation_path"])
                inputs.validate_observation(arrays, obs)
                detections[view] = arrays["detections"]
            rays = selection.RayObservations.from_points(rig, detections)
            pools = {}
            for arm in ARMS:
                tree = reader.tree(*references[arm], f"{arm}/frames/native_{native}")
                pool = pool_arrays(
                    tree,
                    candidates[arm],
                    int(offset),
                    rays,
                    rig,
                    context["tracker_config"]["beam_size"],
                )
                out.write(f"native_{native}/{arm}_pool.npz", pool)
                pools[arm] = pool
                if arm == producer.ARM:
                    out.write(f"native_{native}/point_mass.npz", point_terms(tree["fit"]["packet"]))
            for body, node in np.argwhere(lost[offset]):
                rows.append(
                    dict(
                        native=native,
                        body=int(body),
                        node=int(node),
                        candidate_support={
                            a: candidates[a]["observed"][offset].tolist() for a in ARMS
                        },
                        candidate_source_rows={
                            a: candidates[a]["sources"][offset].tolist() for a in ARMS
                        },
                        new_minus_original_pose_mm=(
                            candidates[producer.ARM]["pose_xyz"][offset]
                            - candidates["original"]["pose_xyz"][offset]
                        ).tolist(),
                        new_minus_ct_gap_pose_mm=(
                            candidates[producer.ARM]["pose_xyz"][offset]
                            - candidates["ct_gap"]["pose_xyz"][offset]
                        ).tolist(),
                        alternatives={
                            a: alternatives(pool, int(body), int(node)) for a, pool in pools.items()
                        },
                    )
                )
        result = dict(
            schema=STAGE,
            status="ok",
            native_interval=[FIRST, STOP],
            selected_native=(FIRST + offsets).tolist(),
            selected_native_limit=3,
            loss_cases=rows,
            fit_trees_read=3 * len(offsets),
            candidate_files_read=3,
            first_selection=(
                "first3 native with Original direct support lost "
                "in completed new same-horizon trace"
            ),
            score_columns=["base", "coefficient_owner", "coefficient_gap", "parent_path_cost"],
            limits=(
                "posthoc own-parent/plans; final trace may differ from current rank0; "
                "alternate slots not biological ID truth; no causal attribution to one loss term"
            ),
            raw_image_decodes=0,
            fit_calls=0,
            model_calls=0,
            GPU_used=False,
            input_verification=(
                "sealed header plus consumed payload SHA only; "
                "full independent evaluation already completed"
            ),
        )
        out.write("summary.json", result)
        prepared.snapshot_sources(evidence, out, captured)
        evidence.final_guard()
        out.final_guard()
        status = dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=base.utc(),
            inputs=evidence.paths,
            outputs=out.records,
            params=dict(all_guards_passed=True),
        )
    except BaseException as error:
        status = dict(
            stage=STAGE,
            status="failed",
            started_at=started,
            finished_at=base.utc(),
            inputs=evidence.paths,
            outputs=out.records,
            error_type=type(error).__name__,
            error=str(error),
        )
        with (destination / "diagnosis.provenance.json").open("xb") as f:
            f.write(base.json_bytes(status))
        raise
    with (destination / "diagnosis.provenance.json").open("xb") as f:
        f.write(base.json_bytes(status))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation-root", type=Path, required=True)
    p.add_argument("--evaluation-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--execute", action="store_true", required=True)
    a = p.parse_args()
    print(json.dumps(run(a.evaluation_root, a.evaluation_sha256, a.output)), flush=True)


if __name__ == "__main__":
    main()
