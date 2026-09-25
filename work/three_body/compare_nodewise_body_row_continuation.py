"""Three saved continuous paths with the same81340 state and83000 terminal horizon.

Stored direct support and raw-slot motion use the frozen comparison definitions.
No fitting, source reassignment, identity alignment or physical-swap scoring.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import compare_body_row_continuation as old
import numpy as np

common, proof, require, WORK = old.common, old.proof, old.require, old.WORK
FIRST, START, STOP, FRAMES = old.FIRST, old.START, old.STOP, old.FRAMES
STAGE = "nodewise_body_row_ten_second_comparison_v1"
NEW_STAGE = "nodewise_body_row_ten_second_continuation_v1"
WHOLE = WORK / "whole_row_body_ten_second_continuation_v1"
WHOLE_SHA = "562acdab952d46cd915ac15ffc16d0e26ed14b2aeba1096cd022f98ae04c9b96"
OLD_SHA = "474640d350944dd7b5969a76526b2a183d1573dd7a8fdf2ec24940a6449b7cfa"
ARMS = ("control", "whole_row", "nodewise_body_row")
SAMPLE_FIELDS = ("frame_idx", "pose_xyz", "observed", "sources", "proposal_kind")


def validate_variant(summary, plan, imported, control):
    """The frozen guard, with explicit terminal and original39-signature context."""
    require(
        summary["process_interval"] == [START, STOP]
        and summary["retained_interval"] == [FIRST, STOP],
        "variant interval differs",
    )
    require(
        summary["initialize_count"] == 1
        and summary["restore_count"] == 7
        and summary["external_pose_resets"] == summary["new_begin_calls"] == 0
        and summary["finish_calls"] == 1,
        "continuous lifecycle differs",
    )
    require(plan["original_binding"] == control["original_binding"], "different inputs")
    require(
        plan["parent_tracking_sha256"]
        == summary["parent_tracking_sha256"]
        == control["parent_tracking_sha256"]
        and plan["parent_checkpoint_sha256"]
        == summary["parent_checkpoint_sha256"]
        == old.PARENT_CHECKPOINT_SHA,
        "different inherited Sparse state",
    )
    require(
        imported["status"] == "ok"
        and imported["previous_frame"] == START - 1
        and imported["processed_frames"] == START - FIRST
        and imported["effective_from_native"] == START
        and imported["added_restore_count"] == 1
        and imported["new_begin_calls"] == 0
        and imported["all_arrays_byte_preserved"] is True
        and imported["original_metadata_unchanged"] is True,
        "unverified state migration",
    )
    require(
        imported["original_metadata"]["restore_count"] == 6 and len(imported["arrays"]) == 39,
        "original Sparse state evidence differs",
    )


def compare_arrays(candidates):
    require(tuple(candidates) == ARMS, "three fixed method axes")
    reference = candidates[ARMS[0]]["frame_idx"]
    require(
        all(np.array_equal(c["frame_idx"], reference) for c in candidates.values()),
        "same native frames required",
    )
    samples, metrics, deltas = {}, {}, {}
    for arm, candidate in candidates.items():
        metrics[arm] = old.summarize(candidate)
        keep = np.isin(candidate["frame_idx"], FRAMES)
        require(np.array_equal(candidate["frame_idx"][keep], FRAMES), "four exact native probes")
        for key in SAMPLE_FIELDS:
            samples[arm + "_" + key] = candidate[key][keep].copy()
    selected = (reference >= START) & (reference < STOP)
    for a, b in combinations(ARMS, 2):
        delta = np.linalg.norm(
            candidates[b]["pose_xyz"][selected] - candidates[a]["pose_xyz"][selected], axis=-1
        )
        deltas[a + "_vs_" + b] = dict(
            raw_slot_pose_delta_mm=dict(median=float(np.median(delta)), maximum=float(delta.max())),
            chosen_prefix_pose_differences=int(
                np.any(
                    candidates[a]["pose_xyz"][~selected] != candidates[b]["pose_xyz"][~selected],
                    axis=(1, 2, 3),
                ).sum()
            ),
        )
    return metrics, samples, deltas


def validate_samples(samples):
    require(set(samples) == {a + "_" + k for a in ARMS for k in SAMPLE_FIELDS}, "15 sample fields")
    for arm in ARMS:
        require(np.array_equal(samples[arm + "_frame_idx"], FRAMES), "sample native order")
        require(
            samples[arm + "_pose_xyz"].shape == (4, 3, 3, 3)
            and np.isfinite(samples[arm + "_pose_xyz"]).all()
            and samples[arm + "_observed"].shape == (4, 3, 3)
            and samples[arm + "_observed"].dtype == np.bool_
            and samples[arm + "_sources"].shape == (4, 3, 3, 3)
            and samples[arm + "_proposal_kind"].shape == (4,),
            "sample field axes",
        )


def run(variant, tracking_sha, output):
    variant, output = Path(variant).resolve(), Path(output).resolve()
    require(
        variant.is_relative_to(WORK) and output.is_relative_to(WORK) and not output.exists(),
        "workspace variant and fresh output required",
    )
    out, evidence, started = proof.Output(output), proof.Evidence(), proof.utc()
    try:
        evidence.add(Path(old.__file__), OLD_SHA)
        evidence.add(Path(__file__))
        # Fail on an absent/unsuccessful new producer before opening old archives.
        new = old.load_candidate(
            variant, "tracking.provenance.json", tracking_sha, NEW_STAGE, evidence
        )
        control = old.load_candidate(
            old.CONTROL, "control.provenance.json", old.CONTROL_SHA, old.CONTROL.name, evidence
        )
        whole = old.load_candidate(
            WHOLE, "tracking.provenance.json", WHOLE_SHA, WHOLE.name, evidence
        )
        loaded = dict(zip(ARMS, (control, whole, new), strict=True))
        plans, imports, references = {}, {}, {}
        for arm, root, seal_name, digest in (
            ("control", old.CONTROL, "control.provenance.json", old.CONTROL_SHA),
            ("whole_row", WHOLE, "tracking.provenance.json", WHOLE_SHA),
            ("nodewise_body_row", variant, "tracking.provenance.json", tracking_sha),
        ):
            _, summary, seal = loaded[arm]
            references[arm] = dict(
                root=str(root),
                seal_name=seal_name,
                seal_sha256=digest,
                candidate=seal["outputs"]["candidate.npz"],
                candidate_arrays=summary["candidate_arrays"],
            )
            if arm != "control":
                plans[arm] = proof.read_json(common.bound_member(root, seal, "plan.json", evidence))
                imports[arm] = proof.read_json(
                    common.bound_member(root, seal, "state_import.json", evidence)
                )
                validate_variant(summary, plans[arm], imports[arm], control[1])
        require(
            imports["whole_row"]["arrays"] == imports["nodewise_body_row"]["arrays"]
            and imports["whole_row"]["original_metadata"]
            == imports["nodewise_body_row"]["original_metadata"],
            "different original39 state signatures/metadata",
        )
        metrics, samples, deltas = compare_arrays({arm: loaded[arm][0] for arm in ARMS})
        validate_samples(samples)
        out.write("review_samples.npz", samples)
        out.write("review_samples_signatures.json", common.signature(samples))
        out.write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                methods=metrics,
                arms=list(ARMS),
                process_interval=[START, STOP],
                terminal_horizon=STOP,
                frames=list(FRAMES),
                pairwise_raw_slot_differences=deltas,
                candidate_references=references,
                chosen_prefix_is_not_the_retained_beam_history=True,
                raw_slots_independently_identity_aligned=False,
                physical_ID_truth=False,
                independent_holdout=False,
                automatic_adoption=False,
                original_video_or_masks_read=False,
                tracking_or_fitting_performed=False,
            ),
        )
        out.write(
            "plan.json",
            dict(
                schema=STAGE,
                source_sha256={
                    str(Path(__file__).resolve()): evidence.paths[str(Path(__file__).resolve())],
                    str(Path(old.__file__).resolve()): OLD_SHA,
                },
                original_binding=control[1]["original_binding"],
                frames=list(FRAMES),
                parent_tracking_sha256=control[1]["parent_tracking_sha256"],
                parent_checkpoint_sha256=old.PARENT_CHECKPOINT_SHA,
                candidate_references=references,
                all_three_paths_terminal_horizon=STOP,
                same_original39_signatures_and_metadata=True,
                tracker_context=plans["nodewise_body_row"]["tracker_context"],
                support_interpretation="Stored direct support, not independent detection accuracy.",
            ),
        )
        evidence.final_guard()
        out.final_guard()
        receipt = dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=out.records,
        )
    except BaseException as exc:
        receipt = dict(
            stage=STAGE,
            status="failed",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=out.records,
            error=repr(exc),
        )
        (output / "comparison.provenance.json").write_bytes(proof.json_bytes(receipt))
        raise
    (output / "comparison.provenance.json").write_bytes(proof.json_bytes(receipt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", type=Path, required=True)
    parser.add_argument("--tracking-sha", required=True)
    parser.add_argument("--output", type=Path, default=WORK / STAGE)
    args = parser.parse_args()
    run(args.variant, args.tracking_sha, args.output)
