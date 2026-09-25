"""Compare the new whole-row rollout with the existing same-horizon Sparse path.

Stored support and row consistency are diagnostics, not independent identity
truth. Images at the four fixed development frames remain necessary to judge
whether a different fitted body actually follows one fish.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import compare_instance_rescore_10s as common
import numpy as np

proof, require, WORK = common.proof, common.require, common.WORK
FIRST, START, STOP = 49800, 81340, 83000
FRAMES = (82720, 82727, 82728, 82750)
STAGE = "body_row_ten_second_comparison_v1"
CONTROL = WORK / "body_row_same_horizon_control_v1"
CONTROL_SHA = "a2bdcb7bd4c16f519ecac5e00580564b27ce9cd0973820602624a9bbb8046bb1"
PARENT_CHECKPOINT_SHA = "aa93bfa25b34beba69e08600ed397340fcf8c1912d8a2ba116d12b648f13e54d"


def longest_true(values):
    """Longest contiguous true run, independently for each trailing coordinate."""
    values = np.asarray(values, bool)
    current = np.zeros(values.shape[1:], int)
    maximum = current.copy()
    for row in values:
        current = np.where(row, current + 1, 0)
        maximum = np.maximum(maximum, current)
    return maximum


def summarize(candidate):
    selected = (candidate["frame_idx"] >= START) & (candidate["frame_idx"] < STOP)
    frames = candidate["frame_idx"][selected]
    require(np.array_equal(frames, np.arange(START, STOP)), "complete comparison interval")
    pose, observed, sources = (candidate[k][selected] for k in ("pose_xyz", "observed", "sources"))
    require(pose.shape == (STOP - START, 3, 3, 3), "full three-body geometry")
    missing = ~observed
    steps = np.linalg.norm(np.diff(pose, axis=0), axis=-1)
    both = observed[1:] & observed[:-1]
    # Missing entries do not create a row conflict. A retained single row may
    # still be biologically incorrect; this is only source provenance.
    lo = np.min(np.where(sources >= 0, sources, np.iinfo(np.int32).max), axis=-1)
    hi = np.max(sources, axis=-1)
    mixed = (hi >= 0) & (lo != hi)
    kinds = np.asarray(candidate["proposal_kind"])[selected]
    new_kind = kinds == "body_row"
    require(not np.any(mixed[new_kind]), "selected whole-row proposal changed its rows")
    events = []
    for offset, fish, node in np.argwhere(both & (steps > 10.0)):
        events.append(
            dict(
                previous_frame=int(frames[offset]),
                frame_idx=int(frames[offset + 1]),
                raw_slot=int(fish),
                node=int(node),
                distance_mm=float(steps[offset, fish, node]),
                physical_fish_motion_confirmed=False,
            )
        )
    return dict(
        frames=len(frames),
        fish_frames=len(frames) * 3,
        node_order=["mouth", "chest", "tail"],
        directly_observed_by_slot_node=observed.sum(0).tolist(),
        missing_by_slot_node=missing.sum(0).tolist(),
        longest_missing_run_by_slot_node=longest_true(missing).tolist(),
        all_three_chests_direct_frames=int(observed[:, :, 1].all(-1).sum()),
        directly_supported_adjacent_pairs_by_slot_node=both.sum(0).tolist(),
        direct_adjacent_steps_over_10mm=events,
        all_estimated_max_step_by_slot_node_mm=steps.max(0).tolist(),
        mixed_source_rows_by_camera=mixed.sum((0, 1)).tolist(),
        body_frames_with_any_mixed_source_rows=int(mixed.any(-1).sum()),
        final_backtrace_body_row_frames=int(new_kind.sum()),
        final_backtrace_kind_counts={str(k): int((kinds == k).sum()) for k in np.unique(kinds)},
        mixed_rows_are_not_a_physical_swap_measure=True,
        support_recalculated_from_original_observations=False,
    )


def load_candidate(root, seal_name, digest, expected_stage, evidence):
    evidence.add(root / seal_name, digest)
    seal = proof.read_json(root / seal_name)
    require(seal["status"] == "ok" and seal["stage"] == expected_stage, "completed producer")
    summary = proof.read_json(common.bound_member(root, seal, "summary.json", evidence))
    path = common.bound_member(root, seal, "candidate.npz", evidence)
    with np.load(path, allow_pickle=False) as archive:
        candidate = {key: archive[key] for key in archive.files}
    require(common.signature(candidate) == summary["candidate_arrays"], "candidate signatures")
    common.metrics.validate_candidate(candidate, FIRST, STOP)
    return candidate, summary, seal


def run(variant, tracking_sha, output):
    variant, output = Path(variant).resolve(), Path(output).resolve()
    require(variant.is_relative_to(WORK), "workspace variant required")
    require(output.is_relative_to(WORK) and not output.exists(), "fresh comparison output")
    out, evidence, started = proof.Output(output), proof.Evidence(), proof.utc()
    try:
        old, old_summary, _ = load_candidate(
            CONTROL, "control.provenance.json", CONTROL_SHA, CONTROL.name, evidence
        )
        new, new_summary, new_seal = load_candidate(
            variant,
            "tracking.provenance.json",
            tracking_sha,
            "whole_row_body_ten_second_continuation_v1",
            evidence,
        )
        require(new_summary["process_interval"] == [START, STOP], "variant interval differs")
        require(
            new_summary["initialize_count"] == 1 and new_summary["restore_count"] == 7,
            "same inherited initialization required",
        )
        require(new_summary["external_pose_resets"] == 0, "no resets required")
        plan = proof.read_json(common.bound_member(variant, new_seal, "plan.json", evidence))
        require(plan["original_binding"] == old_summary["original_binding"], "different inputs")
        require(
            plan["parent_tracking_sha256"]
            == new_summary["parent_tracking_sha256"]
            == old_summary["parent_tracking_sha256"]
            and plan["parent_checkpoint_sha256"]
            == new_summary["parent_checkpoint_sha256"]
            == PARENT_CHECKPOINT_SHA,
            "different inherited Sparse state",
        )
        imported = proof.read_json(
            common.bound_member(variant, new_seal, "state_import.json", evidence)
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
            "unverified inherited-state migration",
        )
        results = {
            arm: summarize(candidate) for arm, candidate in (("control", old), ("body_row", new))
        }
        selected = old["frame_idx"] >= START
        sample = {}
        for arm, candidate in (("control", old), ("body_row", new)):
            keep = np.isin(candidate["frame_idx"], FRAMES)
            for key in ("frame_idx", "pose_xyz", "observed", "sources", "proposal_kind"):
                sample[arm + "_" + key] = candidate[key][keep].copy()
        out.write("review_samples.npz", sample)
        out.write("review_samples_signatures.json", common.signature(sample))
        delta = np.linalg.norm(new["pose_xyz"][selected] - old["pose_xyz"][selected], axis=-1)
        out.write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                process_interval=[START, STOP],
                terminal_horizon=STOP,
                frames=list(FRAMES),
                methods=results,
                raw_slot_pose_delta_mm=dict(
                    median=float(np.median(delta)), maximum=float(delta.max())
                ),
                raw_slots_independently_identity_aligned=False,
                chosen_prefix_pose_differences=int(
                    np.any(
                        old["pose_xyz"][~selected] != new["pose_xyz"][~selected], axis=(1, 2, 3)
                    ).sum()
                ),
                chosen_prefix_is_not_the_retained_beam_history=True,
                physical_ID_truth=False,
                independent_holdout=False,
                automatic_adoption=False,
                variant_tracking_sha256=tracking_sha,
                control_sha256=CONTROL_SHA,
                original_video_or_masks_read=False,
                tracking_or_fitting_performed=False,
            ),
        )
        out.write(
            "plan.json",
            dict(
                schema=STAGE,
                source_sha256={str(Path(__file__).resolve()): evidence.add(Path(__file__))},
                original_binding=old_summary["original_binding"],
                frames=list(FRAMES),
                completion_required=True,
                same_terminal_horizon=True,
                support_interpretation="Stored direct support, not independent detection accuracy.",
            ),
        )
        evidence.final_guard()
        out.final_guard()
        result = dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=out.records,
        )
    except BaseException as exc:
        result = dict(
            stage=STAGE,
            status="failed",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=out.records,
            error=repr(exc),
        )
        (output / "comparison.provenance.json").write_bytes(proof.json_bytes(result))
        raise
    (output / "comparison.provenance.json").write_bytes(proof.json_bytes(result))
    print("Completed whole-row/control comparison", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", type=Path, required=True)
    parser.add_argument("--tracking-sha", required=True)
    parser.add_argument("--output", type=Path, default=WORK / STAGE)
    args = parser.parse_args()
    run(args.variant, args.tracking_sha, args.output)
