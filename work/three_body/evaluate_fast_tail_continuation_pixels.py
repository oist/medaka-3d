"""Saved CPU pixel evaluation of the immutable two/five-minute Fast milestones.

Both comparison and preparation need explicit successful seal hashes. A final
successful five-minute tracking seal is additionally required through the
comparison plan: interim milestone metadata does not seal consumed mask bytes.
No mask archive, raw video, raw SL, fitting, GPU or identity inference is used.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import compare_fast_tail_continuation as comparison
import evaluate_fast_tail_two_minute_pixels as frozen
import numpy as np
import prepare_fast_tail_five_minute_pixels as prep5

producer, proof, old, prep = frozen.producer, frozen.proof, frozen.old, frozen.prep
WORK, REPO, require = frozen.WORK, frozen.REPO, frozen.require
START, HALF, FIRST = 66400, 76360, 49800
ARMS = frozen.ARMS
STAGE = "fast_tail_continuation_unused_pixel_evaluation_v1"
COMPARISON_CODE_SHA = "2e4a749d5ea13116f05a622afd47478af311525fbd856b32a116bd9ac765acba"
FROZEN_CODE_SHA = "773dd64c95d6d60371511808dab2e19b9fb160a19099fe85d430e89ec7c3fb40"
PREP5_CODE_SHA = "aa78fc562b70173cd86e3e2ed373115431fbbb7a60722e651d22b386506c2848"
score_frame = frozen.score_frame
view_counts = frozen.view_counts


def frame_schedule(stop):
    require(type(stop) is int and stop in (86320, 116200), "registered 2/5-minute horizon required")
    return tuple(range(START + 83, stop, 166))


def source_code():
    require(proof.valid_sha(COMPARISON_CODE_SHA), "reviewed comparison source pin pending")
    code = comparison.source_code()
    # Frozen source_code checks its renderer, refraction and sampling source pins.
    require(code == frozen.source_code(), "source closure differs")
    for module, digest in (
        (comparison, COMPARISON_CODE_SHA),
        (frozen, FROZEN_CODE_SHA),
        (prep5, PREP5_CODE_SHA),
    ):
        require(code.get(str(Path(module.__file__).resolve())) == digest, "reviewed source changed")
    require(str(Path(__file__).resolve()) in code, "new evaluator source missing")
    return code


def custom_seal(root, name, stage, digest, evidence):
    require(proof.valid_sha(digest), "explicit successful seal SHA required")
    root = Path(root).resolve(strict=True)
    evidence.add(root / name, digest)
    seal = proof.read_json(root / name)
    require(seal.get("stage") == stage and seal.get("status") == "ok", "successful stage required")
    index = {}
    for role, row in seal["outputs"].items():
        path = Path(row["path"]).resolve()
        require(
            not Path(role).is_absolute()
            and ".." not in Path(role).parts
            and path == (root / role).resolve()
            and path.is_relative_to(root)
            and proof.valid_sha(row["sha256"])
            and type(row["size_bytes"]) is int
            and row["size_bytes"] >= 0
            and str(path) not in index,
            "unsafe or aliased custom output",
        )
        index[str(path)] = row
    return dict(root=root, seal=seal, index=index)


def input_edge(seal, path, digest):
    """Support historical record-valued and new string-valued input digests."""
    path = Path(path).resolve()
    value = seal["inputs"].get(str(path))
    if isinstance(value, dict):
        require(Path(value["path"]).resolve() == path, "input alias path differs")
        value = value["sha256"]
    require(value == digest and proof.valid_sha(digest), "consumed input lineage differs")


def comparison_contract(summary, plan, stop):
    frames = frame_schedule(stop)
    expected = dict(
        schema=comparison.STAGE,
        status="ok",
        native_interval=[START, stop],
        native_frames=stop - START,
        horizon_stop=stop,
        same_parent_and_horizon=True,
        candidate_all21_and_source_arrays_unchanged=True,
        strict_support_recomputed_arms=2,
        strict_support_native_frames=2 * (stop - START),
        external_pose_resets=0,
        final_tracking_seal_verified=True,
        whole_five_minute_process_complete=True,
        milestone_is_immutable_completed_checkpoint=True,
        all_three_ID_truth=False,
        image_evaluation_included=False,
    )
    require(
        all(type(summary.get(k)) is type(v) and summary[k] == v for k, v in expected.items()),
        "successful same-horizon comparison with final tracking seal required",
    )
    require(
        plan["schema"] == comparison.STAGE
        and plan["horizon_stop"] == stop
        and plan["candidate_references"] == summary["candidate_references"]
        and set(plan["candidate_references"]) == set(ARMS)
        and plan["original_source_seal_sha256"] == comparison.ORIGINALS[stop]["seal"]
        and proof.valid_sha(plan["milestone_sha256"])
        and proof.valid_sha(plan.get("final_tracking_sha256")),
        "comparison plan/final consumed-mask seal required",
    )
    expected_windows = {name: list(value) for name, value in comparison.intervals(stop).items()}
    require(
        plan["intervals"] == summary["intervals"] == expected_windows, "reporting scope differs"
    )
    return frames


def validate_preparation(summary, config, stop):
    if stop == 86320:
        frozen.validate_preparation(summary, config)
        return
    frames = frame_schedule(stop)
    expected = dict(
        schema=prep5.STAGE,
        status="ok",
        native_interval=[START, stop],
        scored_frames=list(frames),
        reused_frames=list(frames[:120]),
        newly_sampled_frames=list(frames[120:]),
        sampler_config=config,
        counts=[96, 96, 64],
        seed_offset=8803010,
        original_seal_sha256=prep5.ORIGINAL_SHA,
        reused_preparation_sha256=prep5.PREPARED_SHA,
        newly_consumed_mask_archives=54,
        native_decoded_mask_checks=540,
        reused_mask_payload_reads=0,
        reused_sample_arrays_BYTEEXACT=True,
        fitted_coordinate_overlap=0,
        pixel_draws_per_arm=230400,
        fit_coordinates_reconstructed_deterministically=True,
        saved_run_fit_diagnostics_compared=False,
        pose_values_read=False,
        raw_BGR_frames_decoded=0,
        models=0,
        GPU_initialized=False,
        temporal_holdout=False,
        identity_truth=False,
    )
    require(all(summary.get(k) == v for k, v in expected.items()), "fixed300 preparation differs")
    require(
        config["mode"] == "detectron2"
        and config["width_ratio"] == 0.06
        and [config[k] for k in ("positive_samples", "near_samples", "global_samples")]
        == [384, 384, 256],
        "original renderer/sampler config differs",
    )


def bind_mask_sources(stop, summaries, bundles, original, final, first_minute):
    """Only sealed edges: all36/90 masks must reach both tracking arms and samples."""
    sources = {}
    for first in range(START, stop, prep.CHUNK):
        for view in old.VIEWS:
            path, digest = prep.mask_input(original["seal"], first, view)
            input_edge(final["seal"], path, digest)
            if first < HALF:
                input_edge(first_minute["seal"], path, digest)
                input_edge(bundles[60]["seal"], path, digest)
                require(
                    summaries[60]["source_mask_sha256"].get(str(path)) == digest,
                    "first60 mask differs",
                )
            elif first < 86320:
                input_edge(bundles[120]["seal"], path, digest)
            else:
                input_edge(bundles[300]["seal"], path, digest)
            if first < 86320:
                require(
                    summaries[120]["source_mask_sha256"].get(str(path)) == digest,
                    "first120 mask differs",
                )
            if stop == 116200:
                require(
                    summaries[300]["source_mask_sha256"].get(str(path)) == digest,
                    "fixed300 mask differs",
                )
            sources[str(path)] = digest
    for count, boundary in ((60, HALF), (120, 86320), (300, 116200)):
        if count in summaries:
            expected = {
                str(path): digest
                for first in range(START, min(stop, boundary), prep.CHUNK)
                for view in old.VIEWS
                for path, digest in [prep.mask_input(original["seal"], first, view)]
            }
            require(
                summaries[count]["source_mask_sha256"] == expected, "mask source membership differs"
            )
    require(len(sources) == (stop - START) // prep.CHUNK * 3, "all mask sources required")
    return sources


def load_context(args, evidence):
    stop = args.horizon_stop
    cmp = custom_seal(
        args.comparison,
        "comparison.provenance.json",
        comparison.STAGE,
        args.comparison_sha,
        evidence,
    )
    input_edge(cmp["seal"], Path(comparison.__file__), COMPARISON_CODE_SHA)
    plan = proof.read_json(frozen.custom_member(cmp, "plan.json", evidence))
    summary = proof.read_json(frozen.custom_member(cmp, "summary.json", evidence))
    frames = comparison_contract(summary, plan, stop)
    new = comparison.milestone_metadata(
        stop, plan["milestone_sha256"], evidence, plan["final_tracking_sha256"]
    )
    original = comparison.original_metadata(new, stop, evidence)
    original_bundle = original["bundle"]
    final = custom_seal(
        comparison.BRANCH,
        "tracking.provenance.json",
        comparison.runner.STAGE,
        plan["final_tracking_sha256"],
        evidence,
    )
    for path, digest in (
        (comparison.BRANCH / "tracking.provenance.json", plan["final_tracking_sha256"]),
        (
            comparison.BRANCH / f"milestones/through_{stop:06d}/summary.json",
            plan["milestone_sha256"],
        ),
        (
            original_bundle["root"] / "continuation.provenance.json",
            comparison.ORIGINALS[stop]["seal"],
        ),
    ):
        input_edge(cmp["seal"], path, digest)
    input_edge(final["seal"], Path(comparison.runner.__file__), comparison.RUNNER_SHA)
    config = original["metadata"]["context"]["tracker_config"]
    require(
        config == new["metadata"]["context"]["tracker_config"], "tracking sampler configs differ"
    )
    first_minute = producer.read_seal(
        producer.TAIL,
        "tracking.provenance.json",
        "tail_one_minute_continuous",
        producer.TAIL_SHA,
        evidence,
    )
    input_edge(new["parent"]["seal"], producer.TAIL / "tracking.provenance.json", producer.TAIL_SHA)
    prepared = custom_seal(
        args.prepared,
        "preparation.provenance.json",
        prep.STAGE if stop == 86320 else prep5.STAGE,
        args.preparation_sha,
        evidence,
    )
    selected_count = len(frames)
    selected_summary = proof.read_json(frozen.custom_member(prepared, "summary.json", evidence))
    validate_preparation(selected_summary, config, stop)
    input_edge(
        prepared["seal"],
        Path(prep.__file__) if stop == 86320 else Path(prep5.__file__),
        frozen.PREPARATION_CODE_SHA if stop == 86320 else PREP5_CODE_SHA,
    )
    input_edge(
        prepared["seal"],
        original_bundle["root"] / "continuation.provenance.json",
        comparison.ORIGINALS[stop]["seal"],
    )
    if stop == 116200:
        prior120 = custom_seal(
            prep5.PREPARED, "preparation.provenance.json", prep.STAGE, prep5.PREPARED_SHA, evidence
        )
        input_edge(
            prepared["seal"], prep5.PREPARED / "preparation.provenance.json", prep5.PREPARED_SHA
        )
        summary120 = proof.read_json(frozen.custom_member(prior120, "summary.json", evidence))
        frozen.validate_preparation(summary120, config)
    else:
        prior120, summary120 = prepared, selected_summary
    input_edge(prior120["seal"], Path(prep.__file__), frozen.PREPARATION_CODE_SHA)
    prior60 = producer.read_seal(
        prep.PREPARED, "preparation.provenance.json", old.PREP_STAGE, prep.PREPARED_SHA, evidence
    )
    input_edge(prior120["seal"], prep.PREPARED / "preparation.provenance.json", prep.PREPARED_SHA)
    summary60 = proof.read_json(producer.consumed(prior60, "summary.json", evidence))
    require(
        summary60["sampler_config"] == config
        and summary60["scored_frames"] == list(frames[:60])
        and summary60["fitted_coordinate_overlap"] == 0
        and summary60["calibration"]
        == summary120["calibration"]
        == selected_summary["calibration"],
        "first60 preparation lineage differs",
    )
    summaries = {60: summary60, 120: summary120, selected_count: selected_summary}
    bundles = {60: prior60, 120: prior120, selected_count: prepared}
    for relative, digest in prep.PINS.items():
        input_edge(prior120["seal"], REPO / relative, digest)
        if stop == 116200:
            input_edge(prepared["seal"], REPO / relative, digest)
    masks = bind_mask_sources(stop, summaries, bundles, original_bundle, final, first_minute)
    shared = original["binding"]["inputs"]["seed"]["SL_and_calibration_sha256"]
    require(
        shared == plan["shared_SL_calibration_sha256"] and len(shared) == 7,
        "SL/calibration binding differs",
    )
    calibration = Path(selected_summary["calibration"]).resolve(strict=True)
    calibration_sources = {
        path: digest for path, digest in shared.items() if Path(path).parent == calibration
    }
    require(
        len(calibration_sources) == 4
        and any(Path(path).name == "calibration_manifest.json" for path in calibration_sources),
        "calibration4 required",
    )
    for path, digest in calibration_sources.items():
        for bundle in (original_bundle, final, first_minute, prior60, cmp):
            input_edge(bundle["seal"], path, digest)
        evidence.add(path, digest)
    rig = old.load_rig(calibration)
    cameras = producer.original.phase.COMMON.normalize([asdict(camera) for camera in rig.cameras])
    require(
        cameras
        == original["metadata"]["context"]["cameras"]
        == new["metadata"]["context"]["cameras"],
        "tracking and pixel projection cameras differ",
    )
    membership = proof.read_json(frozen.custom_member(prepared, "membership.json", evidence))
    prep5.validate_membership_order(membership, frames)
    return dict(
        comparison=cmp,
        comparison_plan=plan,
        comparison_summary=summary,
        new=new,
        original=original_bundle,
        final=final,
        prepared=prepared,
        frames=frames,
        config=config,
        rig=rig,
        membership=membership,
        mask_sources=masks,
        calibration_sources=calibration_sources,
    )


def candidate_poses(context, evidence, stop):
    values, references = [], {}
    for arm in ARMS:
        if arm == "original":
            candidate, reference = comparison.previous.load_candidate(
                context[arm], "candidate.npz", evidence, stop
            )
        else:
            candidate, reference = comparison.load_milestone_candidate(
                context["new"], stop, evidence
            )
        require(
            reference == context["comparison_plan"]["candidate_references"][arm],
            "comparison candidate differs",
        )
        input_edge(context["comparison"]["seal"], reference["path"], reference["sha256"])
        before = comparison.previous.signature(candidate)
        frames = context["frames"]
        index = np.searchsorted(candidate["frame_idx"], frames)
        require(
            np.array_equal(candidate["frame_idx"][index], frames),
            "fixed sample frame lookup differs",
        )
        values.append(candidate["pose_xyz"][index].copy())
        require(comparison.previous.signature(candidate) == before, "candidate mutated")
        references[arm] = reference
    return np.stack(values, axis=1), references


def summarize(errors, rows, indices, frames):
    """Frozen expression generalized only from120 to300 rows; retain every view."""
    require(
        len(frames) in (120, 300)
        and tuple(frames) == frame_schedule(START + 166 * len(frames))
        and errors.dtype == np.float64
        and errors.shape == (len(frames), 2, 3)
        and np.isfinite(errors).all()
        and (errors >= 0).all()
        and len(rows) == len(frames)
        and [row["frame_idx"] for row in rows] == list(frames),
        "fixed native schedule and float64 all-view errors required",
    )
    indices = tuple(indices)
    require(
        indices
        and all(type(i) is int and 0 <= i < len(frames) for i in indices)
        and len(set(indices)) == len(indices),
        "nonempty unique reporting indices required",
    )
    if len(frames) == 120:
        return frozen.summarize(errors, rows, list(indices))
    selected = errors[list(indices)]
    selected_rows = [rows[i] for i in indices]
    means = selected.mean((0, 2))
    delta = (selected[:, 1] - selected[:, 0]).mean(1)
    per_view = selected[:, 1] - selected[:, 0]
    views = [row["view_counts"] for row in selected_rows]
    return dict(
        frames=len(selected_rows),
        frame_view_slots=3 * len(selected_rows),
        arms=2,
        pixel_draws_per_arm=len(selected_rows) * 3 * 256,
        original_mean_error=float(means[0]),
        fast_tail_mean_error=float(means[1]),
        mean_difference=float(means[1] - means[0]),
        relative_error_reduction_percent=None
        if means[0] == 0
        else float(100 * (means[0] - means[1]) / means[0]),
        relative_reduction_unavailable_reason="zero_original_mean" if means[0] == 0 else None,
        frames_lower_error=int((delta < 0).sum()),
        frames_higher_error=int((delta > 0).sum()),
        frames_equal_error=int((delta == 0).sum()),
        mean_error_by_arm_view=selected.mean(0).tolist(),
        mean_difference_by_view=per_view.mean(0).tolist(),
        lower_higher_equal_by_view=np.stack(
            [(per_view < 0).sum(0), (per_view > 0).sum(0), (per_view == 0).sum(0)]
        ).tolist(),
        unavailable_source_views=sum(
            not row["source_available"] for rows3 in views for row in rows3
        ),
        available_empty_foreground_views=sum(
            row["available_empty_foreground"] for rows3 in views for row in rows3
        ),
        neutral_zero_weight_views=sum(
            row["neutral_zero_weight"] for rows3 in views for row in rows3
        ),
        no_unused_foreground_views=sum(
            row["no_unused_foreground"] for rows3 in views for row in rows3
        ),
        weighted_draws_per_arm=sum(row["weighted_draws"] for rows3 in views for row in rows3),
        invalid_ray_draws_per_arm=sum(
            sum(row["invalid_ray_draws_by_view"]) for row in selected_rows
        ),
        neutral_views_retained_in_means=True,
        fitted_coordinate_overlap=0,
    )


def run(args):
    require(args.execute is True, "explicit evaluation execution required")
    frames = frame_schedule(args.horizon_stop)
    for value in (args.preparation_sha, args.comparison_sha):
        require(proof.valid_sha(value), "explicit successful seal SHA required")
    code = source_code()
    destination = args.output.resolve()
    require(
        destination.is_relative_to(WORK) and not destination.exists(), "fresh work output required"
    )
    evidence, output = proof.Evidence(), proof.Output(destination)
    started, begin, rows = proof.utc(), time.perf_counter(), []
    try:
        require(not old.torch.cuda.is_initialized(), "CPU-only evaluation required")
        for path, digest in code.items():
            evidence.add(path, digest)
            output.write(
                "code_snapshot/" + Path(path).relative_to(REPO).as_posix(), Path(path).read_bytes()
            )
        context = load_context(args, evidence)
        old.torch.set_num_threads(2)
        poses, references = candidate_poses(context, evidence, args.horizon_stop)
        before = proof.signature(poses)
        output.write(
            "plan.json",
            dict(
                schema=STAGE,
                arms=list(ARMS),
                native_interval=[START, args.horizon_stop],
                horizon_stop=args.horizon_stop,
                scored_native_frames=list(frames),
                candidate_references=references,
                preparation_sha256=args.preparation_sha,
                comparison_sha256=args.comparison_sha,
                final_tracking_sha256=context["comparison_plan"]["final_tracking_sha256"],
                source_sha256=code,
                source_mask_sha256=context["mask_sources"],
                calibration_sha256=context["calibration_sources"],
                sampler_config=context["config"],
                renderer_policy="frozen dynamic-width CPU float64 renderer for both arms",
                score_expression="((union_alpha - target)**2 * weights * ray_valid).sum(-1)",
                all_three_views_remain_in_mean=True,
                reused_first60_sample_arrays=True,
                old_first60_score_equality_expected=False,
                fit_coordinates_reconstructed_deterministically=True,
                saved_run_fit_diagnostics_compared=False,
                mask_archive_payloads_read_here=0,
                checkpoint_array_payloads_read_here=0,
                raw_SL_arrays_read_here=0,
                fast_mask_payload_lineage_verified=True,
                temporal_holdout=False,
                anatomical_or_ID_truth=False,
            ),
        )
        for t, native in enumerate(frames):
            path = frozen.custom_member(context["prepared"], f"frame_{native:06d}.npz", evidence)
            with np.load(path, allow_pickle=False) as archive:
                require(len(archive.files) == len(set(archive.files)), "duplicate sample field")
                samples = {key: archive[key].copy() for key in archive.files}
            saved = comparison.previous.signature(samples)
            counts = view_counts(samples, context["membership"][3 * t : 3 * t + 3], native)
            scored = score_frame(context["rig"], poses[t], samples)
            require(comparison.previous.signature(samples) == saved, "samples mutated")
            error = scored["per_view_error"]
            rows.append(
                dict(
                    frame_idx=native,
                    original_error=error[0].tolist(),
                    fast_tail_error=error[1].tolist(),
                    mean_difference=float((error[1] - error[0]).mean()),
                    view_counts=counts,
                    invalid_ray_draws_by_view=(~scored["ray_valid"]).sum(1).tolist(),
                )
            )
            output.write(f"frame_{native:06d}.npz", dict(poses=poses[t], **scored, **samples))
            if (t + 1) % 10 == 0:
                print(
                    json.dumps(
                        dict(
                            event="continuation_pixel_scores",
                            completed=t + 1,
                            total=len(frames),
                            native=native,
                        )
                    ),
                    flush=True,
                )
        errors = np.array(
            [[row["original_error"], row["fast_tail_error"]] for row in rows], np.float64
        )
        intervals = {"full": range(len(frames))}
        intervals.update(
            {f"minute_{i + 1}": range(i * 60, (i + 1) * 60) for i in range(len(frames) // 60)}
        )
        summary = dict(
            schema=STAGE,
            status="ok",
            horizon_stop=args.horizon_stop,
            native_interval=[START, args.horizon_stop],
            arms=list(ARMS),
            scored_native_frames=list(frames),
            scored_frame_count=len(frames),
            intervals={
                name: summarize(errors, rows, indices, frames)
                for name, indices in intervals.items()
            },
            candidate_references=references,
            strict_support_comparison_sha256=args.comparison_sha,
            final_tracking_seal_sha256=context["comparison_plan"]["final_tracking_sha256"],
            fitted_coordinate_overlap=0,
            all_three_views_remain_in_mean=True,
            fast_mask_payload_lineage_verified=True,
            reused_first60_arrays=True,
            old_first60_scores_not_used_as_control=True,
            fit_coordinates_reconstructed_deterministically=True,
            saved_run_fit_diagnostics_compared=False,
            same_images_spatial_resampling=True,
            temporal_holdout=False,
            anatomical_or_ID_truth=False,
            candidate_reselected=False,
            tracking_performed=False,
            raw_BGR_frames_decoded=0,
            mask_archives_decoded=0,
            raw_SL_arrays_opened=0,
            GPU_used=False,
            adopted=False,
            pose_arrays_unchanged=proof.signature(poses) == before,
            elapsed_seconds_before_final_guards=time.perf_counter() - begin,
        )
        require(summary["pose_arrays_unchanged"], "selected poses mutated")
        require(not old.torch.cuda.is_initialized(), "evaluation initialized CUDA")
        output.write("frames.json", rows)
        output.write("summary.json", summary)
        require(source_code() == code, "source closure changed")
        evidence.final_guard()
        output.final_guard()
        seal = dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=output.records,
        )
    except BaseException as error:
        if "frames.json" not in output.records:
            output.write("frames.json", rows)
        seal = dict(
            stage=STAGE,
            status="failed",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=output.records,
            error=repr(error),
            completed_frames=[row["frame_idx"] for row in rows],
        )
        with (destination / "evaluation.provenance.json").open("xb") as stream:
            stream.write(proof.json_bytes(seal))
        raise
    with (destination / "evaluation.provenance.json").open("xb") as stream:
        stream.write(proof.json_bytes(seal))
    print(json.dumps(summary["intervals"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--horizon-stop", type=int, choices=(86320, 116200), required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--comparison-sha", required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--preparation-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
