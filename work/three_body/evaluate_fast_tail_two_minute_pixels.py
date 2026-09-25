"""Fixed120, same86320, saved-only CPU silhouette comparison of two tracking arms.

All samples are shared. First60 sample arrays retain their old sealed values;
their scores can change with the later backtrace. The next60 fit coordinates
were deterministically reconstructed, not compared with saved fit diagnostics.
No video, mask decode, raw SL array, fit, tracking, CUDA, ID inference or adoption.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import compare_fast_tail_two_minutes as comparison
import numpy as np
import prepare_fast_tail_two_minute_pixels as prep

producer, proof, old = prep.producer, prep.proof, prep.old
WORK, REPO, require = prep.WORK, prep.REPO, prep.require
START, HALF, STOP, FRAMES = prep.START, prep.HALF, prep.STOP, prep.FRAMES
ARMS = ("original", "fast_tail")
STAGE = "fast_tail_two_minute_unused_pixel_evaluation_v1"
PREPARATION_CODE_SHA = "25db5d375dc353e2c640b3c67c287bde4f0b3a5f3e5c7fd9829c1d04b4522f9f"
PREPARED_ROOT = WORK / "fast_tail_two_minute_unused_pixel_samples_v1"
COMPARISON_CODE_SHA = "e07b51f364f5d6f85e8e26d34a0051145412df29e34c91d379d2141aec9b649d"
RENDERER_SHA = "dfb585e9d51214a2a3c9447d4bc81da8af80d3451e8fc3ef7a44b1a841732e1c"
CALIB_SHA = "60849004cb53febe69f437e7afa8b18cb2b65213ba76ce7c921f7022d2be4274"


def source_code():
    require(proof.valid_sha(PREPARATION_CODE_SHA), "reviewed preparation source pin pending")
    code = comparison.source_code()
    pins = {
        **prep.PINS,
        "work/three_body/prepare_fast_tail_two_minute_pixels.py": PREPARATION_CODE_SHA,
        "work/three_body/compare_fast_tail_two_minutes.py": COMPARISON_CODE_SHA,
        "labs/20260909_three_body/renderer.py": RENDERER_SHA,
        "src/medaka3d/calib.py": CALIB_SHA,
    }
    for relative, digest in pins.items():
        require(code.get(str(REPO / relative)) == digest, "frozen source differs: " + relative)
    require(str(Path(__file__).resolve()) in code, "evaluation source missing")
    return code


def custom_seal(root, digest, evidence):
    """The preparation uses Output.size_bytes and string input digests."""
    root = root.resolve(strict=True)
    evidence.add(root / "preparation.provenance.json", digest)
    seal = proof.read_json(root / "preparation.provenance.json")
    require(
        seal.get("stage") == prep.STAGE and seal.get("status") == "ok", "successful prep required"
    )
    index = {}
    for name, item in seal["outputs"].items():
        path = Path(item["path"]).resolve()
        require(
            not Path(name).is_absolute()
            and ".." not in Path(name).parts
            and path == (root / name).resolve()
            and path.is_relative_to(root)
            and proof.valid_sha(item["sha256"])
            and type(item["size_bytes"]) is int
            and item["size_bytes"] >= 0,
            "unsafe custom output member",
        )
        require(str(path) not in index, "custom output alias")
        index[str(path)] = item
    return dict(root=root, seal=seal, index=index)


def custom_member(bundle, name, evidence):
    path = (bundle["root"] / name).resolve(strict=True)
    row = bundle["index"].get(str(path))
    require(row is not None and path.stat().st_size == row["size_bytes"], "unsealed/resized sample")
    evidence.add(path, row["sha256"])
    return path


def custom_edge(seal, path, digest):
    require(
        seal["inputs"].get(str(Path(path).resolve())) == digest, "custom consumed input differs"
    )


def validate_preparation(summary, config):
    required = dict(
        schema=prep.STAGE,
        status="ok",
        native_interval=[START, STOP],
        scored_frames=list(FRAMES),
        reused_frames=list(FRAMES[:60]),
        newly_sampled_frames=list(FRAMES[60:]),
        sampler_config=config,
        counts=[96, 96, 64],
        seed_offset=8803010,
        original_seal_sha256=comparison.ORIGINAL_SHA,
        reused_preparation_sha256=prep.PREPARED_SHA,
        native_decoded_mask_checks=180,
        reused_sample_arrays_BYTEEXACT=True,
        fitted_coordinate_overlap=0,
        pixel_draws_per_arm=92160,
        fit_coordinates_reconstructed_deterministically=True,
        saved_run_fit_diagnostics_compared=False,
        pose_values_read=False,
        raw_BGR_frames_decoded=0,
        models=0,
        GPU_initialized=False,
        temporal_holdout=False,
        identity_truth=False,
    )
    require(
        all(summary.get(k) == v for k, v in required.items()), "fixed120 sample protocol differs"
    )
    require(
        config["mode"] == "detectron2"
        and config["width_ratio"] == 0.06
        and [config[k] for k in ("positive_samples", "near_samples", "global_samples")]
        == [384, 384, 256],
        "original sampler/renderer configuration differs",
    )


def comparison_contract(summary, plan, tracking_sha):
    require(
        summary.get("schema") == comparison.STAGE
        and summary.get("status") == "ok"
        and summary.get("native_frames") == 19920
        and summary.get("horizon_stop") == STOP
        and summary.get("same_parent_and_horizon") is True
        and summary.get("candidate_all21_and_source_arrays_unchanged") is True
        and summary.get("strict_support_recomputed_arms") == 2
        and summary.get("strict_support_native_frames") == 39840
        and plan["arms"] == list(ARMS)
        and plan["horizon_stop"] == STOP
        and plan["source_tracking_sha256"] == tracking_sha
        and plan["old_original_seal_sha256"] == comparison.ORIGINAL_SHA
        and plan["later_image_schedule"] == list(FRAMES)
        and plan["candidate_references"] == summary["candidate_references"],
        "successful same86320 two-arm comparison required",
    )


def bind_mask_sources(summary, previous_summary, prepared, previous, arms):
    """Proof edges only: do not rehash or decode the 36 mask archives."""
    sources = {}
    for first in range(START, STOP, prep.CHUNK):
        for view in old.VIEWS:
            path, digest = prep.mask_input(arms[0]["seal"], first, view)
            other_path, other_digest = prep.mask_input(arms[1]["seal"], first, view)
            require((path, digest) == (other_path, other_digest), "arms consumed different masks")
            require(summary["source_mask_sha256"].get(str(path)) == digest, "prepared mask differs")
            if first < HALF:
                require(
                    previous_summary["source_mask_sha256"].get(str(path)) == digest,
                    "reused first60 mask differs",
                )
                comparison.input_edge(previous["seal"], path, digest)
            else:
                custom_edge(prepared["seal"], path, digest)
            sources[str(path)] = digest
    require(
        len(sources) == 36 and sources == summary["source_mask_sha256"],
        "all36 mask sources required",
    )
    return sources


def load_context(args, evidence):
    """Small seals and metadata only; candidate values and samples are loaded later."""
    cmp = producer.read_seal(
        args.comparison,
        "comparison.provenance.json",
        comparison.STAGE,
        args.comparison_sha,
        evidence,
    )
    require(
        cmp["seal"]["params"].get("all_consumed_source_input_output_SHA_unchanged") is True,
        "comparison final guard missing",
    )
    comparison.input_edge(cmp["seal"], Path(comparison.__file__), COMPARISON_CODE_SHA)
    cmp_summary = proof.read_json(producer.consumed(cmp, "summary.json", evidence))
    cmp_plan = proof.read_json(producer.consumed(cmp, "plan.json", evidence))
    comparison_contract(cmp_summary, cmp_plan, args.tracking_sha)
    original = producer.read_seal(
        comparison.ORIGINAL,
        "continuation.provenance.json",
        "three_body_long_continuation",
        comparison.ORIGINAL_SHA,
        evidence,
    )
    tail = producer.read_seal(
        WORK / producer.STAGE,
        "tracking.provenance.json",
        producer.STAGE,
        args.tracking_sha,
        evidence,
    )
    for bundle, name, digest in (
        (original, "continuation.provenance.json", comparison.ORIGINAL_SHA),
        (tail, "tracking.provenance.json", args.tracking_sha),
    ):
        comparison.input_edge(cmp["seal"], bundle["root"] / name, digest)
    comparison.completion(proof.read_json(producer.consumed(tail, "summary.json", evidence)))
    require(
        original["seal"]["params"]["through_frame"] == STOP
        and tail["seal"]["params"].get("all_source_input_snapshot_output_SHA_unchanged") is True,
        "successful completed producers required",
    )
    binding = original["seal"]["params"]["bindings"]
    require(
        binding == tail["seal"]["params"]["bindings"]["original_binding"],
        "original input binding differs",
    )
    original_meta = proof.read_json(
        producer.consumed(original, "checkpoint/metadata.json", evidence)
    )
    tail_meta = proof.read_json(
        producer.consumed(tail, "checkpoints/through_086320/metadata.json", evidence)
    )
    config = original_meta["context"]["tracker_config"]
    require(config == tail_meta["context"]["tracker_config"], "arm sampler config differs")
    prepared = custom_seal(args.prepared, args.preparation_sha, evidence)
    custom_edge(prepared["seal"], Path(prep.__file__), PREPARATION_CODE_SHA)
    custom_edge(
        prepared["seal"], original["root"] / "continuation.provenance.json", comparison.ORIGINAL_SHA
    )
    custom_edge(prepared["seal"], prep.PREPARED / "preparation.provenance.json", prep.PREPARED_SHA)
    summary = proof.read_json(custom_member(prepared, "summary.json", evidence))
    validate_preparation(summary, config)
    previous = producer.read_seal(
        prep.PREPARED, "preparation.provenance.json", old.PREP_STAGE, prep.PREPARED_SHA, evidence
    )
    previous_summary = proof.read_json(producer.consumed(previous, "summary.json", evidence))
    require(
        previous_summary["sampler_config"] == config
        and previous_summary["scored_frames"] == list(FRAMES[:60])
        and previous_summary["calibration"] == summary["calibration"],
        "reused first60 protocol differs",
    )
    for relative, digest in prep.PINS.items():
        custom_edge(prepared["seal"], REPO / relative, digest)
    sources = bind_mask_sources(summary, previous_summary, prepared, previous, (original, tail))
    shared = binding["inputs"]["seed"]["SL_and_calibration_sha256"]
    require(
        shared == cmp_plan["shared_SL_calibration_sha256"] and len(shared) == 7,
        "shared calibration/SL lineage differs",
    )
    calibration = Path(summary["calibration"]).resolve(strict=True)
    calib_files = {p: h for p, h in shared.items() if Path(p).parent == calibration}
    require(
        len(calib_files) == 4
        and any(Path(p).name == "calibration_manifest.json" for p in calib_files),
        "exact calibration4 required",
    )
    for path, digest in calib_files.items():
        for bundle in (original, tail, previous, cmp):
            comparison.input_edge(bundle["seal"], path, digest)
        evidence.add(path, digest)
    rig = old.load_rig(calibration)
    camera_context = producer.original.phase.COMMON.normalize([asdict(c) for c in rig.cameras])
    require(
        camera_context == original_meta["context"]["cameras"] == tail_meta["context"]["cameras"],
        "common calibrated renderer differs from tracking",
    )
    for path, digest in binding["code_sha256"].items():
        evidence.add(path, digest)
    membership = proof.read_json(custom_member(prepared, "membership.json", evidence))
    require(
        [(r["frame"], r["view"]) for r in membership]
        == [(f, v) for f in FRAMES for v in old.VIEWS],
        "all360 sample membership rows required",
    )
    return dict(
        comparison=cmp,
        comparison_plan=cmp_plan,
        prepared=prepared,
        preparation_summary=summary,
        original=original,
        fast_tail=tail,
        config=config,
        rig=rig,
        membership=membership,
        mask_sources=sources,
        calibration_sources=calib_files,
    )


def candidate_poses(context, evidence):
    """Verify both immutable candidate21 archives; only the fixed120 poses are rendered."""
    poses, references = [], {}
    for arm in ARMS:
        arrays, reference = comparison.load_candidate(context[arm], "candidate.npz", evidence, STOP)
        require(
            reference == context["comparison_plan"]["candidate_references"][arm],
            "comparison candidate21 differs",
        )
        comparison.input_edge(context["comparison"]["seal"], reference["path"], reference["sha256"])
        require(
            np.array_equal(arrays["frame_idx"][START - comparison.FIRST :], np.arange(START, STOP)),
            "same19920 candidate extent required",
        )
        before = comparison.signature(arrays)
        indices = np.searchsorted(arrays["frame_idx"], FRAMES)
        require(
            np.array_equal(arrays["frame_idx"][indices], FRAMES), "fixed120 native lookup differs"
        )
        poses.append(arrays["pose_xyz"][indices].copy())
        require(comparison.signature(arrays) == before, "candidate values changed")
        references[arm] = reference
    return np.stack(poses, axis=1), references


def view_counts(samples, rows, native):
    """All three views retained; unavailable and available-empty are separate."""
    require(
        [(r["frame"], r["view"]) for r in rows] == [(native, v) for v in old.VIEWS],
        "frame membership differs",
    )
    result = []
    for vi, row in enumerate(rows):
        pools = row["available_unused_pool_sizes"]
        require(
            type(row["source_mask_available"]) is bool
            and type(row["source_foreground_pixels"]) is int
            and row["source_foreground_pixels"] >= 0
            and len(pools) == 3
            and all(type(p) is int and p >= 0 for p in pools)
            and row["overlap_count"] == 0
            and row["sampler_seed"] == native * 7 + vi + old.SEED_OFFSET
            and row["known_weight_sum"] == float(samples["weights"][vi].sum()),
            "sample membership/weight differs",
        )
        mass = float(samples["weights"][vi].sum())
        require((mass == 0) == (pools[0] == 0), "neutral empty-foreground rule differs")
        require(
            row["source_mask_available"] or row["source_foreground_pixels"] == 0,
            "unavailable source has foreground",
        )
        result.append(
            dict(
                view=old.VIEWS[vi],
                source_available=row["source_mask_available"],
                source_foreground_pixels=row["source_foreground_pixels"],
                available_empty_foreground=row["source_mask_available"]
                and row["source_foreground_pixels"] == 0,
                neutral_zero_weight=mass == 0,
                no_unused_foreground=pools[0] == 0,
                known_weight_sum=mass,
                weighted_draws=int((samples["weights"][vi] > 0).sum()),
            )
        )
    return result


def score_frame(rig, poses, samples):
    """Exact frozen evaluator expression and CPU float64 dynamic-width renderer."""
    prep.validate_samples(samples)
    require(
        poses.dtype == np.float64 and poses.shape == (2, 3, 3, 3) and np.isfinite(poses).all(),
        "fixed two-arm pose shape/finite",
    )
    renderer = old.RefractiveFishRenderer(
        rig,
        {v: samples["pixels"][i] for i, v in enumerate(old.VIEWS)},
        old.RendererConfig(width_ratio=0.06),
        device="cpu",
        dtype=old.torch.float64,
    )
    with old.torch.no_grad():
        alpha = renderer.render(poses).union_alpha.numpy()
    valid = renderer.ray_valid.numpy()
    error = ((alpha - samples["target"]) ** 2 * samples["weights"] * valid).sum(-1)
    require(
        alpha.shape == (2, 3, 256)
        and valid.shape == (3, 256)
        and error.shape == (2, 3)
        and np.isfinite(error).all(),
        "rendered score invalid",
    )
    return dict(union_alpha=alpha, ray_valid=valid, per_view_error=error)


def summarize(errors, rows, indices):
    require(
        errors.shape == (120, 2, 3) and np.isfinite(errors).all() and (errors >= 0).all(),
        "fixed120 errors required",
    )
    require(
        len(rows) == 120 and [r["frame_idx"] for r in rows] == list(FRAMES),
        "fixed120 score rows required",
    )
    selected = errors[indices]
    selected_rows = [rows[i] for i in indices]
    require(len(selected_rows) > 0, "empty reporting interval")
    means = selected.mean((0, 2))
    delta = (selected[:, 1] - selected[:, 0]).mean(1)
    per_view = selected[:, 1] - selected[:, 0]
    views = [r["view_counts"] for r in selected_rows]
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
        unavailable_source_views=sum(not r["source_available"] for rows3 in views for r in rows3),
        available_empty_foreground_views=sum(
            r["available_empty_foreground"] for rows3 in views for r in rows3
        ),
        neutral_zero_weight_views=sum(r["neutral_zero_weight"] for rows3 in views for r in rows3),
        no_unused_foreground_views=sum(r["no_unused_foreground"] for rows3 in views for r in rows3),
        weighted_draws_per_arm=sum(r["weighted_draws"] for rows3 in views for r in rows3),
        invalid_ray_draws_per_arm=sum(sum(r["invalid_ray_draws_by_view"]) for r in selected_rows),
        neutral_views_retained_in_means=True,
        fitted_coordinate_overlap=0,
    )


def run(args):
    require(args.execute, "explicit evaluation execution required")
    for value in (args.preparation_sha, args.comparison_sha, args.tracking_sha):
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
        poses, references = candidate_poses(context, evidence)
        pose_signature = proof.signature(poses)
        output.write(
            "plan.json",
            dict(
                schema=STAGE,
                arms=list(ARMS),
                native_interval=[START, STOP],
                horizon_stop=STOP,
                scored_native_frames=list(FRAMES),
                candidate_references=references,
                preparation_sha256=args.preparation_sha,
                comparison_sha256=args.comparison_sha,
                tracking_sha256=args.tracking_sha,
                original_sha256=comparison.ORIGINAL_SHA,
                source_sha256=code,
                source_mask_sha256=context["mask_sources"],
                calibration_sha256=context["calibration_sources"],
                sampler_config=context["config"],
                renderer_policy="same frozen dynamic-width CPU float64 renderer for both arms",
                score_expression="((union_alpha - target)**2 * weights * ray_valid).sum(-1)",
                all_three_views_remain_in_mean=True,
                reused_first60_sample_arrays=True,
                old_first60_score_equality_expected=False,
                first60_score_change_reason=(
                    "Same sample values but final86320 can revise the76360 backtrace."
                ),
                fit_coordinates_reconstructed_deterministically=True,
                saved_run_fit_diagnostics_compared=False,
                mask_archive_payloads_read_here=0,
                checkpoint_array_payloads_read_here=0,
                raw_SL_arrays_read_here=0,
                temporal_holdout=False,
                anatomical_or_ID_truth=False,
            ),
        )
        for t, native in enumerate(FRAMES):
            path = custom_member(context["prepared"], f"frame_{native:06d}.npz", evidence)
            with np.load(path, allow_pickle=False) as archive:
                samples = {k: archive[k].copy() for k in archive.files}
            before = comparison.signature(samples)
            view_info = view_counts(samples, context["membership"][3 * t : 3 * t + 3], native)
            scored = score_frame(context["rig"], poses[t], samples)
            require(comparison.signature(samples) == before, "prepared samples mutated")
            error = scored["per_view_error"]
            rows.append(
                dict(
                    frame_idx=native,
                    original_error=error[0].tolist(),
                    fast_tail_error=error[1].tolist(),
                    mean_difference=float((error[1] - error[0]).mean()),
                    view_counts=view_info,
                    invalid_ray_draws_by_view=(~scored["ray_valid"]).sum(1).tolist(),
                )
            )
            output.write(f"frame_{native:06d}.npz", dict(poses=poses[t], **scored, **samples))
            if (t + 1) % 10 == 0:
                print(
                    json.dumps(dict(event="fixed_pixel_scores", completed=t + 1, native=native)),
                    flush=True,
                )
        errors = np.array([[r["original_error"], r["fast_tail_error"]] for r in rows], np.float64)
        result = dict(
            schema=STAGE,
            status="ok",
            native_interval=[START, STOP],
            horizon_stop=STOP,
            arms=list(ARMS),
            scored_native_frames=list(FRAMES),
            scored_frame_count=120,
            intervals={
                name: summarize(errors, rows, ix)
                for name, ix in (
                    ("full", range(120)),
                    ("first_minute", range(60)),
                    ("second_minute", range(60, 120)),
                )
            },
            candidate_references=references,
            fitted_coordinate_overlap=0,
            reused_first60_arrays=True,
            old_first60_scores_not_used_as_control=True,
            fit_coordinates_reconstructed_deterministically=True,
            saved_run_fit_diagnostics_compared=False,
            strict_support_comparison_sha256=args.comparison_sha,
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
            pose_arrays_unchanged=proof.signature(poses) == pose_signature,
            elapsed_seconds_before_final_guards=time.perf_counter() - begin,
            runtime=dict(
                numpy=np.__version__,
                torch=old.torch.__version__,
                torch_threads=old.torch.get_num_threads(),
            ),
        )
        require(result["pose_arrays_unchanged"], "selected pose values changed")
        require(not old.torch.cuda.is_initialized(), "evaluation initialized CUDA")
        output.write("frames.json", rows)
        output.write("summary.json", result)
        require(source_code() == code, "loaded source set changed")
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
            completed_frames=[r["frame_idx"] for r in rows],
            inputs=evidence.paths,
            outputs=output.records,
            error_type=type(error).__name__,
            error=str(error),
        )
        with (destination / "evaluation.provenance.json").open("xb") as stream:
            stream.write(proof.json_bytes(seal))
        raise
    with (destination / "evaluation.provenance.json").open("xb") as stream:
        stream.write(proof.json_bytes(seal))
    print(json.dumps(result["intervals"]), flush=True)
    return seal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--prepared", type=Path, default=PREPARED_ROOT)
    parser.add_argument("--comparison", type=Path, default=WORK / comparison.STAGE)
    parser.add_argument("--output", type=Path, default=WORK / STAGE)
    for name in ("preparation-sha", "comparison-sha", "tracking-sha"):
        parser.add_argument("--" + name, required=True)
    run(parser.parse_args())
