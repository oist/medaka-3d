"""CPU image-term counterfactual on three complete saved fitted Nodewise banks.

Both frozen observation modes use the same seed and counts, not the same top
pixels. Other costs and parents remain saved values; no fitting or trajectory
is computed. CPU/CUDA baseline disagreement is measured, never called bit exact.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import compare_temporal_point64_v2 as proof
import cv2
import numpy as np
import score_saved_endpoint_mask_priority as saved
import torch

WORK = Path(__file__).resolve().parent
REPO = WORK.parent.parent
LAB = REPO / "labs/20260909_three_body"
if str(LAB) not in sys.path:
    sys.path.insert(1, str(LAB))

import observations  # noqa: E402
import renderer as rendering  # noqa: E402

require = saved.require
FRAMES = (90141, 90258, 90259)
VIEWS = ("top_cam", "side1_cam", "side2_cam")
MODES = ("detectron2", "top_contrast")
STAGE = "saved_top_foreground_image_term_comparison_v1"
PRODUCER = WORK / "nodewise_body_row_negative_ten_second_continuation_v1"
PRODUCER_SHA = "2774d83411c7b2a28a7d6bfc42a1ca424d1409c8f431cce0b3b6c64adf39ac44"
PHOTOS = WORK / "endpoint_negative_native_photos_v1"
PHOTO_SHA = "dac90d016b249e45fd63d0030ec0f4b041a1b7f414f46cd93049ba29acf67256"
MASKS = WORK / "nodewise_body_mask_classification_all_cases_v1"
MASK_SHA = "12ac9157d89c1ad46b9201e913fa39bc43a6d5f55361493fcfdfecba3f021189"
EXPECTED_COUNTS = {90141: 15, 90258: 22, 90259: 22}
BASELINE_ATOL = 1e-9
BASELINE_RTOL = 1e-8
PINS = {
    "work/three_body/score_saved_endpoint_mask_priority.py": (
        "0f1175ca074661fc495312455c4f2b9f56c2ff99374143b0497dc9784a899015"
    ),
    "work/three_body/compare_temporal_point64_v2.py": (
        "2461f383d7c69223c98cb9371058373361a7fb30a409a37d617b60fc2db53790"
    ),
    "labs/20260909_three_body/observations.py": (
        "f1efcd27b31e63ce693229c9e765152acfc7984b3b8839f1792bcc2efd9cb5df"
    ),
    "labs/20260909_three_body/renderer.py": (
        "dfb585e9d51214a2a3c9447d4bc81da8af80d3451e8fc3ef7a44b1a841732e1c"
    ),
    "src/medaka3d/calib.py": "60849004cb53febe69f437e7afa8b18cb2b65213ba76ce7c921f7022d2be4274",
}


def source_code(inputs, guards):
    for relative, digest in PINS.items():
        saved.capture(REPO / relative, digest, inputs, guards)
    require(
        Path(observations.__file__).resolve() == LAB / "observations.py"
        and Path(rendering.__file__).resolve() == LAB / "renderer.py",
        "frozen sampling/renderer import differs",
    )
    # Record the actual imported helper closure, without importing any tracker.
    for module in tuple(sys.modules.values()):
        value = getattr(module, "__file__", None)
        if value:
            path = Path(value).resolve()
            if (
                path.suffix == ".py"
                and path.is_relative_to(REPO)
                and ".venv" not in path.parts
                and path.is_file()
            ):
                saved.capture(path, proof.sha(path), inputs, guards)


def load_headers(inputs, guards):
    producer = saved.successful(PRODUCER, "tracking.provenance.json", PRODUCER_SHA, inputs, guards)
    photo = saved.successful(PHOTOS, "preparation.provenance.json", PHOTO_SHA, inputs, guards)
    masks = saved.successful(MASKS, "classification.provenance.json", MASK_SHA, inputs, guards)

    def document(root, seal, name):
        ref = saved.member(root, seal, name)
        return saved.read_json(ref["path"], ref["sha256"], inputs, guards)

    plan = document(PRODUCER, producer, "plan.json")
    summary = document(PRODUCER, producer, "summary.json")
    index = document(PRODUCER, producer, "input_index.json")
    pp = document(PHOTOS, photo, "plan.json")
    photo_records = document(PHOTOS, photo, "frame_evidence.json")["records"]
    mp = document(MASKS, masks, "plan.json")
    ms = document(MASKS, masks, "summary.json")
    signatures = document(MASKS, masks, "frame_masks_signatures.json")
    require(
        plan["process_interval"] == [89640, 91300]
        and plan["retained_interval"] == [49800, 91300]
        and summary["status"] == "ok"
        and summary["finish_calls"] == 1
        and summary["native_frames"] == 1660
        and summary["finish_matches_checkpoint_backtrace"] is True,
        "completed negative Nodewise91300 producer required",
    )
    require(
        pp["producer_sha256"] == PRODUCER_SHA
        and pp["frames"] == list(FRAMES)
        and pp["original_binding"] == plan["original_binding"] == index["original_binding"],
        "native photographs use a different producer/input",
    )
    chunk = index["prefix_chunks"][24]
    require(chunk == pp["source_chunk"], "photograph consumed chunk differs")
    require(
        (chunk["index"], chunk["start"], chunk["stop"]) == (24, 89640, 91300), "chunk24 required"
    )
    for view in VIEWS:
        captured = ms["captured_mask_records"]["chunk24_" + view]
        original = chunk["output_hashes"][view + "_masks"]
        require(
            captured["path"] == original["path"] and captured["sha256"] == original["sha256"],
            "saved frame masks came from a different consumed archive",
        )
    rig_refs = mp["registered_calibration"]
    shared = plan["original_binding"]["inputs"]["seed"]["SL_and_calibration_sha256"]
    require(len(rig_refs) == 4, "registered four-file rig required")
    require(all(shared.get(p) == h for p, h in rig_refs.items()), "registered rig differs")
    config = plan["tracker_context"]["tracker_config"]
    require(
        config == summary.get("tracker_context", plan["tracker_context"])["tracker_config"],
        "config differs",
    )
    require(
        config["mode"] == "detectron2"
        and config["mask_weight"] == 3.0
        and [config[k] for k in ("positive_samples", "near_samples", "global_samples")]
        == [384, 384, 256]
        and config["width_ratio"] == 0.06,
        "fixed image configuration differs",
    )
    refs = {}
    for frame in FRAMES:
        refs[frame] = dict(
            companion=saved.member(PRODUCER, producer, f"diagnostics/f{frame}.json"),
            archive=saved.member(PRODUCER, producer, f"diagnostics/f{frame}.npz"),
        )
    for record in photo_records:
        expected = saved.member(
            PHOTOS, photo, f"native/frame_{record['frame_idx']}_{record['view']}.png"
        )
        require(record["image"] == expected, "photo output descriptor differs")
    require(
        len(photo_records) == 9
        and {(r["frame_idx"], r["view"]) for r in photo_records}
        == {(f, v) for f in FRAMES for v in VIEWS},
        "nine unique native images required",
    )
    return dict(
        bank_references=refs,
        rig_refs=rig_refs,
        config=config,
        photo_records=photo_records,
        mask_signatures=signatures,
        mask_archive=saved.member(MASKS, masks, "frame_masks.npz"),
        top_background=chunk["output_hashes"]["top_cam_background"],
        consumed_chunk=chunk,
    )


def image(ref, inputs, guards, signature=None):
    data = saved.capture(ref["path"], ref["sha256"], inputs, guards)
    result = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    require(
        result is not None and result.shape == (1200, 1200, 3) and result.dtype == np.uint8,
        "native uint8 BGR image required",
    )
    if signature is not None:
        require(saved.signature(result) == signature, "native BGR signature differs")
    result.flags.writeable = False
    return result


def load_frame(context, archive, frame, background, inputs, guards):
    masks, images = {}, {}
    for view in VIEWS:
        key = f"f{frame}_{view}"
        ref = context["mask_signatures"][key]
        value = archive[key]
        require(
            ref["frame_idx"] == frame
            and ref["view"] == view
            and type(ref["available"]) is bool
            and value.dtype == bool
            and value.ndim == 3
            and value.shape[1:] == (1200, 1200)
            and saved.signature(value) == ref["signature"]
            and ref["original_mask_sha256"] == ref["signature"]["sha256"],
            "saved native mask signature/availability differs",
        )
        value.flags.writeable = False
        masks[view] = value if ref["available"] else None
        record = next(
            r for r in context["photo_records"] if r["frame_idx"] == frame and r["view"] == view
        )
        images[view] = image(record["image"], inputs, guards, record["bgr_signature"])
    # Sampling does not consume detections. No fabricated SLEAP rows are supplied.
    return observations.FrameObservation(frame, {}, images, masks, {"top_cam": background})


def sample_modes(frame, config):
    counts = [config[k] for k in ("positive_samples", "near_samples", "global_samples")]
    targets = {mode: observations.sample_pixels(frame, mode, *counts) for mode in MODES}
    a, b = (targets[mode] for mode in MODES)
    for index, view in enumerate(VIEWS[1:], 1):
        require(
            np.array_equal(a.pixels[view], b.pixels[view])
            and np.array_equal(a.target[index], b.target[index])
            and np.array_equal(a.weights[index], b.weights[index])
            and np.array_equal(a.foreground[index], b.foreground[index]),
            "side pixels/targets/weights changed between modes",
        )
    return targets


def image_terms(poses, target, rig, config):
    require(
        poses.dtype == np.float64 and np.isfinite(poses).all(), "finite saved FP64 poses required"
    )
    renderer = rendering.RefractiveFishRenderer(
        rig,
        target.pixels,
        rendering.RendererConfig(width_ratio=config["width_ratio"]),
        device="cpu",
        dtype=torch.float64,
    )
    with torch.no_grad():
        alpha = renderer.render(poses).union_alpha
        expected = torch.as_tensor(target.target, dtype=torch.float64, device="cpu")
        weight = (
            torch.as_tensor(target.weights, dtype=torch.float64, device="cpu") * renderer.ray_valid
        )
        residual = (alpha - expected) ** 2 * weight
        cost = residual.sum((1, 2)) / 3  # exact source expression; parts[:,4] is unweighted
        per_view = residual.sum(2) / 3
        p, n = config["positive_samples"], config["near_samples"]
        strata = (
            torch.stack(
                (
                    residual[:, :, :p].sum(2),
                    residual[:, :, p : p + n].sum(2),
                    residual[:, :, p + n :].sum(2),
                ),
                dim=-1,
            )
            / 3
        )
    arrays = dict(
        silhouette=cost.numpy().copy(),
        per_view_silhouette=per_view.numpy().copy(),
        per_view_positive_near_global=strata.numpy().copy(),
        sampled_pixels=np.stack([target.pixels[v] for v in VIEWS]),
        sampled_target=target.target.copy(),
        sampled_weights=target.weights.copy(),
        sampled_foreground=target.foreground.copy(),
        ray_valid=renderer.ray_valid.numpy().copy(),
        ray_origins=renderer.ray_origins.numpy().copy(),
        ray_directions=renderer.ray_directions.numpy().copy(),
    )
    require(all(np.isfinite(v).all() for v in arrays.values()), "nonfinite renderer image term")
    return arrays


def compare_bank(bank, frame, rig, config):
    original_signatures = {k: saved.signature(v) for k, v in bank.items()}
    targets = sample_modes(frame, config)
    evaluated = {
        mode: image_terms(bank["candidate_pose_xyz"], targets[mode], rig, config) for mode in MODES
    }
    a, b = (evaluated[mode] for mode in MODES)
    # Independent reconstruction per mode: never reuse changed top pixel rays.
    for field in ("ray_valid", "ray_origins", "ray_directions", "sampled_pixels"):
        require(np.array_equal(a[field][1:], b[field][1:]), "side ray reconstruction differs")
    require(
        np.array_equal(a["per_view_silhouette"][:, 1:], b["per_view_silhouette"][:, 1:]),
        "side image costs changed between modes",
    )
    stored = bank["original_five_loss_parts"][:, 4]
    error = a["silhouette"] - stored
    tolerance = BASELINE_ATOL + BASELINE_RTOL * np.abs(stored)
    reproduced = np.abs(error) <= tolerance
    delta = config["mask_weight"] * (b["silhouette"] - a["silhouette"])
    local = bank["adjusted_local_cost"] + delta
    path = bank["parent_path_cost"] + local
    arrays = dict(
        bank,
        baseline_cpu_minus_saved_silhouette=error,
        baseline_absolute_tolerance=tolerance,
        baseline_within_tolerance=reproduced,
        weighted_image_delta=delta,
        counterfactual_local_cost=local,
        counterfactual_path_cost=path,
    )
    for mode, values in evaluated.items():
        arrays.update({mode + "__" + k: v for k, v in values.items()})
    ranked = {}
    # If the old CPU image term cannot be reproduced, preserve measurements but
    # refuse to report alternative winners from an unqualified replacement.
    if reproduced.all():
        for name, costs in (
            ("saved_local", bank["adjusted_local_cost"]),
            ("saved_path", bank["adjusted_path_cost"]),
            ("counterfactual_local", local),
            ("counterfactual_path", path),
        ):
            order = np.argsort(costs, kind="stable")
            arrays[name + "_order"] = order
            ranked[name] = dict(
                first_index=int(order[0]),
                minimum=float(costs[order[0]]),
                exact_minimum_indices=np.flatnonzero(costs == costs[order[0]]).tolist(),
            )
    require(
        original_signatures == {k: saved.signature(v) for k, v in bank.items()},
        "saved bank modified",
    )
    return arrays, dict(
        frame_idx=frame.frame_idx,
        candidate_count=len(stored),
        baseline_CPU_CUDA_within_declared_tolerance=bool(reproduced.all()),
        baseline_max_absolute_error=float(np.abs(error).max()),
        baseline_bit_exact=bool(np.array_equal(a["silhouette"], stored)),
        same_seed_and_sample_counts=True,
        side_samples_rays_costs_exactly_equal=True,
        top_sample_coordinates_equal=bool(
            np.array_equal(a["sampled_pixels"][0], b["sampled_pixels"][0])
        ),
        top_rays_reconstructed_separately=True,
        rankings=ranked,
        counterfactual_is_fixed_parent_one_step_not_final91300_selection=True,
    )


def run(output):
    output = Path(output).resolve()
    require(output.is_relative_to(WORK) and not output.exists(), "fresh workspace output required")
    torch.set_num_threads(1)
    inputs, guards, started = {}, {}, proof.utc()
    out = proof.Output(output)
    try:
        source_code(inputs, guards)
        context = load_headers(inputs, guards)
        rig = saved.load_rig(context, inputs, guards)
        background = image(context["top_background"], inputs, guards)
        ref = context["mask_archive"]
        payload = saved.capture(ref["path"], ref["sha256"], inputs, guards)
        results = []
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            require(
                len(archive.files) == len(set(archive.files))
                and set(archive.files) == set(context["mask_signatures"]),
                "saved mask membership differs",
            )
            for native in FRAMES:
                frame = load_frame(context, archive, native, background, inputs, guards)
                bank, kinds = saved.load_bank(
                    context["bank_references"][native], native, inputs, guards
                )
                require(
                    len(bank["candidate_index"]) == EXPECTED_COUNTS[native],
                    "complete fixed bank required",
                )
                arrays, result = compare_bank(bank, frame, rig, context["config"])
                result["candidate_kinds"] = kinds
                result["bank_reference"] = context["bank_references"][native]
                result["arrays"] = {k: saved.signature(v) for k, v in arrays.items()}
                out.write(f"cases/f{native}.npz", arrays)
                out.write(f"cases/f{native}.json", result)
                results.append(result)
                del frame, bank, arrays
        out.write(
            "plan.json",
            dict(
                schema=STAGE,
                frames=list(FRAMES),
                modes=list(MODES),
                config=context["config"],
                producer_sha256=PRODUCER_SHA,
                photo_sha256=PHOTO_SHA,
                masks_sha256=MASK_SHA,
                registered_calibration=context["rig_refs"],
                top_background=context["top_background"],
                baseline_atol=BASELINE_ATOL,
                baseline_rtol=BASELINE_RTOL,
                tolerance_is_declared_engineering_check_not_prior_CPU_CUDA_qualification=True,
                all_fitted_candidates=True,
                mode_changes_top_sampling_pools_and_rng_consumption=True,
                no_new_background_created=True,
                background_median_native_frames=list(range(5000, 45001, 5000)),
                background_introduction_period_and_unlabeled_foreground_limit_retained=True,
                other_costs_and_candidate_poses_fixed=True,
                retained_parent_costs_fixed=True,
                delta_policy="saved local + 3*(contrast CPU image term - D2 CPU image term)",
                original_sixth_term_preserved=True,
                five_parts_not_recombined=True,
                original_video_SL_full_candidate_checkpoint_GPU_refit_tracking_reads_or_calls=False,
            ),
        )
        accepted = all(r["baseline_CPU_CUDA_within_declared_tolerance"] for r in results)
        out.write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok" if accepted else "baseline_reproduction_failed",
                candidate_count=sum(r["candidate_count"] for r in results),
                frames=list(FRAMES),
                cases=[
                    {
                        k: v
                        for k, v in r.items()
                        if k not in ("arrays", "candidate_kinds", "bank_reference")
                    }
                    for r in results
                ],
                mask_archives_read=1,
                frame_mask_keys_decoded=9,
                native_PNG_decodes=9,
                background_PNG_decodes=1,
                registered_rig_JSONs=4,
                CPU_float64=True,
                CPU_intraop_threads=torch.get_num_threads(),
                refitted=False,
                physical_accuracy_claimed=False,
                selected_development_frames_not_holdout=True,
                threshold_or_weight_search=False,
                automatic_tracking_adoption=False,
            ),
        )
        require(accepted, "saved CUDA image term outside declared CPU reproduction tolerance")
        require(
            all(saved.stamp(Path(p)) == s for p, s in guards.items()),
            "input changed during analysis",
        )
        out.final_guard()
        seal = dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=proof.utc(),
            inputs=inputs,
            outputs=out.records,
        )
    except BaseException as error:
        seal = dict(
            stage=STAGE,
            status="failed",
            started_at=started,
            finished_at=proof.utc(),
            inputs=inputs,
            outputs=out.records,
            error=repr(error),
        )
        (output / "comparison.provenance.json").write_bytes(proof.json_bytes(seal))
        raise
    (output / "comparison.provenance.json").write_bytes(proof.json_bytes(seal))
    return seal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=WORK / STAGE)
    run(parser.parse_args().output)
