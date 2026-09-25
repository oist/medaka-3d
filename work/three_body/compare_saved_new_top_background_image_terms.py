"""Fixed 59-candidate CPU image terms with a separately prepared top background.

D2 and old-background costs come from the successful saved comparison. Only
the new-background mode is rendered, using the unchanged three-view FP64 sum.
Side samples/rays/costs must be byte equal. No refit, new candidates or tracking.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import compare_saved_top_foreground_image_terms as old
import numpy as np

proof, saved, require, torch = old.proof, old.saved, old.require, old.torch
WORK, REPO = old.WORK, old.REPO
FRAMES, VIEWS = old.FRAMES, old.VIEWS
STAGE = "saved_new_top_background_image_term_comparison_v1"
OLD_ROOT = WORK / old.STAGE
OLD_SHA = "fe2b4cd02f4ceb064c853a0d206d2fd5520d08a6adbde885438cb81dc3343a3b"
OLD_SOURCE_SHA = "f064df1b7414c37d3f9fd95bcb6711cc179548b1a485abd6e63d50c817aaae53"
BACKGROUND_ROOT = WORK / "top_background_5to6_minutes_v1"
BACKGROUND_SHA = "07de226b3ad2fe35b533bdd90650c98fbe9e46ba9cada9542699892b9bea142b"
BACKGROUND_SOURCE_SHA = "3c4bd35a83c192cc61082ef783fe5d3859438943d2ea23f62b4b25b6e8b6175c"
BACKGROUND_FRAMES = (49800, 51045, 52290, 53535, 54780, 56025, 57270, 58515, 59760)
BACKGROUND_PNG_SHA = "d196bb88c1bde15dc0ea57a63d2728334f89a9f0f06f82776c9071f18ab9277d"
BACKGROUND_BGR_SHA = "54530d2c8fada6d399ff809dc8a771708ce33bc5bb8cdfd66d3831d2c63471fa"
ARRAY_FIELDS = (
    "silhouette",
    "per_view_silhouette",
    "per_view_positive_near_global",
    "sampled_pixels",
    "sampled_target",
    "sampled_weights",
    "sampled_foreground",
    "ray_valid",
    "ray_origins",
    "ray_directions",
)


def document(root, seal, name, inputs, guards):
    ref = saved.member(root, seal, name)
    return saved.read_json(ref["path"], ref["sha256"], inputs, guards)


def edge(seal, path, digest):
    require(seal["inputs"].get(str(Path(path).resolve())) == digest, "sealed input edge differs")


def source_code(inputs, guards):
    # Only actual pure helper modules; no dynamic imported-tracker inventory.
    pins = {
        **old.PINS,
        "work/three_body/compare_saved_top_foreground_image_terms.py": OLD_SOURCE_SHA,
        "src/medaka3d/types.py": saved.PINS["src/medaka3d/types.py"],
    }
    for relative, digest in pins.items():
        saved.capture(REPO / relative, digest, inputs, guards)
    saved.capture(Path(__file__), proof.sha(Path(__file__)), inputs, guards)


def load_headers(inputs, guards):
    legacy = saved.successful(OLD_ROOT, "comparison.provenance.json", OLD_SHA, inputs, guards)
    edge(legacy, Path(old.__file__), OLD_SOURCE_SHA)
    plan = document(OLD_ROOT, legacy, "plan.json", inputs, guards)
    summary = document(OLD_ROOT, legacy, "summary.json", inputs, guards)
    require(
        plan["schema"] == old.STAGE
        and plan["frames"] == list(FRAMES)
        and plan["modes"] == list(old.MODES)
        and plan["original_sixth_term_preserved"]
        and plan["retained_parent_costs_fixed"]
        and plan["five_parts_not_recombined"],
        "saved fixed-parent image comparison differs",
    )
    config = plan["config"]
    require(
        config["mode"] == "detectron2"
        and config["mask_weight"] == 3.0
        and config["width_ratio"] == 0.06
        and [config[k] for k in ("positive_samples", "near_samples", "global_samples")]
        == [384, 384, 256],
        "fixed renderer/sampling configuration differs",
    )
    require(
        summary["status"] == "ok"
        and summary["frames"] == list(FRAMES)
        and summary["candidate_count"] == 59
        and summary["CPU_float64"]
        and summary["CPU_intraop_threads"] == 1
        and all(r["baseline_CPU_CUDA_within_declared_tolerance"] for r in summary["cases"]),
        "qualified complete59 CPU baseline required",
    )
    photo = saved.successful(
        old.PHOTOS, "preparation.provenance.json", old.PHOTO_SHA, inputs, guards
    )
    masks = saved.successful(
        old.MASKS, "classification.provenance.json", old.MASK_SHA, inputs, guards
    )
    edge(legacy, old.PHOTOS / "preparation.provenance.json", old.PHOTO_SHA)
    edge(legacy, old.MASKS / "classification.provenance.json", old.MASK_SHA)
    require(
        plan["photo_sha256"] == old.PHOTO_SHA and plan["masks_sha256"] == old.MASK_SHA,
        "saved observation sources differ",
    )
    pp = document(old.PHOTOS, photo, "plan.json", inputs, guards)
    records = document(old.PHOTOS, photo, "frame_evidence.json", inputs, guards)["records"]
    require(
        pp["frames"] == list(FRAMES)
        and pp["producer_sha256"] == plan["producer_sha256"] == old.PRODUCER_SHA,
        "same saved negative91300 photograph context required",
    )
    require(
        len(records) == 9
        and {(r["frame_idx"], r["view"]) for r in records}
        == {(f, v) for f in FRAMES for v in VIEWS},
        "nine saved native photographs required",
    )
    for r in records:
        require(
            r["image"]
            == saved.member(old.PHOTOS, photo, f"native/frame_{r['frame_idx']}_{r['view']}.png"),
            "photograph descriptor differs",
        )
        edge(legacy, r["image"]["path"], r["image"]["sha256"])
    mask_ref = saved.member(old.MASKS, masks, "frame_masks.npz")
    edge(legacy, mask_ref["path"], mask_ref["sha256"])
    mask_signatures = document(old.MASKS, masks, "frame_masks_signatures.json", inputs, guards)
    for p, h in plan["registered_calibration"].items():
        edge(legacy, p, h)

    bg = saved.successful(
        BACKGROUND_ROOT, "preparation.provenance.json", BACKGROUND_SHA, inputs, guards
    )
    bp = document(BACKGROUND_ROOT, bg, "plan.json", inputs, guards)
    bs = document(BACKGROUND_ROOT, bg, "summary.json", inputs, guards)
    be = document(BACKGROUND_ROOT, bg, "frame_evidence.json", inputs, guards)
    snapshot = saved.member(BACKGROUND_ROOT, bg, "source_snapshot.py")
    require(
        snapshot["sha256"] == bp["source_sha256"] == BACKGROUND_SOURCE_SHA,
        "new background reviewed source differs",
    )
    saved.capture(snapshot["path"], BACKGROUND_SOURCE_SHA, inputs, guards)
    source_ref = bg["inputs"][str((WORK / "prepare_top_background_5to6_minutes.py").resolve())]
    require(source_ref["sha256"] == BACKGROUND_SOURCE_SHA, "background execution source edge")
    require(
        bp["native_frames"] == bs["native_frames"] == list(BACKGROUND_FRAMES)
        and bp["all_background_samples_before_science"]
        and bs["status"] == "ok"
        and not bs["science_frames_used"]
        and not bs["background_adopted_for_tracking"],
        "separate before-science nine-frame background required",
    )
    video_sha = pp["original_binding"]["inputs"]["detector_binding"]["input_sha256"][
        "video:top_cam"
    ]
    require(
        bp["context"]["inherited_whole_video_sha256"] == video_sha
        and bg["video_reference"]["inherited_sha256"] == video_sha
        and bp["context"]["prior_background_descriptor_only"]["sha256"]
        == plan["top_background"]["sha256"],
        "new background video/old background lineage differs",
    )
    bg_ref = saved.member(BACKGROUND_ROOT, bg, "background_top_cam.png")
    signature = be["background"]["bgr_signature"]
    require(
        bg_ref == be["background"]["image"]
        and bg_ref["sha256"] == BACKGROUND_PNG_SHA
        and signature == bs["median_bgr_signature"]
        and signature["sha256"] == BACKGROUND_BGR_SHA
        and signature["shape"] == [1200, 1200, 3]
        and signature["dtype"] == "|u1",
        "new background PNG/BGR binding differs",
    )
    return dict(
        legacy=legacy,
        legacy_plan=plan,
        config=config,
        photo_records=records,
        rig_refs=plan["registered_calibration"],
        mask_archive=mask_ref,
        mask_signatures=mask_signatures,
        background=bg_ref,
        background_signature=signature,
        background_plan=bp,
    )


def load_case(context, frame, inputs, guards):
    result = document(OLD_ROOT, context["legacy"], f"cases/f{frame}.json", inputs, guards)
    require(
        result["frame_idx"] == frame
        and result["candidate_count"] == old.EXPECTED_COUNTS[frame]
        and result["baseline_CPU_CUDA_within_declared_tolerance"],
        "complete qualified saved bank",
    )
    ref = saved.member(OLD_ROOT, context["legacy"], f"cases/f{frame}.npz")
    data = saved.capture(ref["path"], ref["sha256"], inputs, guards)
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        require(
            len(z.files) == len(set(z.files)) and set(z.files) == set(result["arrays"]),
            "saved case membership",
        )
        arrays = {k: z[k] for k in z.files}
    require(
        {k: saved.signature(v) for k, v in arrays.items()} == result["arrays"],
        "saved case signatures",
    )
    n = result["candidate_count"]
    require(
        np.array_equal(arrays["candidate_index"], np.arange(n))
        and arrays["candidate_pose_xyz"].shape == (n, 3, 3, 3)
        and arrays["candidate_pose_xyz"].dtype == np.float64
        and len(result["candidate_kinds"]) == n
        and arrays["baseline_within_tolerance"].all(),
        "same complete candidate bank required",
    )
    for value in arrays.values():
        value.flags.writeable = False
    return arrays, result, ref


def load_frame(context, archive, native, background, inputs, guards):
    images, masks = {}, {"top_cam": None}
    for view in VIEWS:
        record = next(
            r for r in context["photo_records"] if r["frame_idx"] == native and r["view"] == view
        )
        images[view] = old.image(record["image"], inputs, guards, record["bgr_signature"])
        if view == "top_cam":
            continue  # Frozen top_contrast never uses this instance mask.
        key = f"f{native}_{view}"
        ref, value = context["mask_signatures"][key], archive[key]
        require(
            ref["frame_idx"] == native
            and ref["view"] == view
            and type(ref["available"]) is bool
            and value.dtype == bool
            and value.ndim == 3
            and value.shape[1:] == (1200, 1200)
            and saved.signature(value) == ref["signature"]
            and ref["original_mask_sha256"] == ref["signature"]["sha256"],
            "saved side mask binding",
        )
        value.flags.writeable = False
        masks[view] = value if ref["available"] else None
    return old.observations.FrameObservation(native, {}, images, masks, {"top_cam": background})


def byte_equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def rankings(costs):
    require(costs.ndim == 1 and len(costs) and np.isfinite(costs).all(), "finite cost ranking")
    order = np.argsort(costs, kind="stable")
    return dict(
        order=order.tolist(),
        first_index=int(order[0]),
        minimum=float(costs[order[0]]),
        exact_minimum_indices=np.flatnonzero(costs == costs[order[0]]).tolist(),
        first_to_second_gap=None if len(order) < 2 else float(costs[order[1]] - costs[order[0]]),
    )


def compare_terms(bank, new, weight):
    """Preserve parent+(saved-local+delta) and reuse both old costs literally."""
    require(weight == 3.0, "fixed mask weight required")
    for mode in old.MODES:
        for key in ARRAY_FIELDS:
            if key == "silhouette":
                continue
            axis = 1 if key in ("per_view_silhouette", "per_view_positive_near_global") else 0
            a, b = bank[mode + "__" + key], new[key]
            side = (slice(None), slice(1, None)) if axis == 1 else (slice(1, None),)
            require(byte_equal(a[side], b[side]), "side sample/ray/cost differs: " + key)
    require(all(np.isfinite(v).all() for v in new.values()), "nonfinite new image terms")
    delta = weight * (new["silhouette"] - bank["detectron2__silhouette"])
    local = bank["adjusted_local_cost"] + delta
    path = bank["parent_path_cost"] + local
    require(
        byte_equal(
            bank["adjusted_path_cost"], bank["parent_path_cost"] + bank["adjusted_local_cost"]
        ),
        "saved fixed parent path differs",
    )
    old_delta = weight * (bank["top_contrast__silhouette"] - bank["detectron2__silhouette"])
    require(
        byte_equal(bank["counterfactual_local_cost"], bank["adjusted_local_cost"] + old_delta)
        and byte_equal(
            bank["counterfactual_path_cost"],
            bank["parent_path_cost"] + bank["counterfactual_local_cost"],
        ),
        "stored old-background counterfactual association differs",
    )
    arrays = {k: bank[k] for k in saved.FIELDS}
    arrays.update(
        new_background_weighted_delta=delta,
        new_background_local_cost=local,
        new_background_path_cost=path,
        detectron2_local_cost=bank["adjusted_local_cost"],
        detectron2_path_cost=bank["adjusted_path_cost"],
        old_background_local_cost=bank["counterfactual_local_cost"],
        old_background_path_cost=bank["counterfactual_path_cost"],
    )
    for key, value in new.items():
        arrays["new_background__" + key] = value
    report = {}
    for name, sil, lc, pc in (
        (
            "detectron2",
            bank["detectron2__silhouette"],
            bank["adjusted_local_cost"],
            bank["adjusted_path_cost"],
        ),
        (
            "old_background",
            bank["top_contrast__silhouette"],
            bank["counterfactual_local_cost"],
            bank["counterfactual_path_cost"],
        ),
        ("new_background", new["silhouette"], local, path),
    ):
        per_view = (
            new["per_view_silhouette"]
            if name == "new_background"
            else bank[
                ("detectron2" if name == "detectron2" else "top_contrast") + "__per_view_silhouette"
            ]
        )
        # Alternate reduction is measured only, never substituted into the ranks.
        arrays[name + "_per_view_reduction_minus_full"] = per_view.sum(1) - sil
        arrays[name + "_silhouette"] = sil
        report[name] = dict(local=rankings(lc), fixed_parent_path=rankings(pc))
    old_pixels = bank["top_contrast__sampled_pixels"][0]
    return arrays, dict(
        rankings=report,
        side_samples_rays_strata_costs_byte_equal=True,
        top_old_to_new_sample_positions_changed=int(
            np.any(old_pixels != new["sampled_pixels"][0], axis=1).sum()
        ),
        top_old_to_new_target_entries_changed=int(
            np.count_nonzero(bank["top_contrast__sampled_target"][0] != new["sampled_target"][0])
        ),
        top_old_to_new_weight_entries_changed=int(
            np.count_nonzero(bank["top_contrast__sampled_weights"][0] != new["sampled_weights"][0])
        ),
        full_three_view_reduction_order_unchanged=True,
        maximum_alternative_reduction_absolute_difference={
            name: float(np.abs(arrays[name + "_per_view_reduction_minus_full"]).max())
            for name in report
        },
        numerical_rank_changes_are_not_accuracy_improvements=True,
        fixed_parent_single_frame_not_final_trajectory=True,
    )


def run(output):
    output = Path(output).resolve()
    require(
        output.is_relative_to(WORK) and output != WORK and not output.exists(),
        "fresh workspace output",
    )
    require(not torch.cuda.is_initialized(), "CPU-only diagnostic")
    torch.set_num_threads(1)
    inputs, guards, started = {}, {}, proof.utc()
    out = proof.Output(output)
    try:
        source_code(inputs, guards)
        context = load_headers(inputs, guards)
        rig = saved.load_rig(context, inputs, guards)
        background = old.image(
            context["background"], inputs, guards, context["background_signature"]
        )
        ref = context["mask_archive"]
        payload = saved.capture(ref["path"], ref["sha256"], inputs, guards)
        results = []
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            require(
                len(archive.files) == len(set(archive.files))
                and set(archive.files) == set(context["mask_signatures"]),
                "saved mask snapshot membership",
            )
            for native in FRAMES:
                bank, previous, bank_ref = load_case(context, native, inputs, guards)
                before = {k: saved.signature(v) for k, v in bank.items()}
                frame = load_frame(context, archive, native, background, inputs, guards)
                config = context["config"]
                target = old.observations.sample_pixels(
                    frame,
                    "top_contrast",
                    *[config[k] for k in ("positive_samples", "near_samples", "global_samples")],
                )
                new = old.image_terms(bank["candidate_pose_xyz"], target, rig, config)
                arrays, result = compare_terms(bank, new, config["mask_weight"])
                foreground = old.observations.foreground_contrast(
                    frame.images_bgr["top_cam"], background
                )
                result.update(
                    frame_idx=native,
                    candidate_count=len(bank["candidate_index"]),
                    candidate_kinds=previous["candidate_kinds"],
                    saved_case=bank_ref,
                    baseline_CPU_CUDA_qualification_inherited=previous[
                        "baseline_CPU_CUDA_within_declared_tolerance"
                    ],
                    top_new_foreground_pixels=int(foreground.sum()),
                    old_full_foreground_pixel_count=None,
                    old_full_foreground_pixel_count_reason=(
                        "not saved in the reused result; old background not read"
                    ),
                    arrays={k: saved.signature(v) for k, v in arrays.items()},
                )
                require(
                    before == {k: saved.signature(v) for k, v in bank.items()},
                    "saved bank modified",
                )
                out.write(f"cases/f{native}.npz", arrays)
                out.write(f"cases/f{native}.json", result)
                results.append({k: v for k, v in result.items() if k != "arrays"})
        out.write(
            "plan.json",
            dict(
                schema=STAGE,
                frames=list(FRAMES),
                source_sha256=inputs.copy(),
                old_comparison_sha256=OLD_SHA,
                new_background_sha256=BACKGROUND_SHA,
                config=context["config"],
                background=context["background"],
                background_native_frames=list(BACKGROUND_FRAMES),
                same_saved59_candidates=True,
                old_D2_and_old_background_costs_reused=True,
                original_sixth_and_nonimage_costs_fixed=True,
                delta_policy=(
                    "saved adjusted local + 3*(new-background CPU silhouette - "
                    "saved D2 CPU silhouette); path=parent+local"
                ),
                full_three_view_renderer_expression_reused=True,
                threshold_or_weight_search=False,
            ),
        )
        out.write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                frames=list(FRAMES),
                candidate_count=59,
                cases=results,
                background_adopted_for_tracking=False,
                CPU_float64=True,
                CPU_intraop_threads=torch.get_num_threads(),
                old_case_archives_read=3,
                side_mask_keys_decoded=6,
                mask_snapshot_reads=1,
                native_PNG_reads=9,
                new_background_PNG_reads=1,
                old_background_PNG_reads=0,
                rig_JSON_reads=4,
                old_D2_or_old_background_render_calls=0,
                new_background_render_calls=3,
                original_bank_full_candidate_SL_checkpoint_video_reads=0,
                refit_or_tracking_or_GPU_calls=0,
                numerical_ranks_are_not_physical_accuracy=True,
                selected_development_cases_not_holdout=True,
                physical_ID_truth=False,
            ),
        )
        require(not torch.cuda.is_initialized(), "GPU initialized during CPU diagnostic")
        require(all(saved.stamp(Path(p)) == s for p, s in guards.items()), "input changed")
        out.final_guard()
        status = dict(status="ok")
    except BaseException as error:
        status = dict(status="failed", error=repr(error))
        raise
    finally:
        (output / "comparison.provenance.json").write_bytes(
            proof.json_bytes(
                dict(
                    stage=STAGE,
                    started_at=started,
                    finished_at=proof.utc(),
                    inputs=inputs,
                    outputs=out.records,
                    **status,
                )
            )
        )
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=WORK / STAGE)
    run(parser.parse_args().output)
