"""Draw saved versus image-term-counterfactual minima on the same native photos.

This consumes a successful CPU analysis. It does not fit bodies, sample masks,
read original trajectories, or identify the final91300 temporal winner.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import compare_saved_top_foreground_image_terms as measure
import numpy as np
import render_partial_body_bank_comparison as drawing

proof, saved, require = measure.proof, measure.saved, measure.require
WORK, REPO, FRAMES, VIEWS = measure.WORK, measure.REPO, measure.FRAMES, measure.VIEWS
STAGE = "saved_top_foreground_image_term_visual_comparison_v1"
ANALYSIS_SOURCE_SHA = "f064df1b7414c37d3f9fd95bcb6711cc179548b1a485abd6e63d50c817aaae53"
DRAWING_SOURCE_SHA = "7658faac53fa0cb19ccfb6b1323144e26ef4c4ef8d4030fb4182d988af3d684e"
FIELDS = (
    "candidate_pose_xyz",
    "candidate_parent",
    "parent_path_cost",
    "base_local_cost",
    "sixth_instance_term",
    "adjusted_local_cost",
    "adjusted_path_cost",
    "weighted_image_delta",
    "counterfactual_local_cost",
    "counterfactual_path_cost",
)


def document(root, seal, name, inputs, guards):
    ref = saved.member(root, seal, name)
    return saved.read_json(ref["path"], ref["sha256"], inputs, guards)


def load_context(root, digest, inputs, guards):
    root = Path(root).resolve()
    require(
        root.is_relative_to(WORK) and proof.valid_sha(digest), "explicit saved analysis required"
    )
    seal = saved.read_json(root / "comparison.provenance.json", digest, inputs, guards)
    require(
        seal["stage"] == measure.STAGE and seal["status"] == "ok",
        "successful CPU analysis required",
    )
    source = WORK / "compare_saved_top_foreground_image_terms.py"
    require(seal["inputs"].get(str(source)) == ANALYSIS_SOURCE_SHA, "unreviewed analysis source")
    saved.capture(source, ANALYSIS_SOURCE_SHA, inputs, guards)
    plan = document(root, seal, "plan.json", inputs, guards)
    summary = document(root, seal, "summary.json", inputs, guards)
    require(
        plan["schema"] == summary["schema"] == measure.STAGE
        and summary["status"] == "ok"
        and plan["frames"] == summary["frames"] == list(FRAMES)
        and plan["producer_sha256"] == measure.PRODUCER_SHA
        and plan["photo_sha256"] == measure.PHOTO_SHA
        and summary["candidate_count"] == 59,
        "fixed complete three-bank analysis differs",
    )
    photo_path = measure.PHOTOS / "preparation.provenance.json"
    require(seal["inputs"].get(str(photo_path)) == measure.PHOTO_SHA, "analysis used other photos")
    photos = saved.successful(
        measure.PHOTOS, "preparation.provenance.json", measure.PHOTO_SHA, inputs, guards
    )
    photo_records = document(measure.PHOTOS, photos, "frame_evidence.json", inputs, guards)[
        "records"
    ]
    require(
        len(photo_records) == 9
        and {(r["frame_idx"], r["view"]) for r in photo_records}
        == {(f, v) for f in FRAMES for v in VIEWS},
        "nine native photos required",
    )
    rig_refs = plan["registered_calibration"]
    require(
        len(rig_refs) == 4 and all(seal["inputs"].get(p) == h for p, h in rig_refs.items()),
        "analysis rig not bound",
    )
    return dict(
        root=root,
        seal=seal,
        sha256=digest,
        plan=plan,
        summary=summary,
        photo_seal=photos,
        photo_records=photo_records,
        rig_refs=rig_refs,
    )


def select_rows(arrays, case):
    count = case["candidate_count"]
    require(
        count > 0 and arrays["candidate_pose_xyz"].shape == (count, 3, 3, 3),
        "complete fitted poses required",
    )
    require(
        all(np.isfinite(v).all() for v in arrays.values()), "finite saved render values required"
    )
    selections = []
    for label, cost_key, local_key, path_key in (
        ("saved_path", "adjusted_path_cost", "adjusted_local_cost", "adjusted_path_cost"),
        (
            "counterfactual_path",
            "counterfactual_path_cost",
            "counterfactual_local_cost",
            "counterfactual_path_cost",
        ),
        (
            "counterfactual_local",
            "counterfactual_local_cost",
            "counterfactual_local_cost",
            "counterfactual_path_cost",
        ),
    ):
        cost = arrays[cost_key]
        require(cost.shape == (count,), "saved cost axis differs")
        order = np.argsort(cost, kind="stable")
        ties = np.flatnonzero(cost == cost[order[0]]).tolist()
        ranking = case["rankings"][label]
        require(
            ranking["first_index"] == int(order[0])
            and ranking["exact_minimum_indices"] == ties
            and ranking["minimum"] == float(cost[order[0]]),
            "saved minimum/ties differ",
        )
        index = int(order[0])
        selections.append(
            dict(
                mode=label,
                candidate_index=index,
                exact_minimum_indices=ties,
                candidate_parent=int(arrays["candidate_parent"][index]),
                parent_path_cost=float(arrays["parent_path_cost"][index]),
                candidate_kind=case["candidate_kinds"][index],
                path_cost=float(arrays[path_key][index]),
                local_cost=float(arrays[local_key][index]),
                original_base_local_cost=float(arrays["base_local_cost"][index]),
                original_sixth_term=float(arrays["sixth_instance_term"][index]),
                weighted_image_delta=float(arrays["weighted_image_delta"][index]),
                pose_xyz=arrays["candidate_pose_xyz"][index].copy(),
            )
        )
    same = selections[0]["candidate_index"] == selections[1]["candidate_index"]
    return selections, same


def unique_rows(selections):
    """Keep every distinct selected candidate; preserve all mode/tie records separately."""
    seen, result = set(), []
    for row in selections:
        if row["candidate_index"] not in seen:
            result.append(row)
            seen.add(row["candidate_index"])
    return result


def load_case(context, frame, inputs, guards):
    root, seal = context["root"], context["seal"]
    case = document(root, seal, f"cases/f{frame}.json", inputs, guards)
    require(
        case["frame_idx"] == frame and case["baseline_CPU_CUDA_within_declared_tolerance"] is True,
        "qualified native case required",
    )
    ref = saved.member(root, seal, f"cases/f{frame}.npz")
    payload = saved.capture(ref["path"], ref["sha256"], inputs, guards)
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        require(
            len(archive.files) == len(set(archive.files))
            and set(archive.files) == set(case["arrays"]),
            "saved case membership differs",
        )
        fields = list(FIELDS)
        if frame == 90258:
            fields += [
                mode + "__" + key
                for mode in measure.MODES
                for key in ("sampled_pixels", "sampled_target", "sampled_weights", "ray_valid")
            ]
        arrays = {k: archive[k] for k in fields}
    require(
        all(saved.signature(a) == case["arrays"][k] for k, a in arrays.items()),
        "selected output signature differs",
    )
    rows, same = select_rows({k: arrays[k] for k in FIELDS}, case)
    samples = None
    if frame == 90258:
        samples = {
            mode: {
                key: arrays[mode + "__" + key][0].copy()
                for key in ("sampled_pixels", "sampled_target", "sampled_weights", "ray_valid")
            }
            for mode in measure.MODES
        }
    return rows, same, samples


def draw_case(frame, selections, same, images):
    shown = unique_rows(selections)
    fig, axes = drawing.prior.plt.subplots(
        len(shown), 3, figsize=(18, 5 * len(shown) + 2), squeeze=False
    )
    crops = {}
    try:
        for vi, view in enumerate(VIEWS):
            empty = np.empty((0, 3, 2))
            points = np.stack([s["projections"][view] for s in shown])
            lo, hi = drawing.visual.display_crop(images[view], None, empty, points)
            crops[view] = dict(lo=lo.tolist(), hi=hi.tolist())
            for ri, row in enumerate(shown):
                ax = axes[ri, vi]
                drawing.draw_panel(ax, images[view], empty, row["projections"][view])
                ax.set_xlim(lo[0], hi[0])
                ax.set_ylim(hi[1], lo[1])
                title = (
                    f"{view} | {row['mode']} candidate {row['candidate_index']}\n"
                    f"parent {row['candidate_parent']} | {row['candidate_kind']} | "
                    f"path {row['path_cost']:.7g}; local {row['local_cost']:.7g}\n"
                    f"exact minima {row['exact_minimum_indices']}"
                )
                if same and row["candidate_index"] == selections[0]["candidate_index"]:
                    other = selections[1]
                    title += (
                        f"\nSAME candidate: counterfactual path {other['path_cost']:.7g}; "
                        f"exact minima {other['exact_minimum_indices']}"
                    )
                ax.set_title(title, fontsize=8)
        fig.suptitle(
            f"Native {frame}: saved fitted Nodewise bank, producer ends at91300\n"
            "Fixed-parent ONE-STEP image-cost comparison; not the final91300 path selection.\n"
            "M circle / C square / T triangle: all OPEN fitted points; dashed raw slots. "
            "Cyan S0 / yellow S1 / magenta S2 are NOT physical IDs.\n"
            "Same lossless photographs and crop; no refit, new support, "
            "sex or accuracy certification.\n"
            + " | ".join(
                f"{r['mode']} exact minima {r['exact_minimum_indices']}" for r in selections
            ),
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.84 if len(shown) == 1 else 0.90))
        stream = io.BytesIO()
        fig.savefig(stream, format="png", dpi=160)
        return stream.getvalue(), crops
    finally:
        drawing.prior.plt.close(fig)


def draw_top_samples(image, samples):
    """Show only saved sampled positives, on a separate full-native photograph."""
    fig, axes = drawing.prior.plt.subplots(1, 2, figsize=(16, 9), squeeze=False)
    records = {}
    try:
        for index, mode in enumerate(measure.MODES):
            row, ax = samples[mode], axes[0, index]
            points, target, weights, valid = (
                row[k] for k in ("sampled_pixels", "sampled_target", "sampled_weights", "ray_valid")
            )
            require(
                points.shape == (1024, 2)
                and target.shape == weights.shape == valid.shape == (1024,),
                "fixed saved top sample axes required",
            )
            positive = (target == 1) & (weights > 0)
            used, ignored = positive & valid, positive & ~valid
            ax.imshow(drawing.prior.cv2.cvtColor(image, drawing.prior.cv2.COLOR_BGR2RGB))
            ax.scatter(points[used, 0], points[used, 1], s=3, c="#00ff66", alpha=0.65, linewidths=0)
            if ignored.any():
                ax.scatter(points[ignored, 0], points[ignored, 1], s=7, c="#ff9900", marker="x")
            ax.set_xlim(0, image.shape[1] - 1)
            ax.set_ylim(image.shape[0] - 1, 0)
            distinct = len(np.unique(points[positive], axis=0))
            ax.set_title(
                f"{mode}: positive draws={int(positive.sum())}, distinct pixels={distinct}\n"
                f"valid rays={int(used.sum())}; ignored invalid rays={int(ignored.sum())}",
                fontsize=10,
            )
            records[mode] = dict(
                positive_draws=int(positive.sum()),
                distinct_positive_pixels=distinct,
                valid_positive_rays=int(used.sum()),
                invalid_positive_rays=int(ignored.sum()),
                positive_xy=points[positive].tolist(),
                positive_ray_valid=valid[positive].tolist(),
            )
        fig.suptitle(
            "Native 90258 TOP: saved POSITIVE sampling locations only\n"
            "Green dots: positive targets with valid rays; "
            "orange x: invalid rays (zero cost weight).\n"
            "Same seed/count, different foreground pools and top samples. No fitted-body overlay.\n"
            "These are detector/contrast responses, "
            "not fish labels or evidence of tracking improvement.",
            fontsize=11,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        stream = io.BytesIO()
        fig.savefig(stream, format="png", dpi=160)
        return stream.getvalue(), records
    finally:
        drawing.prior.plt.close(fig)


def run(root, digest, output):
    output = Path(output).resolve()
    require(output.is_relative_to(WORK) and not output.exists(), "fresh workspace output required")
    inputs, guards, started = {}, {}, proof.utc()
    out = proof.Output(output)
    try:
        saved.capture(
            WORK / "render_partial_body_bank_comparison.py", DRAWING_SOURCE_SHA, inputs, guards
        )
        context = load_context(root, digest, inputs, guards)
        for module in tuple(sys.modules.values()):
            name = getattr(module, "__file__", None)
            if name:
                path = Path(name).resolve()
                if (
                    path.suffix == ".py"
                    and path.is_relative_to(REPO)
                    and ".venv" not in path.parts
                    and path.is_file()
                ):
                    saved.capture(path, proof.sha(path), inputs, guards)
        rig = saved.load_rig(context, inputs, guards)
        results, sampling_record = [], None
        for frame in FRAMES:
            rows, same, samples = load_case(context, frame, inputs, guards)
            images = {}
            for view in VIEWS:
                record = next(
                    r
                    for r in context["photo_records"]
                    if r["frame_idx"] == frame and r["view"] == view
                )
                ref = saved.member(
                    measure.PHOTOS, context["photo_seal"], f"native/frame_{frame}_{view}.png"
                )
                require(ref == record["image"], "native photo record differs")
                images[view] = measure.image(ref, inputs, guards, record["bgr_signature"])
            for row in rows:
                row["projections"] = {v: rig.camera(v).project(row["pose_xyz"]) for v in VIEWS}
                require(
                    all(
                        p.shape == (3, 3, 2) and np.isfinite(p).all()
                        for p in row["projections"].values()
                    ),
                    "finite saved-pose projections required",
                )
            png, crops = draw_case(frame, rows, same, images)
            ref = out.write(f"frame_{frame}.png", png)
            if samples is not None:
                sampling_png, sampling = draw_top_samples(images["top_cam"], samples)
                sampling_ref = out.write("frame_90258_top_positive_samples.png", sampling_png)
                sampling_record = dict(frame_idx=frame, image=sampling_ref, modes=sampling)
            results.append(
                dict(
                    frame_idx=frame,
                    image=ref,
                    selections=[
                        {
                            **row,
                            "pose_xyz": row["pose_xyz"].tolist(),
                            "projections": {
                                view: xy.tolist() for view, xy in row["projections"].items()
                            },
                        }
                        for row in rows
                    ],
                    crops=crops,
                    same_candidate=same,
                    displayed_unique_candidates=[r["candidate_index"] for r in unique_rows(rows)],
                    duplicate_candidate_rows_omitted=len(rows) - len(unique_rows(rows)),
                )
            )
        out.write(
            "projection_evidence.json",
            dict(schema=STAGE, cases=results, top_positive_sampling=sampling_record),
        )
        out.write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                comparison_sha256=digest,
                photo_sha256=measure.PHOTO_SHA,
                frames=list(FRAMES),
                PNG_count=len(results) + int(sampling_record is not None),
                body_comparison_PNG_count=len(results),
                duplicate_rows_omitted=sum(r["duplicate_candidate_rows_omitted"] for r in results),
                counterfactual_local_minimum_included=True,
                fixed_parent_one_step_only=True,
                final_temporal_winner_identified=False,
                original_video_SL_masks_fullcandidate_checkpoint_GPU_reads=False,
                raw_slot_labels_are_physical_IDs=False,
                support_recomputed=False,
            ),
        )
        require(
            all(saved.stamp(Path(p)) == s for p, s in guards.items()),
            "input changed while rendering",
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
        (output / "render.provenance.json").write_bytes(proof.json_bytes(seal))
        raise
    (output / "render.provenance.json").write_bytes(proof.json_bytes(seal))
    return seal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", type=Path, default=WORK / measure.STAGE)
    parser.add_argument("--comparison-sha", required=True)
    parser.add_argument("--output", type=Path, default=WORK / STAGE)
    args = parser.parse_args()
    run(args.comparison_root, args.comparison_sha, args.output)
