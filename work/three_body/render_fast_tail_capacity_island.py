"""Nine native raw decodes for a three-frame, two-method capacity-island diagnosis.

This is a posthoc model diagnostic, never anonymous human truth or an ID audit.
Masks are bound to the successful Original run's consumed chunk. The incomplete
FastTail five-minute process is not represented as having a final mask-input seal.
"""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path

import compare_fast_tail_continuation as comparison
import cv2
import map_long_review_to_final as legacy
import matplotlib
import node_geometry
import numpy as np
import prepare_sequence
import render_tail_shape_raw_review as drawing

matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK, REPO, proof = comparison.WORK, comparison.REPO, comparison.proof
require, raw, native = comparison.require, legacy.raw, legacy.native
FRAMES = (77709, 77723, 77880)
HORIZON = 86320
STAGE = "fast_tail_capacity_island_raw_diagnosis_v1"
BODY = WORK / "fast_tail_two_minute_continuation_comparison_v1"
BODY_SHA = "b8e74089939e332833c533c44dfe16dc47c4f03eda204380a5d079e7fe2b8fb3"
KEYS = ("frame_idx", "pose_xyz", "sources", "observed", "identity_review")


def legacy_member(seal, root, role, filename, evidence):
    path = native.checked_relative(root, filename)
    row = seal["outputs"][role]
    require(
        Path(row["path"]).resolve() == path and path.stat().st_size == row["size"],
        "Original output path/size differs",
    )
    evidence.add(path, row["sha256"])
    return path


def load(evidence):
    seal = comparison.json_input(BODY / "comparison.provenance.json", evidence, BODY_SHA)
    require(
        seal["stage"] == comparison.STAGE and seal["status"] == "ok", "successful2min comparison"
    )
    plan = proof.read_json(comparison.custom_member(seal, BODY, "plan.json", evidence))
    summary = proof.read_json(comparison.custom_member(seal, BODY, "summary.json", evidence))
    require(
        summary["status"] == "ok"
        and summary["horizon_stop"] == HORIZON
        and summary["candidate_references"] == plan["candidate_references"]
        and summary["whole_five_minute_process_complete"] is False,
        "fixed completed2min comparison scope differs",
    )
    new = comparison.milestone_metadata(HORIZON, plan["milestone_sha256"], evidence)
    original = comparison.original_metadata(new, HORIZON, evidence)
    for module in (comparison, legacy, raw, native, node_geometry, prepare_sequence, drawing):
        path = Path(module.__file__).resolve()
        digest = seal["inputs"].get(str(path))
        evidence.add(path, digest)
    rows = {}
    for arm, reference in plan["candidate_references"].items():
        path = raw.local(reference["path"])
        require(
            seal["inputs"].get(str(path)) == reference["sha256"], "compared candidate input differs"
        )
        evidence.add(path, reference["sha256"])
        with np.load(path, allow_pickle=False) as archive:
            arrays = {k: archive[k] for k in KEYS}
        require(
            comparison.previous.signature(arrays) == {k: reference["arrays"][k] for k in KEYS},
            "five candidate field signatures differ",
        )
        rows[arm] = legacy.anchor_arrays(arrays, np.array(FRAMES, np.int64))
    root = comparison.ORIGINALS[HORIZON]["root"]
    producer = comparison.json_input(
        root / "continuation.provenance.json", evidence, comparison.ORIGINALS[HORIZON]["seal"]
    )
    consumed = proof.read_json(
        legacy_member(producer, root, "consumed_chunks", "consumed_chunks.json", evidence)
    )
    chunk = raw.select_chunk(consumed, 49800, HORIZON, FRAMES[0], FRAMES[-1] + 1)
    binding = producer["params"]["bindings"]
    require(
        binding == new["binding"] == original["binding"], "same registered source context required"
    )
    fixed = binding["inputs"]
    attempt = native.checked_relative(raw.local(fixed["input_stage_root"]), chunk["attempt_path"])
    manifest = comparison.json_input(attempt / "manifest.json", evidence, chunk["manifest_sha256"])
    detector = comparison.json_input(
        attempt / "staging.provenance.json", evidence, chunk["provenance_sha256"]
    )
    require(
        detector["stage"] == "prepare_independent_mask_chunk"
        and detector["status"] == "ok"
        and detector["params"]["binding"] == fixed["detector_binding"]
        and manifest["protocol_sha256"] == fixed["protocol_sha256"],
        "detector source context differs",
    )
    raw.validate_source(manifest, chunk, fixed["detector_binding"])
    require(
        {
            k: dict(path=str(Path(v["path"]).resolve()), sha256=v["sha256"])
            for k, v in detector["outputs"].items()
        }
        == chunk["output_hashes"],
        "consumed detector output membership differs",
    )
    videos, packed = {}, {}
    for view in legacy.VIEWS:
        descriptor = manifest["raw"][view]
        path = native.checked_relative(attempt, descriptor["path"])
        link = chunk["raw_hardlinks"][view]
        canonical = raw.local(link["source_path"])
        stat = path.stat()
        require(
            Path(link["path"]).resolve() == path
            and os.path.samefile(path, canonical)
            and (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
            == (link["device"], link["inode"], link["bytes"], link["mtime_ns"])
            and fixed["detector_inputs"]["video:" + view]
            == dict(path=str(canonical), sha256=descriptor["sha256"])
            and link["sha256"] == descriptor["sha256"],
            "original hardlink/source identity differs",
        )
        for category, suffix in (("raw", "_raw"), ("masks", "_masks")):
            desc = manifest[category][view]
            payload = native.checked_relative(attempt, desc["path"])
            role = detector["outputs"][view + suffix]
            require(
                Path(role["path"]).resolve() == payload
                and role["sha256"] == desc["sha256"]
                and role["size"] == payload.stat().st_size,
                "native payload output differs",
            )
            evidence.add(payload, desc["sha256"])
            if category == "masks":
                with np.load(payload, allow_pickle=False) as archive:
                    packed[view] = {k: archive[k] for k in archive.files}
                require(
                    np.array_equal(packed[view]["frame_idx"], manifest["frames"]),
                    "mask native frames differ",
                )
        videos[view] = path
    shared = fixed["seed"]["SL_and_calibration_sha256"]
    require(shared == plan["shared_SL_calibration_sha256"], "compared SL/calibration files differ")
    rig, points, _ = legacy.load_anchor_geometry(
        shared, new["metadata"]["context"], np.array(FRAMES), evidence
    )
    return dict(
        rows=rows,
        plan=plan,
        chunk=chunk,
        manifest=manifest,
        videos=videos,
        packed=packed,
        rig=rig,
        points=points,
    )


def finite_json(values):
    values = np.asarray(values)
    return np.where(np.isfinite(values), values, None).tolist()


def metrics(pose, observed, sources, projected, masks, points, rig):
    rays = node_geometry.RayObservations.from_points(rig, points)
    supported, actual, distance = node_geometry.supported_nodes(
        pose, sources, rays, rig, maximum_ray_mm=2.0, maximum_pixel_error=12.0
    )
    require(
        np.array_equal(supported, sources) and np.array_equal(actual, observed),
        "native support differs",
    )
    result = dict(
        observed=observed.tolist(),
        sources=sources.tolist(),
        assigned_ray_distance_mm=finite_json(distance),
        views={},
    )
    for vi, view in enumerate(legacy.VIEWS):
        uv, detections = projected[view], points[view]
        union = (
            masks[view].any(axis=0)
            if masks[view] is not None and len(masks[view])
            else np.zeros((1200, 1200), bool)
        )
        distance_to_mask = cv2.distanceTransform(
            (~union).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )
        details = []
        for slot in range(3):
            for node in range(3):
                nearest = np.linalg.norm(detections[:, node] - uv[slot, node], axis=-1)
                valid = np.flatnonzero(np.isfinite(nearest))
                best = int(valid[np.argmin(nearest[valid])]) if len(valid) else None
                xy = np.rint(uv[slot, node]).astype(int)
                inside = bool(
                    np.isfinite(uv[slot, node]).all() and np.all(xy >= 0) and np.all(xy < 1200)
                )
                maskids = (
                    []
                    if not inside or masks[view] is None
                    else np.flatnonzero(masks[view][:, xy[1], xy[0]]).tolist()
                )
                details.append(
                    dict(
                        slot=slot,
                        node=("mouth", "chest", "tail")[node],
                        observed=bool(observed[slot, node]),
                        assigned_row=int(sources[slot, vi, node]),
                        projected_uv=finite_json(uv[slot, node]),
                        nearest_SLEAP_row=best,
                        nearest_SLEAP_px=None if best is None else float(nearest[best]),
                        mask_instance_indices=maskids,
                        union_distance_px=float(distance_to_mask[xy[1], xy[0]])
                        if inside and union.any()
                        else None,
                    )
                )
        result["views"][view] = details
    return result


def render(frame, images, masks, points, projections, rows, t):
    fig, axes = plt.subplots(3, 3, figsize=(18, 17), dpi=160)
    fig.suptitle(
        f"Native {frame} | saved 2-minute horizon {HORIZON} | diagnostic, not ID truth\n"
        "D2 contours white; SLEAP rows dotted. S0 cyan / S1 yellow / S2 magenta are RAW slots.\n"
        "M circle / C square / T triangle: filled = observed in >=2 views; open = latent. "
        "Same crop per row.",
        fontsize=12,
    )
    for vi, view in enumerate(legacy.VIEWS):
        allpoints = np.concatenate(
            [points[view].reshape(-1, 2), *(projections[a][view].reshape(-1, 2) for a in rows)]
        )
        visible = allpoints[np.isfinite(allpoints).all(axis=1)]
        lo, hi = np.maximum(0, visible.min(axis=0) - 60), np.minimum(1200, visible.max(axis=0) + 60)
        for col, arm in enumerate(("detections", "original", "fast_tail")):
            ax = axes[vi, col]
            ax.imshow(cv2.cvtColor(images[view], cv2.COLOR_BGR2RGB))
            drawing.draw_observation(
                ax,
                dict(
                    masks=np.empty((0, 1200, 1200), bool) if masks[view] is None else masks[view],
                    detections=points[view],
                ),
            )
            if arm != "detections":
                drawing.draw_pose(ax, projections[arm][view], rows[arm]["observed"][t])
            ax.set_xlim(float(lo[0]), float(hi[0]))
            ax.set_ylim(float(hi[1]), float(lo[1]))
            ax.set_title(f"{view} | {arm}")
            ax.tick_params(labelsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    stream = io.BytesIO()
    fig.savefig(stream, format="png")
    plt.close(fig)
    return stream.getvalue()


def run(destination):
    require(
        destination.is_relative_to(WORK) and not destination.exists(),
        "fresh bounded diagnosis output",
    )
    out, evidence = proof.Output(destination), comparison.producer.original.phase.Digests()
    started, frame_records = proof.utc(), []
    try:
        require(not comparison.producer.torch.cuda.is_initialized(), "CPU diagnostic only")
        evidence.add(Path(__file__).resolve())
        context = load(evidence)
        before = {arm: comparison.previous.signature(rows) for arm, rows in context["rows"].items()}
        for t, frame in enumerate(FRAMES):
            iterator = raw.iter_native(context["videos"], context["manifest"], frame, frame + 1)
            try:
                actual, images = next(iterator)
                require(
                    actual == frame and next(iterator, None) is None, "single native frame required"
                )
            finally:
                iterator.close()
            masks, observation = {}, {}
            for view in legacy.VIEWS:
                packed = context["packed"][view]
                pos = frame - context["chunk"]["start"]
                decoded = prepare_sequence.decode_frame_masks(packed, pos)
                digest = prepare_sequence.array_sha256(decoded)
                require(
                    digest == bytes(packed["frame_mask_sha256"][pos]).decode(),
                    "decoded mask SHA differs",
                )
                masks[view] = decoded if packed["view_available"][pos] else None
                observation[view] = dict(
                    bgr_sha256=prepare_sequence.array_sha256(images[view]),
                    mask_sha256=digest,
                    mask_available=bool(packed["view_available"][pos]),
                    mask_count=len(decoded),
                    SLEAP_points=finite_json(context["points"][frame][view]),
                )
            projections = {
                arm: {
                    v: context["rig"].camera(v).project(rows["pose_xyz"][t]) for v in legacy.VIEWS
                }
                for arm, rows in context["rows"].items()
            }
            diagnostic = {
                arm: metrics(
                    rows["pose_xyz"][t],
                    rows["observed"][t],
                    rows["sources"][t],
                    projections[arm],
                    masks,
                    context["points"][frame],
                    context["rig"],
                )
                for arm, rows in context["rows"].items()
            }
            out.write(
                f"frame_{frame}.png",
                render(
                    frame, images, masks, context["points"][frame], projections, context["rows"], t
                ),
            )
            frame_records.append(dict(frame=frame, observations=observation, methods=diagnostic))
            print(f"Rendered native {frame}; cumulative raw decodes {(t + 1) * 3}/9", flush=True)
        require(
            before == {a: comparison.previous.signature(r) for a, r in context["rows"].items()},
            "candidate rows changed",
        )
        out.write(
            "source_and_frame_evidence.json",
            dict(
                schema=STAGE,
                status="ok",
                frames=list(FRAMES),
                comparison_sha256=BODY_SHA,
                horizon_stop=HORIZON,
                candidate_references=context["plan"]["candidate_references"],
                source_chunk=context["chunk"],
                records=frame_records,
                native_BGR_decodes=9,
                decoded_mask_frames=9,
                observed_source_gate_recomputed=True,
                support_gates=dict(ray_mm=2.0, pixel_error=12.0, minimum_views=2),
                candidate_fields_unchanged=True,
                checkpoint_or_DAG_payloads_opened=False,
                five_minute_process_completion_claimed=False,
                FastTail_consumed_mask_lineage_reaudited=False,
                independent_masks_bound_to_successful_Original_consumed_chunk=True,
                raw_slot_labels_are_physical_IDs=False,
                human_labels_created=False,
                all_three_ID_truth=False,
                scope="Three posthoc diagnostic frames in [77662,77880]; "
                "no dense continuity or ground truth.",
            ),
        )
        evidence.final_guard()
        out.final_guard()
        require(not comparison.producer.torch.cuda.is_initialized(), "diagnostic initialized CUDA")
        seal = dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=out.records,
        )
    except BaseException as error:
        seal = dict(
            stage=STAGE,
            status="failed",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=out.records,
            error=repr(error),
        )
        (destination / "diagnosis.provenance.json").write_bytes(proof.json_bytes(seal))
        raise
    (destination / "diagnosis.provenance.json").write_bytes(proof.json_bytes(seal))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output.resolve())
