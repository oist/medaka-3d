"""Six diagnostic native frames: sealed Dense -6 versus ExpandedSparse -12.

CPU only: 18 requested BGR frames, 18 masks, two compared candidate archives
(five required fields only), three packed-mask archives, registered SL and rig.
No tracking, inference, checkpoint39, full-video hashing or identity truth.
The regression-selected frames are post-hoc checks, not held-out validation.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import compare_expanded_depth_instance_10s as comparison
import numpy as np
import qualify_sparse_instance_saved_10s as observations
import render_instance_rescore_10s_review as previous

proof, require, prior = previous.proof, previous.require, previous.prior
WORK, REPO, VIEWS, KEYS = previous.WORK, previous.REPO, previous.VIEWS, previous.KEYS
STAGE = "expanded_depth_instance_ten_second_raw_review_v1"
COMPARISON_ROOT = WORK / comparison.STAGE
COMPARISON_SEAL_SHA = "bb6002ab37441a86daac6a05ed19176b512601e8ad8c30a1df620fb064848f59"
COMPARISON_SOURCE_SHA = "628178b5083308df12e1a290111e5c97c0bb1dd11f0000d704be4f9ff8f35c72"
HELPER_PINS = {
    "render_instance_rescore_10s_review.py": (
        "0d2e01e4b6bcc99bbeef3e85d7189daa0fe2b6d9623e1efd85b2984e83d13a7d"
    ),
    "qualify_sparse_instance_saved_10s.py": (
        "801ce45028523e77d23196a65222f22afb8db16bc3a7a99594facfac91779b2a"
    ),
}
ARMS = comparison.ARMS
FRAMES = (77413, 77677, 77709, 77723, 77808, 77880)
DEVELOPMENT_FRAMES = (77709, 77723, 77880)
REGRESSION_SELECTION = {
    77413: "S0 mouth: representative of a five-frame direct-support loss",
    77677: "S0 chest/tail: representative of four/five-frame direct-support losses",
    77808: "S0 tail: representative of a four-frame direct-support loss",
}
LABELS = ("Dense instance, lower z = -6 mm", "Expanded sparse, lower z = -12 mm")
EXECUTION_READY = True


def validate_comparison(seal, plan, summary):
    require(
        seal["stage"] == comparison.STAGE and seal["status"] == "ok",
        "successful expanded-depth comparison required",
    )
    for doc in (plan, summary):
        require(
            doc["schema"] == comparison.STAGE
            and doc["native_interval"] == [76360, 78020]
            and doc["native_frames"] == 1660
            and doc["terminal_horizon_by_arm"] == dict.fromkeys(ARMS, 78020),
            "same78020 comparison scope differs",
        )
    require(
        summary["status"] == "ok"
        and summary["candidate_references"] == plan["candidate_references"]
        and set(summary["candidate_references"]) == set(ARMS)
        and summary["all_three_physical_ID_truth"] is False
        and summary["physical_depth_truth"] is False
        and summary["candidate_modified"] is False
        and summary["review_modified"] is False,
        "compared candidate binding or truth scope differs",
    )


def validate_indices(dense, expanded):
    shared = {"original_binding", "prefix_chunks"}
    require(
        set(dense) == shared and set(expanded) == shared | {"rationale_references"},
        "actual producer input-index schema differs",
    )
    require(
        all(dense[key] == expanded[key] for key in shared), "producer observation histories differ"
    )
    chunks = dense["prefix_chunks"]
    require(
        len(chunks) == 17
        and [c["index"] for c in chunks] == list(range(17))
        and chunks[-1]["start"] == 76360
        and chunks[-1]["stop"] == 78020,
        "same complete consumed chunk required",
    )
    return chunks[-1]


def anchor_rows(arrays, reference):
    require(
        set(arrays) == set(KEYS)
        and all(proof.signature(v) == reference["arrays"][k] for k, v in arrays.items()),
        "compared five-field numeric signatures differ",
    )
    require(
        reference["terminal_horizon"] == 78020
        and np.array_equal(arrays["frame_idx"], np.arange(49800, 78020)),
        "compared candidate native horizon differs",
    )
    # The old human-review selector is deliberately limited to three anchors;
    # this renderer owns a separate six-frame selection, without changing it.
    selected = np.array(FRAMES, np.int64)
    positions = np.searchsorted(arrays["frame_idx"], selected)
    require(np.array_equal(arrays["frame_idx"][positions], selected), "six anchors not present")
    rows = {key: value[positions].copy() for key, value in arrays.items()}
    for array in rows.values():
        array.flags.writeable = False
    return rows


def load(evidence):
    comp = previous.read(
        COMPARISON_ROOT / "comparison.provenance.json", evidence, COMPARISON_SEAL_SHA
    )
    plan, summary = [
        json.loads(previous.member(COMPARISON_ROOT, comp, name, evidence).read_text("utf8"))
        for name in ("plan.json", "summary.json")
    ]
    validate_comparison(comp, plan, summary)
    require(
        comp["inputs"].get(str(Path(comparison.__file__).resolve())) == COMPARISON_SOURCE_SHA,
        "successful comparison source edge differs",
    )
    evidence.add(Path(comparison.__file__), COMPARISON_SOURCE_SHA)
    for name, digest in HELPER_PINS.items():
        evidence.add(WORK / name, digest)
    # Current renderer closure is evidence of THIS rendering, not a claim that
    # every rendering helper was consumed by the earlier tracking/comparison.
    code = comparison.source_code()
    for path, digest in code.items():
        evidence.add(path, digest)
    evidence.add(Path(observations.fixed.__file__), observations.FIXED_SHA)
    evidence.add(Path(observations.fixed.source.__file__), observations.fixed.SOURCE_SHA)
    rows, references, producer_seals, indices = {}, {}, {}, []
    for arm, stage in zip(ARMS, (comparison.CONTROL_STAGE, comparison.VARIANT_STAGE), strict=True):
        ref = summary["candidate_references"][arm]
        root = WORK / stage
        path = root / "tracking.provenance.json"
        require(
            comp["inputs"].get(str(path)) == ref["tracking_seal_sha256"],
            "producer successful seal input edge differs",
        )
        seal = previous.read(path, evidence, ref["tracking_seal_sha256"])
        require(
            seal["status"] == "ok" and seal["stage"] == stage, "successful actual producer required"
        )
        source_candidate = root / "candidate.npz"
        require(
            Path(ref["path"]).resolve() == source_candidate
            and comp["inputs"].get(str(source_candidate)) == ref["sha256"]
            and Path(seal["outputs"]["candidate.npz"]["path"]).resolve() == source_candidate
            and seal["outputs"]["candidate.npz"]["sha256"] == ref["sha256"],
            "final producer candidate binding differs",
        )
        index_path = root / "input_index.json"
        require(
            comp["inputs"].get(str(index_path)) == seal["outputs"]["input_index.json"]["sha256"],
            "comparison-bound actual producer index required",
        )
        indices.append(
            json.loads(previous.member(root, seal, "input_index.json", evidence).read_text("utf8"))
        )
        producer_seals[arm] = seal
        name = f"candidates/{arm}.npz"
        candidate_path = previous.member(COMPARISON_ROOT, comp, name, evidence)
        with np.load(candidate_path, allow_pickle=False) as archive:
            require(
                len(archive.files) == len(set(archive.files)) == 21
                and set(archive.files) == set(ref["arrays"]),
                "compared all21 archive membership differs",
            )
            arrays = {key: archive[key] for key in KEYS}
        rows[arm] = anchor_rows(arrays, ref)
        references[arm] = dict(
            comparison_copy=comp["outputs"][name],
            original_final=ref,
            decoded_fields=list(KEYS),
            all21_decoded=False,
        )
    chunk = validate_indices(*indices)
    source = previous.read(
        previous.RAW_ROOT / "diagnosis.provenance.json", evidence, previous.RAW_SHA
    )
    require(
        source["status"] == "ok" and source["stage"] == prior.STAGE,
        "successful old raw evidence required",
    )
    old = json.loads(
        previous.member(
            previous.RAW_ROOT, source, "source_and_frame_evidence.json", evidence
        ).read_text("utf8")
    )
    require(
        old["frames"] == list(DEVELOPMENT_FRAMES)
        and old["source_chunk"] == chunk
        and old["horizon_stop"] == 86320
        and old["native_BGR_decodes"] == 9
        and old["decoded_mask_frames"] == 9,
        "previous three-frame pixel evidence scope differs",
    )
    dense, expanded = (producer_seals[a] for a in ARMS)
    # This frozen reader hashes captured mask bytes once, retains all13 sidecars,
    # loads the same CachedNodes.at selector and verifies SL/calibration hashes.
    nodes, rig, packed, mask_records, loaded_chunk = observations.load_observations(
        dense, indices[0], evidence, {}
    )
    require(loaded_chunk == chunk, "observation helper chose another chunk")
    manifest_path = prior.raw.local(chunk["output_hashes"]["manifest"]["path"])
    manifest = json.loads(manifest_path.read_text("utf8"))
    binding = indices[0]["original_binding"]["inputs"]
    prior.raw.validate_source(manifest, chunk, binding["detector_binding"])
    paths = {
        str(manifest_path): chunk["manifest_sha256"],
        str(manifest_path.parent / "staging.provenance.json"): chunk["provenance_sha256"],
        **binding["seed"]["SL_and_calibration_sha256"],
        **{r["path"]: r["sha256"] for r in mask_records.values()},
    }
    require(
        all(
            dense["inputs"].get(p) == expanded["inputs"].get(p) == digest
            for p, digest in paths.items()
        ),
        "both producers must consume identical observations",
    )
    videos = {}
    for view in VIEWS:
        descriptor = manifest["raw"][view]
        path = prior.native.checked_relative(manifest_path.parent, descriptor["path"])
        link = chunk["raw_hardlinks"][view]
        require(
            source["inputs"].get(str(path)) == descriptor["sha256"] == link["sha256"]
            and chunk["output_hashes"][view + "_raw"]
            == dict(path=str(path), sha256=descriptor["sha256"])
            and binding["detector_inputs"]["video:" + view]
            == dict(path=str(prior.raw.local(link["source_path"])), sha256=descriptor["sha256"]),
            "actual canonical raw/consumed hardlink/SHA lineage differs",
        )
        videos[view] = path
    previous.video_stat_guard(videos, chunk)
    return dict(
        rows=rows,
        references=references,
        old=old,
        nodes=nodes,
        rig=rig,
        packed=packed,
        mask_records=mask_records,
        chunk=chunk,
        manifest=manifest,
        videos=videos,
        source_code=code,
    )


def display_crop(image, masks, points, projections):
    """Identical per-camera crop for both arms and raw/D2 modes; include masks."""
    cloud = [points.reshape(-1, 2), *(p.reshape(-1, 2) for p in projections)]
    if masks is not None and len(masks):
        y, x = np.nonzero(masks.any(0))
        if len(x):
            cloud.append(np.array([[x.min(), y.min()], [x.max(), y.max()]]))
    cloud = np.concatenate(cloud)
    finite = cloud[np.isfinite(cloud).all(1)]
    extent = np.array([image.shape[1], image.shape[0]])
    lo, hi = np.zeros(2), extent.copy()
    if len(finite):
        lower, upper = (
            np.clip(finite.min(0) - 60, 0, extent),
            np.clip(finite.max(0) + 60, 0, extent),
        )
        if np.all(lower < upper):
            lo, hi = lower, upper
    return lo, hi


def draw_panel(ax, image, masks, points, projected, observed, mode):
    require(mode in ("raw", "D2"), "unknown display mode")
    ax.imshow(prior.cv2.cvtColor(image, prior.cv2.COLOR_BGR2RGB))
    if masks is not None:
        if mode == "D2":
            overlay = np.zeros((*image.shape[:2], 4), dtype=np.float32)
            for number, mask in enumerate(masks):
                overlay[mask] = (*prior.plt.get_cmap("tab10")(number % 10)[:3], 0.45)
            ax.imshow(overlay)
        for number, mask in enumerate(masks):
            y, x = np.nonzero(mask)
            if len(x):
                ax.text(
                    float(np.median(x)),
                    float(np.median(y)),
                    f"D2#{number}",
                    color="white",
                    fontsize=8,
                    clip_on=True,
                    bbox=dict(facecolor="black", alpha=0.6, pad=1),
                )
    prior.drawing.draw_observation(
        ax,
        dict(
            masks=np.empty((0, *image.shape[:2]), bool) if masks is None else masks,
            detections=points,
        ),
    )
    for row, nodes in enumerate(points):
        if np.isfinite(nodes[0]).all():
            ax.plot(*nodes[0], marker="+", color="#ff9632", ms=7, mew=1.3)
            ax.annotate(
                f"SL{row}M",
                nodes[0],
                xytext=(3, -8),
                textcoords="offset points",
                color="#ff9632",
                fontsize=7,
                clip_on=True,
            )
    if projected is not None:
        prior.drawing.draw_pose(ax, projected, observed)


def render(frame, images, masks, points, projections, rows, t, mode):
    fig, axes = prior.plt.subplots(3, 3, figsize=(18, 17), dpi=160)
    purpose = "development-selected" if frame in DEVELOPMENT_FRAMES else "regression-selected"
    fig.suptitle(
        f"Native {frame} | {mode} | same terminal 78020 | {purpose} diagnostic\n"
        "Rows: observations / Dense lower z=-6 / Expanded sparse lower z=-12. "
        "Columns: three cameras.\n"
        "S0 cyan / S1 yellow / S2 magenta: anonymous RAW slots; sex and physical IDs unconfirmed.\n"
        "M circle / C square / T triangle: filled = >=2-view direct support, open = latent. "
        "SL#M orange +; D2# = frame-local mask index. Same crop; no truth labels.",
        fontsize=11,
    )
    crops = {}
    try:
        for vi, view in enumerate(VIEWS):
            lo, hi = display_crop(
                images[view], masks[view], points[view], [projections[a][view] for a in ARMS]
            )
            crops[view] = dict(lower_xy=lo.tolist(), upper_xy=hi.tolist())
            for ri, label in enumerate(("Independent observations", *LABELS)):
                arm = ARMS[ri - 1] if ri else None
                ax = axes[ri, vi]
                draw_panel(
                    ax,
                    images[view],
                    masks[view],
                    points[view],
                    projections[arm][view] if arm else None,
                    rows[arm]["observed"][t] if arm else None,
                    mode,
                )
                ax.set_xlim(float(lo[0]), float(hi[0]))
                ax.set_ylim(float(hi[1]), float(lo[1]))
                ax.set_title(f"{view} | {label}", fontsize=10)
                ax.tick_params(labelsize=7)
        fig.tight_layout(rect=(0, 0, 1, 0.91))
        stream = io.BytesIO()
        fig.savefig(stream, format="png")
        return stream.getvalue(), crops
    finally:
        prior.plt.close(fig)


def verify_old_observation(frame, view, image_sha, mask_sha, masks, available, points, old):
    if frame not in DEVELOPMENT_FRAMES:
        return False
    row = next(r for r in old["records"] if r["frame"] == frame)["observations"][view]
    require(
        image_sha == row["bgr_sha256"]
        and mask_sha == row["mask_sha256"]
        and len(masks) == row["mask_count"]
        and available == row["mask_available"]
        and prior.finite_json(points) == row["SLEAP_points"],
        "old raw BGR/mask/SL frame evidence differs",
    )
    return True


def run(output):
    require(EXECUTION_READY, "source review and explicit ROOT render Go required")
    output = output.resolve()
    require(output.is_relative_to(WORK) and not output.exists(), "fresh workspace output required")
    out, evidence, started = proof.Output(output), proof.Evidence(), proof.utc()
    try:
        require(not sys.modules["torch"].cuda.is_initialized(), "CPU-only renderer required")
        context = load(evidence)
        rows, nodes = context["rows"], context["nodes"]
        before = {a: {k: proof.signature(v) for k, v in r.items()} for a, r in rows.items()}
        node_before = {v: [proof.signature(a) for a in pair] for v, pair in nodes.data.items()}
        out.write(
            "plan.json",
            dict(
                schema=STAGE,
                frames=list(FRAMES),
                arms=list(ARMS),
                terminal_horizon=78020,
                comparison_sha256=COMPARISON_SEAL_SHA,
                development_frames=list(DEVELOPMENT_FRAMES),
                regression_selection=REGRESSION_SELECTION,
                selected_frames_are_holdout=False,
                requested_BGR_frames=18,
                target_mask_decodes=18,
                image_outputs=12,
                source_code=context["source_code"],
                raw_slot_labels_are_physical_IDs=False,
                sex_confirmed=False,
                whole_video_rehashes=0,
                checkpoint_or_DAG_reads=0,
                GPU_used=False,
            ),
        )
        records = []
        for t, frame in enumerate(FRAMES):
            iterator = prior.raw.iter_native(
                context["videos"], context["manifest"], frame, frame + 1
            )
            try:
                native, images = next(iterator)
                require(native == frame and next(iterator, None) is None, "single native required")
            finally:
                iterator.close()
            masks, points, observation_rows = {}, {}, {}
            for view in VIEWS:
                pos = frame - context["chunk"]["start"]
                packed = context["packed"][view]
                require(int(packed["frame_idx"][pos]) == frame, "native mask row differs")
                decoded = prior.prepare_sequence.decode_frame_masks(packed, pos)
                digest = prior.prepare_sequence.array_sha256(decoded)
                require(
                    digest == bytes(packed["frame_mask_sha256"][pos]).decode(),
                    "target mask pixel SHA differs",
                )
                available = bool(packed["view_available"][pos])
                masks[view] = decoded if available else None
                points[view] = nodes.at(view, frame)
                image_sha = prior.prepare_sequence.array_sha256(images[view])
                matched = verify_old_observation(
                    frame, view, image_sha, digest, decoded, available, points[view], context["old"]
                )
                observation_rows[view] = dict(
                    bgr_sha256=image_sha,
                    mask_sha256=digest,
                    mask_count=len(decoded),
                    mask_available=available,
                    SL_points_signature=proof.signature(points[view]),
                    SLEAP_points=prior.finite_json(points[view]),
                    prior_raw_pixels_checked=matched,
                )
            projections = {
                a: {v: context["rig"].camera(v).project(row["pose_xyz"][t]) for v in VIEWS}
                for a, row in rows.items()
            }
            methods = {
                a: dict(
                    pose_xyz_mm=row["pose_xyz"][t].tolist(),
                    sources=row["sources"][t].tolist(),
                    observed=row["observed"][t].tolist(),
                    identity_review=row["identity_review"][t].tolist(),
                    projections={v: prior.finite_json(p) for v, p in projections[a].items()},
                )
                for a, row in rows.items()
            }
            crops = None
            for mode in ("raw", "D2"):
                image, current = render(frame, images, masks, points, projections, rows, t, mode)
                require(crops is None or crops == current, "raw/D2 crops differ")
                crops = current
                out.write(f"frame_{frame}_{mode}.png", image)
            records.append(
                dict(
                    frame=frame, observations=observation_rows, methods=methods, display_crops=crops
                )
            )
            print(
                f"Rendered native {frame}; requested BGR/mask frames {(t + 1) * 3}/18", flush=True
            )
        require(
            before == {a: {k: proof.signature(v) for k, v in r.items()} for a, r in rows.items()},
            "candidate anchor rows changed",
        )
        require(
            node_before == {v: [proof.signature(a) for a in p] for v, p in nodes.data.items()},
            "registered SL rows changed",
        )
        for view, record in context["mask_records"].items():
            require(
                observations.file_stat(Path(record["path"])) == record["file_stat"]
                and {k: proof.signature(v) for k, v in context["packed"][view].items()}
                == record["arrays"],
                "captured mask file/stat/sidecars changed",
            )
        previous.video_stat_guard(context["videos"], context["chunk"])
        out.write(
            "source_and_frame_evidence.json",
            dict(
                schema=STAGE,
                status="ok",
                frames=list(FRAMES),
                terminal_horizon=78020,
                comparison_sha256=COMPARISON_SEAL_SHA,
                previous_raw_diagnosis_sha256=previous.RAW_SHA,
                candidate_references=context["references"],
                selected_candidate_arrays=before,
                source_chunk=context["chunk"],
                captured_mask_records=context["mask_records"],
                records=records,
                requested_BGR_frames=18,
                target_mask_decodes=18,
                previous_nine_BGR_mask_SL_match=True,
                packed_mask_payload_reads=3,
                images=12,
                whole_video_rehashes=0,
                checkpoint_or_DAG_reads=0,
                GPU_used=False,
                candidate_modified=False,
                review_modified=False,
                physical_ID_truth=False,
                sex_confirmed=False,
                physical_depth_truth=False,
                automatic_adoption=False,
                selected_frames_are_holdout=False,
                limitation=(
                    "Six development/regression-selected frames; "
                    "no dense ID continuity or physical truth."
                ),
            ),
        )
        evidence.final_guard()
        out.final_guard()
        require(not sys.modules["torch"].cuda.is_initialized(), "renderer initialized CUDA")
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
        (output / "render.provenance.json").write_bytes(proof.json_bytes(seal))
        raise
    (output / "render.provenance.json").write_bytes(proof.json_bytes(seal))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
