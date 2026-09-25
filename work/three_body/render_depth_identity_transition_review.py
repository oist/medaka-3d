"""Nine selected native instants: same-final Sparse -6 / Depth -12 body evidence.

CPU only. Read five saved candidate fields, three registered SLEAP caches and
six consumed mask archives. Decode 27 requested BGR frames and mask stacks.
These static, selected observations neither label sex nor certify physical IDs.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import render_fast_tail_five_minute_id_transition as frozen

visual, previous, observations = frozen.visual, frozen.previous, frozen.observations
proof, require, prior = frozen.proof, frozen.require, frozen.prior
WORK, REPO, VIEWS, KEYS = frozen.WORK, frozen.REPO, frozen.VIEWS, frozen.KEYS
STAGE = "depth_identity_transition_raw_review_v1"
FRAMES = (82121, 82690, 82718, 82719, 82720, 82727, 82728, 82750, 83265)
FIRST, START, STOP, CHUNK = 49800, 66400, 116200, 1660
ARMS = ("sparse_instance_rescore", "expanded_depth_instance_rescore")
LABELS = ("Sparse lower z=-6 mm", "Depth lower z=-12 mm")
COMPARISON_ROOT = WORK / "expanded_depth_five_minute_comparison_v1"
COMPARISON_SHA = "556c23675fd298880047210c106f0de095c72e540501716a402e7088bc167713"
COMPARISON_SOURCE_SHA = "db30f8fd32a58a60b77ef1daa50dbf42da7a0d181eb4b43da47cdf2b2ee2694e"
ROOT_NAMES = (
    "sparse_instance_five_minute_continuation_v1",
    "expanded_depth_five_minute_continuation_v1",
)
TRACKING_SHAS = (
    "596b23afe5fb572a66a72b96b0b1f5ba9b3971bfef4dd43cb8ee6b32459e32ad",
    "44da9cd5e55b02992156251e97d3a8e331d535399116bddcbaa3135f31f486ec",
)
CHUNK_SHAS = {
    19: "52d88451c0c1162c004acef567992802e5452ea8505fcd86a848f9ab356074ce",
    20: "02774d64b9c289a40261aa03b5bb679b7706cceeac8aba5c8ed8c6c7655b9db2",
}
PINS = {
    "work/three_body/run_fast_tail_five_minute_identity_beam.py": (
        "9bf62d84cf2eff9cb2dc19ed2dd6beab901633b8221ca6491ab5f0b2bc3a55e1"
    ),
    "work/three_body/render_fast_tail_five_minute_id_transition.py": (
        "169c8a7dcd7e3ab932412a968ddae4e04574d99c4e517ee986cdbe4aa544236e"
    ),
    "work/three_body/render_expanded_depth_10s_review.py": (
        "36eea92fa9126a65495b055a0f9046d88de31382d36535ba45478651b5933e25"
    ),
    "work/three_body/render_instance_rescore_10s_review.py": (
        "0d2e01e4b6bcc99bbeef3e85d7189daa0fe2b6d9623e1efd85b2984e83d13a7d"
    ),
    "work/three_body/compare_temporal_point64_v2.py": (
        "2461f383d7c69223c98cb9371058373361a7fb30a409a37d617b60fc2db53790"
    ),
    "work/three_body/render_fast_tail_capacity_island.py": (
        "79592f20dc8fe24ba07d9b8a63a8a856a09b02724e737838fdea0012df2061f2"
    ),
    "work/three_body/prepare_long_raw_review.py": (
        "c1a799d248f9b0f3b0e90329fb034e73af1a6a2c74fab9270dd8df84d07c66f0"
    ),
    "work/three_body/long_final_native_samples.py": (
        "f9c82e6fc7150edd20f424dd9fa75fda2ef321ad10f41076744ef7a7689e7791"
    ),
    "work/three_body/render_tail_shape_raw_review.py": (
        "0c0f486537cd76d922eda136d047c95d33f00e974d5e0acdc3a93717f83b7490"
    ),
    "work/three_body/qualify_sparse_instance_saved_10s.py": (
        "801ce45028523e77d23196a65222f22afb8db16bc3a7a99594facfac91779b2a"
    ),
    "labs/20260909_three_body/prepare_sequence.py": (
        "959e862cc58c5840a00a763d8a63b635de134492a637ac2f1e8a7131f25178b8"
    ),
    "labs/20260909_three_body/observations.py": (
        "f1efcd27b31e63ce693229c9e765152acfc7984b3b8839f1792bcc2efd9cb5df"
    ),
    "labs/20260909_three_body/chunk_masks.py": (
        "9e08248ea593b4ccb105a6ff351fd85c6db24aa6754413b2df766404cdfe4b81"
    ),
    "src/medaka3d/calib.py": ("60849004cb53febe69f437e7afa8b18cb2b65213ba76ce7c921f7022d2be4274"),
}
EXECUTION_READY = True


def edge(seal, path, digest):
    require(seal["inputs"].get(str(Path(path).resolve())) == digest, "consumed input edge differs")


def document(root, seal, name, evidence):
    return proof.read_json(previous.member(root, seal, name, evidence))


def source_code(evidence, comparison_plan):
    # Preserve the actual comparison's calibration/projection implementation,
    # including its registered dirty calib.py, plus explicit later visual helpers.
    code = dict(comparison_plan["source_sha256"])
    for name, digest in PINS.items():
        path = str(REPO / name)
        require(path not in code or code[path] == digest, "helper disagrees with compared source")
        code[path] = digest
    code[str(Path(__file__).resolve())] = proof.sha(__file__)
    for path, digest in code.items():
        evidence.add(Path(path), digest)
    return code


def select_chunks(index):
    require(
        set(index) == {"original_binding", "prefix_chunks"}, "actual input index schema differs"
    )
    chunks = index["prefix_chunks"]
    require(len(chunks) == 40, "final forty chunks required")
    for i, row in enumerate(chunks):
        require(
            all(type(row[k]) is int for k in ("index", "start", "stop"))
            and (row["index"], row["start"], row["stop"])
            == (i, FIRST + i * CHUNK, FIRST + (i + 1) * CHUNK),
            "consumed native chunk order differs",
        )
    selected = {i: chunks[i] for i in sorted({(f - FIRST) // CHUNK for f in FRAMES})}
    require(set(selected) == {19, 20}, "two fixed source chunks required")
    require(
        all(selected[i]["manifest_sha256"] == CHUNK_SHAS[i] for i in selected), "chunk SHA differs"
    )
    return selected


def frame_chunk(chunks, frame):
    matches = [c for c in chunks.values() if c["chunk"]["start"] <= frame < c["chunk"]["stop"]]
    require(len(matches) == 1, "native frame must have exactly one source chunk")
    return matches[0]


def load_chunk(chunk, binding, tracking, evidence):
    root = prior.native.checked_relative(
        prior.raw.local(binding["input_stage_root"]), chunk["attempt_path"]
    )
    path = prior.raw.local(chunk["output_hashes"]["manifest"]["path"])
    require(path == root / "manifest.json", "consumed manifest location differs")
    for seal in tracking:
        edge(seal, path, chunk["manifest_sha256"])
        edge(seal, root / "staging.provenance.json", chunk["provenance_sha256"])
    manifest = previous.read(path, evidence, chunk["manifest_sha256"])
    staged = previous.read(root / "staging.provenance.json", evidence, chunk["provenance_sha256"])
    require(
        staged["status"] == "ok" and staged["stage"] == "prepare_independent_mask_chunk",
        "chunk stage failed",
    )
    require(
        {k: dict(path=v["path"], sha256=v["sha256"]) for k, v in staged["outputs"].items()}
        == chunk["output_hashes"],
        "consumed chunk outputs differ",
    )
    prior.raw.validate_source(manifest, chunk, binding["detector_binding"])
    require(
        manifest["binding"] == staged["params"]["binding"]
        and manifest["source_interval_half_open"] == [chunk["start"], chunk["stop"]]
        and manifest["frames"] == list(range(chunk["start"], chunk["stop"]))
        and set(manifest["masks"]) == set(VIEWS),
        "source interval or detector differs",
    )
    videos, mask_paths = {}, {}
    for view in VIEWS:
        desc, link = manifest["raw"][view], chunk["raw_hardlinks"][view]
        path = prior.native.checked_relative(root, desc["path"])
        require(
            chunk["output_hashes"][view + "_raw"] == dict(path=str(path), sha256=desc["sha256"])
            and desc["sha256"] == link["sha256"],
            "raw output binding differs",
        )
        for seal in tracking:
            frozen.canonical_video_edge(seal, staged, binding, link, view)
        videos[view] = path  # No video hashing, only frozen native reader + stat guard.
        desc = manifest["masks"][view]
        path = prior.native.checked_relative(root, desc["path"])
        require(
            chunk["output_hashes"][view + "_masks"] == dict(path=str(path), sha256=desc["sha256"]),
            "mask binding differs",
        )
        for seal in tracking:
            edge(seal, path, desc["sha256"])
        mask_paths[view] = path
    return dict(chunk=chunk, manifest=manifest, videos=videos, mask_paths=mask_paths)


def load_headers(evidence):
    seal = previous.read(COMPARISON_ROOT / "comparison.provenance.json", evidence, COMPARISON_SHA)
    require(
        seal["stage"] == "expanded_depth_five_minute_comparison_v1" and seal["status"] == "ok",
        "successful final comparison required",
    )
    edge(seal, WORK / "compare_expanded_depth_five_minutes.py", COMPARISON_SOURCE_SHA)
    plan, summary = [
        document(COMPARISON_ROOT, seal, n + ".json", evidence) for n in ("plan", "summary")
    ]
    require(
        summary["status"] == "ok"
        and plan["candidate_references"] == summary["candidate_references"],
        "comparison references differ",
    )
    for doc in (plan, summary):
        require(
            doc["native_interval"] == [START, STOP]
            and doc["native_frames"] == STOP - START
            and doc["terminal_horizon_by_arm"] == dict.fromkeys(ARMS, STOP)
            and doc["physical_ID_truth"] is False
            and doc["strict_support_ray_mm"] == 2.0
            and doc["strict_support_pixel_gate"] == 12.0,
            "same final comparison scope differs",
        )
    references, tracking, indices = plan["candidate_references"], [], []
    for arm, name, digest in zip(ARMS, ROOT_NAMES, TRACKING_SHAS, strict=True):
        root, ref = WORK / name, references[arm]
        edge(seal, root / "tracking.provenance.json", digest)
        producer = previous.read(root / "tracking.provenance.json", evidence, digest)
        require(
            producer["stage"] == name and producer["status"] == "ok", "successful producer required"
        )
        final = document(root, producer, "summary.json", evidence)
        require(
            all(
                final[k] == v
                for k, v in dict(
                    scientific_interval=[START, STOP],
                    retained_interval=[FIRST, STOP],
                    native_frames=STOP - START,
                    initialize_count=1,
                    restore_count=6,
                    new_begin_calls=0,
                    external_pose_resets=0,
                    finish_calls=1,
                    finish_matches_checkpoint_backtrace=True,
                ).items()
            ),
            "producer completion differs",
        )
        require(
            ref["tracking_seal_sha256"] == digest
            and ref["terminal_horizon"] == STOP
            and len(ref["arrays"]) == 21,
            "candidate final reference differs",
        )
        for role, key in (
            ("candidate.npz", "path"),
            ("milestones/through_116200/candidate.npz", "milestone_path"),
        ):
            r = producer["outputs"][role]
            require(
                r["path"] == ref[key] and r["sha256"] == ref["sha256"] == ref["milestone_sha256"],
                "final/milestone candidate edge differs",
            )
            edge(seal, Path(r["path"]), r["sha256"])
        copy = seal["outputs"][f"candidates/{arm}.npz"]
        require(
            copy["sha256"] == ref["sha256"]
            and copy["size_bytes"] == producer["outputs"]["candidate.npz"]["size_bytes"],
            "comparison copy differs",
        )
        index = document(root, producer, "input_index.json", evidence)
        edge(seal, root / "input_index.json", producer["outputs"]["input_index.json"]["sha256"])
        tracking.append(producer)
        indices.append(index)
    require(indices[0] == indices[1], "both methods must consume identical source chunks and seed")
    binding = indices[0]["original_binding"]["inputs"]
    chunks = {
        i: load_chunk(c, binding, tracking, evidence) for i, c in select_chunks(indices[0]).items()
    }
    shared = binding["seed"]["SL_and_calibration_sha256"]
    require(
        len(shared) == 7 and sum(Path(p).suffix == ".npz" for p in shared) == 3,
        "registered SL3/rig4 required",
    )
    for path, digest in shared.items():
        for producer in (*tracking, seal):
            edge(producer, Path(path), digest)
    return dict(
        seal=seal,
        references=references,
        chunks=chunks,
        binding=binding,
        shared=shared,
        source_code=source_code(evidence, plan),
    )


def candidate_rows(arrays, reference):
    require(set(arrays) == set(KEYS), "only five fields required")
    require(
        all(proof.signature(a) == reference["arrays"][k] for k, a in arrays.items()),
        "candidate field signature differs",
    )
    n = STOP - FIRST
    shapes = dict(
        frame_idx=(n,),
        pose_xyz=(n, 3, 3, 3),
        sources=(n, 3, 3, 3),
        observed=(n, 3, 3),
        identity_review=(n, 3),
    )
    dtypes = dict(
        frame_idx="<i8", pose_xyz="<f8", sources="<i4", observed="|b1", identity_review="|b1"
    )
    require(
        all(a.shape == shapes[k] and a.dtype == np.dtype(dtypes[k]) for k, a in arrays.items()),
        "candidate schema differs",
    )
    require(
        np.array_equal(arrays["frame_idx"], np.arange(FIRST, STOP)),
        "candidate native horizon differs",
    )
    rows = {k: a[np.asarray(FRAMES) - FIRST].copy() for k, a in arrays.items()}
    require(np.isfinite(rows["pose_xyz"]).all(), "selected finite estimated bodies required")
    for a in rows.values():
        a.flags.writeable = False
    return rows


def load_inputs(context, evidence):
    rows = {}
    for arm in ARMS:
        path = previous.member(COMPARISON_ROOT, context["seal"], f"candidates/{arm}.npz", evidence)
        with np.load(path, allow_pickle=False) as archive:
            require(
                len(archive.files) == len(set(archive.files)) == 21
                and set(archive.files) == set(context["references"][arm]["arrays"]),
                "candidate all21 membership differs",
            )
            rows[arm] = candidate_rows({k: archive[k] for k in KEYS}, context["references"][arm])
    shared = context["shared"]
    for path, digest in shared.items():
        evidence.add(Path(path), digest)
    # CachedNodes requires a contiguous interval. Decode each cache once, select
    # only our nine instants through its native-frame accessor, then freeze them.
    seed = context["binding"]["seed"]
    experiments = {c["manifest"]["experiment"] for c in context["chunks"].values()}
    require(len(experiments) == 1, "chunk experiments differ")
    nodes = observations.observations.CachedNodes(
        seed["input_root"], experiments.pop(), np.arange(FRAMES[0], FRAMES[-1] + 1)
    )
    require(
        {str(Path(p).resolve()) for p in nodes.paths.values()}
        == {p for p in shared if Path(p).suffix == ".npz"},
        "registered SL membership differs",
    )
    points = {f: {v: nodes.at(v, f).copy() for v in VIEWS} for f in FRAMES}
    for frame in points.values():
        for a in frame.values():
            a.flags.writeable = False
    paths = [Path(p) for p in shared if Path(p).name == "calibration_manifest.json"]
    require(len(paths) == 1, "one rig manifest required")
    config = proof.read_json(paths[0])
    models = {
        str(prior.native.checked_relative(paths[0].parent, r["path"]))
        for r in config["model_files"]
    }
    require(
        len(models) == 3
        and models | {str(paths[0])} == {p for p in shared if Path(p).suffix == ".json"},
        "registered rig model membership differs",
    )
    rig = observations.calib.load_rig(paths[0].parent)
    require(tuple(rig.views) == tuple(VIEWS) and rig.world_units == "mm", "rig views/units differ")
    context.update(rows=rows, points=points, rig=rig)
    return context


def load_chunk_masks(chunk):
    require("packed" not in chunk and "mask_records" not in chunk, "mask chunk already captured")
    previous.video_stat_guard(chunk["videos"], chunk["chunk"])
    chunk["packed"], chunk["mask_records"] = {}, {}
    for view in VIEWS:
        chunk["packed"][view], chunk["mask_records"][view] = frozen.capture_masks(
            chunk["mask_paths"][view],
            chunk["manifest"]["masks"][view]["sha256"],
            np.asarray(chunk["manifest"]["frames"], np.int64),
        )


def native_images(chunk, frame):
    iterator = prior.raw.iter_native(chunk["videos"], chunk["manifest"], frame, frame + 1)
    try:
        native, images = next(iterator)
        require(
            native == frame and next(iterator, None) is None, "one absolute native frame required"
        )
        return images
    finally:
        iterator.close()


def frame_masks(chunk, frame, view):
    pos = frame - chunk["chunk"]["start"]
    data = chunk["packed"][view]
    require(
        0 <= pos < len(data["frame_idx"]) and int(data["frame_idx"][pos]) == frame,
        "native mask offset differs",
    )
    masks = prior.prepare_sequence.decode_frame_masks(data, pos)
    digest = prior.prepare_sequence.array_sha256(masks)
    require(
        digest == bytes(data["frame_mask_sha256"][pos]).decode(), "decoded mask pixel SHA differs"
    )
    available = bool(data["view_available"][pos])
    count = int(chunk["manifest"]["masks"][view]["instances_per_frame"][pos])
    require(len(masks) == count, "manifest mask instance count differs")
    return masks if available else None, dict(
        mask_sha256=digest, mask_count=len(masks), mask_available=available
    )


def png_bytes(image):
    ok, encoded = prior.cv2.imencode(".png", image)
    require(ok, "native PNG encoding failed")
    require(
        np.array_equal(prior.cv2.imdecode(encoded, prior.cv2.IMREAD_COLOR), image),
        "native PNG pixels changed",
    )
    return encoded.tobytes()


def render(frame, images, masks, points, projections, rows, t, mode):
    fig, axes = prior.plt.subplots(3, 3, figsize=(18, 17), dpi=160)
    fig.suptitle(
        f"Native {frame} ({frame / 166:.6f} source seconds) | {mode} | same final 116200\n"
        "Rows: independent observations / Sparse z >= -6 / Depth z >= -12. "
        "Columns: three cameras.\n"
        "S0 cyan / S1 yellow / S2 magenta: anonymous RAW slots; no sex or physical-ID labels.\n"
        "M circle / C square / T triangle: filled >=2-view direct support, open latent. "
        "SL and D2 indices are frame/view-local.\n"
        "Selected static evidence; no continuous-ID certification. "
        "Same crop across arms and modes.",
        fontsize=11,
    )
    crops = {}
    try:
        for vi, view in enumerate(VIEWS):
            lo, hi = visual.display_crop(
                images[view], masks[view], points[view], [projections[a][view] for a in ARMS]
            )
            crops[view] = dict(lower_xy=lo.tolist(), upper_xy=hi.tolist())
            for ri, label in enumerate(("Independent observations", *LABELS)):
                arm, ax = (ARMS[ri - 1] if ri else None), axes[ri, vi]
                visual.draw_panel(
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
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        stream = io.BytesIO()
        fig.savefig(stream, format="png")
        return stream.getvalue(), crops
    finally:
        prior.plt.close(fig)


def signatures(context):
    return dict(
        candidates={
            a: {k: proof.signature(v) for k, v in r.items()} for a, r in context["rows"].items()
        },
        SL={
            str(f): {v: proof.signature(p) for v, p in views.items()}
            for f, views in context["points"].items()
        },
    )


def guard_chunk(chunk):
    previous.video_stat_guard(chunk["videos"], chunk["chunk"])
    for view, record in chunk["mask_records"].items():
        require(
            observations.file_stat(Path(record["path"])) == record["file_stat"],
            "captured mask file changed",
        )
        if "packed" in chunk:
            require(
                {k: proof.signature(v) for k, v in chunk["packed"][view].items()}
                == record["arrays"],
                "captured mask memory changed",
            )


def final_input_guard(context):
    for chunk in context["chunks"].values():
        guard_chunk(chunk)


def cpu_guard():
    require(
        "torch" not in sys.modules or not sys.modules["torch"].cuda.is_initialized(),
        "CPU-only renderer required",
    )


def run(output):
    require(EXECUTION_READY, "source review and ROOT execution Go required")
    output = Path(output).resolve()
    require(output.is_relative_to(WORK) and not output.exists(), "fresh workspace output required")
    cpu_guard()
    out, evidence, started = proof.Output(output), proof.Evidence(), proof.utc()
    try:
        context = load_inputs(load_headers(evidence), evidence)
        before = signatures(context)
        rows = context["rows"]
        common = dict(
            schema=STAGE,
            frames=list(FRAMES),
            arms=list(ARMS),
            terminal_horizon=STOP,
            comparison_sha256=COMPARISON_SHA,
            tracking_sha256=dict(zip(ARMS, TRACKING_SHAS, strict=True)),
            candidate_references=context["references"],
            selected_candidate_arrays=before["candidates"],
            requested_BGR_frames=27,
            target_mask_decodes=27,
            packed_mask_payload_reads=6,
            native_PNGs=27,
            contact_sheets=18,
            whole_video_rehashes=0,
            checkpoint_or_DAG_reads=0,
            GPU_used=False,
            candidate_modified=False,
            review_modified=False,
            physical_ID_truth=False,
            sex_confirmed=False,
            automatic_adoption=False,
            selected_frames_are_holdout=False,
            candidate_fields_directly_verified=list(KEYS),
            other_16_candidate_signatures_inherited=True,
            physical_ID_mapping_applied=False,
            SL_used_by_fit_is_not_truth=True,
        )
        out.write(
            "plan.json",
            dict(
                **common,
                source_code=context["source_code"],
                SL_cache_interval=[FRAMES[0], FRAMES[-1] + 1],
                selection=(
                    "female endpoint bracket, nearby motion-label boundary and "
                    "remaining direct-tail event; no human labels added"
                ),
            ),
        )
        for arm in ARMS:
            out.write(f"selected/{arm}.npz", rows[arm])
        records, active_chunk = [], None
        for t, frame in enumerate(FRAMES):
            chunk = frame_chunk(context["chunks"], frame)
            if chunk is not active_chunk:
                if active_chunk is not None:
                    guard_chunk(active_chunk)
                    del active_chunk["packed"]  # Release chunk19 before capturing chunk20.
                load_chunk_masks(chunk)
                active_chunk = chunk
            images, masks, observed_inputs = native_images(chunk, frame), {}, {}
            points = context["points"][frame]
            for view in VIEWS:
                masks[view], record = frame_masks(chunk, frame, view)
                image = images[view]
                require(
                    image.dtype == np.uint8 and image.shape == (1200, 1200, 3),
                    "native BGR schema differs",
                )
                digest = prior.prepare_sequence.array_sha256(image)
                pos = frame - chunk["chunk"]["start"]
                require(
                    digest == chunk["manifest"]["raw"][view]["frame_bgr_sha256"][pos],
                    "native BGR manifest SHA differs",
                )
                name = f"native/frame_{frame}_{view}.png"
                out.write(name, png_bytes(image))
                observed_inputs[view] = dict(
                    **record,
                    bgr_sha256=digest,
                    native_png=name,
                    canonical_video_frame=frame,
                    chunk_local_offset=pos,
                    camera_sync_offset_applied=0,
                    source_chunk_index=chunk["chunk"]["index"],
                    SL_points_signature=proof.signature(points[view]),
                    SLEAP_points=prior.finite_json(points[view]),
                )
            projections = {
                a: {v: context["rig"].camera(v).project(rows[a]["pose_xyz"][t]) for v in VIEWS}
                for a in ARMS
            }
            methods = {
                a: dict(
                    pose_xyz_mm=rows[a]["pose_xyz"][t].tolist(),
                    sources=rows[a]["sources"][t].tolist(),
                    observed=rows[a]["observed"][t].tolist(),
                    identity_review=rows[a]["identity_review"][t].tolist(),
                    projections={v: prior.finite_json(p) for v, p in projections[a].items()},
                )
                for a in ARMS
            }
            crops = None
            for mode in ("raw", "D2"):
                image, current = render(frame, images, masks, points, projections, rows, t, mode)
                require(crops is None or crops == current, "raw/D2 crops differ")
                crops = current
                out.write(f"frame_{frame}_{mode}.png", image)
            records.append(
                dict(
                    frame=frame, observations=observed_inputs, methods=methods, display_crops=crops
                )
            )
            print(f"Rendered native {frame}; requested BGR/masks {(t + 1) * 3}/27", flush=True)
        require(signatures(context) == before, "saved candidate or SL values changed")
        final_input_guard(context)
        out.write(
            "source_and_frame_evidence.json",
            dict(
                **common,
                status="ok",
                records=records,
                source_chunks={str(i): c["chunk"] for i, c in context["chunks"].items()},
                captured_mask_records={
                    str(i): c["mask_records"] for i, c in context["chunks"].items()
                },
                shared_SL_calibration_sha256=context["shared"],
                pixel_evidence=(
                    "consumed manifest BGR hashes + consumed packed-mask per-frame "
                    "hashes; no companion pixel hashes available"
                ),
            ),
        )
        evidence.final_guard()
        out.final_guard()
        cpu_guard()
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)


if __name__ == "__main__":
    main()
