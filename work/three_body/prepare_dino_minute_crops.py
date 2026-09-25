"""Prepare fixed60 native RGB/all-mask crops, without a model or candidate arrays.

New plan uses only59 adjacent pairs. Frozen six-frame helpers are not patched.
The successful same76360 comparison authenticates historical candidate geometry;
this stage binds its paths/digests but never deserializes that geometry.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
import prepare_dino_fixed_crops as prior

REPO = Path(__file__).resolve().parents[2]
WORK = REPO / "work/three_body"
sys.path[:0] = [str(REPO / "src"), str(REPO / "labs/20260909_three_body")]
import long_final_native_samples as native  # noqa: E402
from observations import CachedNodes  # noqa: E402
from source_sequence import validate_source_manifest  # noqa: E402

assets, partial = prior.assets, prior.partial
require, signature, read_json = prior.require, prior.signature, prior.read_json
FRAMES = tuple(66483 + 166 * k for k in range(60))
VIEWS, CONDITIONS = prior.VIEWS, prior.CONDITIONS
START, STOP, CHUNK = 66400, 76360, 1660
STAGE = "dino_minute_adjacent_crop_preparation_v1"
PLAN = WORK / "DINO_MINUTE_ADJACENT_PLAN.md"
PLAN_SHA = "55ad7c8ad7d719eaa9a81c4e9ce14e9471b497f34b687f8c9a62552902eb9517"
ORIGINAL = WORK / "long_remaining_supervisor_v1/phases/f066400-076360/tracking"
ORIGINAL_SHA = "95098e39bbd4d1e09d7ef98e3a7c244fe89ab2caf36a3650f360d35fc8289382"
COMPARISON = WORK / "tail_one_minute_comparison_v1"
COMPARISON_SHA = "8a3a77326f34e77801f834b529fef79626116fba63a1f99155ec33f2cea3e557"
PIXELS = WORK / "directional_minute_unused_pixel_samples_v1"
PIXELS_SHA = "a219bbad9980acfc1e4d17928829fad0a39792c97be6bd9e14abc5e8938cfa4e"
CANDIDATES = {
    "original": (
        ORIGINAL / "candidate.npz",
        "870836f9e086db6d1675dd283846e758038fb942fb6e8df6296f63aad98a5877",
    ),
    "tail": (
        WORK / "tail_one_minute_66400-76360_v1/candidate.npz",
        "4a86511a1bb01a28de7f4497e848ca0e9c71badefe61fe23722039a2e993f7f7",
    ),
}
PINS = {
    **prior.PINS,
    Path(
        prior.__file__
    ).resolve(): "eeea2cff12f8dca71b1a553960bfc0ef6380361f0c3ec090cc74757e748d36d0",
    Path(
        native.__file__
    ).resolve(): "f9c82e6fc7150edd20f424dd9fa75fda2ef321ad10f41076744ef7a7689e7791",
    PLAN: PLAN_SHA,
}


def bind(path, digest, evidence):
    """Hash each unique input once initially, and all again at final_guard."""
    path = Path(path).resolve(strict=True)
    if str(path) in evidence:
        require(evidence[str(path)] == digest, "conflicting input SHA")
        return path
    return prior.bind(path, digest, evidence)


def header(root, filename, expected, stage, evidence):
    path = bind(Path(root) / filename, expected, evidence)
    value = read_json(path)
    require(
        value.get("stage") == stage and value.get("status") == "ok" and value.get("finished_at"),
        "successful pinned input seal required",
    )
    return value


def member(record, group, path, evidence, *, hash_file=True):
    """Resolve real paths; older seals have both named and absolute output roles."""
    path = Path(path).resolve(strict=True)
    rows = [r for r in record[group].values() if Path(r["path"]).resolve() == path]
    require(
        rows and len({r["sha256"] for r in rows}) == 1, "input/output membership absent/conflicting"
    )
    require(
        all(r.get("size", r.get("size_bytes")) == path.stat().st_size for r in rows),
        "bound file size changed",
    )
    digest = rows[0]["sha256"]
    if hash_file:
        bind(path, digest, evidence)
    return digest


def selected_records(records):
    require(
        isinstance(records, list) and len(records) == 16, "original76360 consumed prefix required"
    )
    for i, row in enumerate(records):
        require(
            type(row.get("index")) is int
            and row["index"] == i
            and type(row.get("start")) is int
            and row["start"] == 49800 + i * CHUNK
            and type(row.get("stop")) is int
            and row["stop"] == 49800 + (i + 1) * CHUNK,
            "consumed native prefix order changed",
        )
    return records[10:16]


def validate_minute(summary, pixel_summary):
    require(
        summary.get("schema") == "tail_one_minute_comparison_v1"
        and summary.get("status") == "ok"
        and summary.get("native_interval") == [START, STOP]
        and summary.get("horizon_stop") == STOP
        and summary.get("native_frames") == 9960
        and all(
            summary.get(k) is True
            for k in (
                "same_parent_and_horizon",
                "one_minute_completed",
                "all39_arrays_semantically_decoded",
                "all21_arrays_exact_backtrace",
            )
        ),
        "completed same76360 comparison required",
    )
    require(
        pixel_summary.get("status") == "ok"
        and pixel_summary.get("native_interval") == [START, STOP]
        and pixel_summary.get("scored_frames") == list(FRAMES),
        "fixed60 prior frames changed",
    )
    for method, (path, digest) in CANDIDATES.items():
        require(
            Path(summary["candidate_paths"][method]).resolve() == path.resolve()
            and summary["candidate_sha256"][method] == digest,
            "fixed candidate origin changed",
        )


def validate_chunk(row, root, manifest, seal, fixed, evidence, videos):
    require(
        manifest.get("experiment") == native.EXPERIMENT
        and manifest.get("frames") == list(range(row["start"], row["stop"]))
        and manifest.get("fps") == 166
        and manifest.get("image_shape_hw") == [1200, 1200]
        and manifest.get("binding") == fixed["detector_binding"] == seal["params"]["binding"]
        and manifest.get("tracking_performed") is False
        and manifest.get("human_truth_created") is False,
        "chunk/source independent binding changed",
    )
    validate_source_manifest(root, manifest)
    for view in VIEWS:
        info = manifest["raw"][view]
        raw = row["raw_hardlinks"][view]
        canonical = Path(raw["source_path"]).resolve(strict=True)
        linked = native.checked_relative(root, info["path"])
        require(
            linked == Path(raw["path"]).resolve() and os.path.samefile(linked, canonical),
            "canonical raw hardlink differs",
        )
        stat = canonical.stat()
        require(
            [stat.st_size, stat.st_dev, stat.st_ino, stat.st_mtime_ns]
            == [raw["bytes"], raw["device"], raw["inode"], raw["mtime_ns"]]
            and info["sha256"] == raw["sha256"],
            "canonical raw identity differs",
        )
        require(len(info["frame_bgr_sha256"]) == CHUNK, "full chunk BGR signatures required")
        require(view not in videos or videos[view] == canonical, "view canonical video changed")
        videos[view] = canonical
        bind(canonical, raw["sha256"], evidence)
        require(
            member(seal, "outputs", linked, evidence, hash_file=False) == raw["sha256"],
            "raw output role differs",
        )
        for category, role in (("raw", view + "_raw"), ("masks", view + "_masks")):
            path = native.checked_relative(root, manifest[category][view]["path"])
            prior_row = row["output_hashes"][role]
            require(
                Path(prior_row["path"]).resolve() == path
                and prior_row["sha256"]
                == manifest[category][view]["sha256"]
                == member(seal, "outputs", path, evidence, hash_file=category != "raw"),
                "consumed manifest/payload SHA differs",
            )


def load_inputs(evidence):
    comparison = header(
        COMPARISON,
        "comparison.provenance.json",
        COMPARISON_SHA,
        "tail_one_minute_comparison_v1",
        evidence,
    )
    for item in comparison["outputs"].values():
        path = Path(item["path"]).resolve(strict=True)
        require(path.is_relative_to(COMPARISON.resolve()), "comparison output escaped root")
        member(comparison, "outputs", path, evidence)
    summary_path = COMPARISON / "summary.json"
    member(comparison, "outputs", summary_path, evidence)
    summary = read_json(summary_path)
    pixels = header(
        PIXELS,
        "preparation.provenance.json",
        PIXELS_SHA,
        "directional_minute_unused_pixel_preparation",
        evidence,
    )
    member(pixels, "outputs", PIXELS / "summary.json", evidence)
    pixel_summary = read_json(PIXELS / "summary.json")
    validate_minute(summary, pixel_summary)
    original = header(
        ORIGINAL,
        "continuation.provenance.json",
        ORIGINAL_SHA,
        "three_body_long_continuation",
        evidence,
    )
    member(comparison, "inputs", ORIGINAL / "continuation.provenance.json", evidence)
    for _, (path, digest) in CANDIDATES.items():
        require(member(comparison, "inputs", path, evidence) == digest, "candidate SHA differs")
    require(original["params"]["through_frame"] == STOP, "wrong original horizon")
    fixed = original["params"]["bindings"]["inputs"]
    shared = fixed["seed"]["SL_and_calibration_sha256"]
    require(
        shared == summary["shared_SL_calibration_sha256"]
        and len(shared) == 7
        and sum(Path(p).suffix == ".npz" for p in shared) == 3
        and sum(Path(p).suffix == ".json" for p in shared) == 4,
        "sharedSL3/calib4 differs",
    )
    for path, digest in shared.items():
        bind(path, digest, evidence)
        require(
            member(comparison, "inputs", path, evidence) == digest
            and member(original, "inputs", path, evidence) == digest,
            "shared inputs differ",
        )
    for path, digest in original["params"]["bindings"]["code_sha256"].items():
        bind(path, digest, evidence)
    consumed_path = ORIGINAL / "consumed_chunks.json"
    member(original, "outputs", consumed_path, evidence)
    records = selected_records(read_json(consumed_path))
    stage = Path(fixed["input_stage_root"]).resolve(strict=True)
    chunks, videos = [], {}
    for row in records:
        root = native.checked_relative(stage, row["attempt_path"])
        seal = header(
            root,
            "staging.provenance.json",
            row["provenance_sha256"],
            "prepare_independent_mask_chunk",
            evidence,
        )
        manifest_path = bind(root / "manifest.json", row["manifest_sha256"], evidence)
        require(
            member(seal, "outputs", manifest_path, evidence) == row["manifest_sha256"],
            "manifest output binding differs",
        )
        manifest = read_json(manifest_path)
        validate_chunk(row, root, manifest, seal, fixed, evidence, videos)
        for view in VIEWS:
            path = native.checked_relative(root, manifest["masks"][view]["path"])
            require(
                pixel_summary["source_mask_sha256"][str(path)] == evidence[str(path)]
                and member(original, "inputs", path, evidence) == evidence[str(path)],
                "original fixed mask source differs",
            )
        chunks.append(dict(record=row, root=root, manifest=manifest))
    require(len(pixel_summary["source_mask_sha256"]) == 18, "all6x3 mask bindings required")
    return dict(
        chunks=chunks,
        videos=videos,
        shared=shared,
        fixed=fixed,
        candidate_references={k: dict(path=str(p), sha256=d) for k, (p, d) in CANDIDATES.items()},
    )


def enumerate_plan(counts):
    require(
        set(counts) == {(n, v) for n in FRAMES for v in VIEWS}, "all180 observation counts required"
    )
    require(
        all(n is None or (type(n) is int and n >= 0) for n in counts.values()),
        "mask count must be nonnegative Pythonint or None",
    )
    crops = [
        dict(frame_idx=n, view=v, mask_row=i, crop_id=f"f{n}_{v}_m{i:03d}")
        for n in FRAMES
        for v in VIEWS
        for i in range(counts[n, v] or 0)
    ]
    requests, pairs = [], []
    for early, late in pairwise(FRAMES):
        for view in VIEWS:
            left, right = counts[early, view], counts[late, view]
            request = dict(
                request_index=len(requests),
                earlier_frame=early,
                later_frame=late,
                view=view,
                earlier_mask_count=left,
                later_mask_count=right,
                status="view_unavailable"
                if left is None or right is None
                else "no_mask_pairs"
                if not left or not right
                else "available",
                mask_pair_count=(left or 0) * (right or 0),
            )
            requests.append(request)
            for i in range(left or 0):
                for j in range(right or 0):
                    pairs.append(
                        dict(
                            pair_index=len(pairs),
                            request_index=request["request_index"],
                            earlier_frame=early,
                            later_frame=late,
                            view=view,
                            earlier_mask_row=i,
                            later_mask_row=j,
                            earlier_crop_id=f"f{early}_{view}_m{i:03d}",
                            later_crop_id=f"f{late}_{view}_m{j:03d}",
                        )
                    )
    return dict(
        schema=STAGE,
        frames=list(FRAMES),
        views=list(VIEWS),
        conditions=list(CONDITIONS),
        crops=crops,
        requests=requests,
        pairs=pairs,
        maximum_forward_calls=2 * len(crops),
        adjacent_time_pairs=59,
        requested_pair_views=177,
        model_inference_in_this_stage=False,
        posthoc_after_fixed6_results=True,
        temporal_holdout=False,
        biological_ID_truth=False,
    )


def mask_header(data, chunk):
    require(set(data) == set(native.SCHEMA_DTYPES), "all13 packed fields required")
    for key, dtype in native.SCHEMA_DTYPES.items():
        require(data[key].dtype == dtype, "packed dtype changed: " + key)
    first, stop = chunk["record"]["start"], chunk["record"]["stop"]
    require(
        np.array_equal(data["frame_idx"], np.arange(first, stop, dtype=np.int64))
        and np.array_equal(data["image_shape_hw"], [1200, 1200]),
        "packed native/shape differs",
    )


def prepare_crop(image, mask):
    core = partial.eroded_core(mask)
    return prior.prepared_crop(
        image, mask, dict(core_sha256=signature(core)["sha256"], core_pixels=int(core.sum()))
    )


def source_code():
    code = {Path(__file__).resolve(), PLAN, *PINS}
    for module in list(sys.modules.values()):
        value = getattr(module, "__file__", None)
        if value:
            path = Path(value).resolve()
            if path.is_relative_to(REPO) and path.suffix == ".py" and ".venv" not in path.parts:
                code.add(path)
    return {str(p): PINS.get(p, assets.sha(p)) for p in sorted(code)}


def run(output):
    output = Path(output).resolve()
    require(output.is_relative_to(WORK) and not output.exists(), "new workspace output required")
    prior.no_model_imports()
    started, wall = assets.now(), time.perf_counter()
    code = source_code()
    evidence = {}
    for path, digest in code.items():
        bind(path, digest, evidence)
    output.mkdir(parents=True)
    outputs = {}
    timings = {}
    captures = {}

    def record(path):
        outputs[path.relative_to(output).as_posix()] = dict(
            path=str(path), sha256=assets.sha(path), size_bytes=path.stat().st_size
        )

    def write(name, value):
        path = output / name
        assets.write_json(path, value)
        record(path)

    def dump(name, arrays):
        path = output / name
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        record(path)

    try:
        t = time.perf_counter()
        inputs = load_inputs(evidence)
        asset_manifest, asset_evidence = prior.smoke.verify_assets(
            prior.ASSET_ROOT, prior.ASSET_SHA
        )
        for path, digest in asset_evidence.items():
            bind(path, digest, evidence)
        timings["input_verification_seconds"] = time.perf_counter() - t
        for path, digest in code.items():
            dest = output / "code_snapshot" / Path(path).relative_to(REPO)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
            require(assets.sha(dest) == digest, "source snapshot differs")
            record(dest)
        counts = {}
        t = time.perf_counter()
        for chunk in inputs["chunks"]:
            for view in VIEWS:
                path = native.checked_relative(
                    chunk["root"], chunk["manifest"]["masks"][view]["path"]
                )
                with np.load(path, allow_pickle=False) as archive:
                    fields = {
                        k: archive[k].copy()
                        for k in ("frame_idx", "frame_offsets", "view_available")
                    }
                require(
                    np.array_equal(
                        fields["frame_idx"],
                        np.arange(chunk["record"]["start"], chunk["record"]["stop"]),
                    )
                    and fields["frame_idx"].dtype == np.int64
                    and fields["frame_offsets"].dtype == np.int64
                    and fields["frame_offsets"].shape == (CHUNK + 1,)
                    and fields["frame_offsets"][0] == 0
                    and np.all(np.diff(fields["frame_offsets"]) >= 0)
                    and fields["view_available"].dtype == bool
                    and fields["view_available"].shape == (CHUNK,),
                    "mask header metadata differs",
                )
                for n in FRAMES:
                    if chunk["record"]["start"] <= n < chunk["record"]["stop"]:
                        i = n - chunk["record"]["start"]
                        count = chunk["manifest"]["masks"][view]["instances_per_frame"][i]
                        require(
                            type(count) is int
                            and count >= 0
                            and fields["frame_offsets"][i + 1] - fields["frame_offsets"][i]
                            == count,
                            "manifest native count differs",
                        )
                        available = bool(fields["view_available"][i])
                        require(available or count == 0, "unavailable has instances")
                        counts[n, view] = count if available else None
        plan = enumerate_plan(counts)
        plan.update(
            source_hashes=code,
            shared_SL_calibration_sha256=inputs["shared"],
            candidate_references=inputs["candidate_references"],
            raw_delivered_images=180,
            source_plan_sha256=PLAN_SHA,
            asset_manifest_sha256=assets.sha(prior.ASSET_ROOT / "assets.manifest.json"),
            official_model=asset_manifest["model"],
            official_commit=asset_manifest["commit"],
        )
        write("plan.json", plan)
        timings["mask_count_registration_seconds"] = time.perf_counter() - t
        t = time.perf_counter()
        nodes = CachedNodes(
            inputs["fixed"]["seed"]["input_root"], native.EXPERIMENT, np.arange(START, STOP)
        )
        require(
            {str(p.resolve()) for p in nodes.paths.values()}
            == {p for p in inputs["shared"] if Path(p).suffix == ".npz"},
            "SL loader source paths differ",
        )
        timings["SL_archive_load_seconds"] = time.perf_counter() - t
        for directory in ("raw", "observations", "crops"):
            (output / directory).mkdir()
        crop_rows = []
        native_rows = []
        for view in VIEWS:
            captures[view] = cv2.VideoCapture(str(inputs["videos"][view]))
            native.validate_capture(captures[view])
        for key in (
            "raw_decode_seconds",
            "mask_decode_seconds",
            "mask_archive_load_seconds",
            "crop_seconds",
            "payload_save_seconds",
        ):
            timings[key] = 0.0
        for chunk in inputs["chunks"]:
            selected = [
                n for n in FRAMES if chunk["record"]["start"] <= n < chunk["record"]["stop"]
            ]
            masks = {}
            for view in VIEWS:
                t = time.perf_counter()
                path = native.checked_relative(
                    chunk["root"], chunk["manifest"]["masks"][view]["path"]
                )
                with np.load(path, allow_pickle=False) as z:
                    require(len(z.files) == len(set(z.files)), "duplicate NPZ keys")
                    masks[view] = {k: z[k].copy() for k in z.files}
                mask_header(masks[view], chunk)
                timings["mask_archive_load_seconds"] += time.perf_counter() - t
            for n in selected:
                offset = n - chunk["record"]["start"]
                for view in VIEWS:
                    t = time.perf_counter()
                    image = native.read_native_frame(
                        captures[view],
                        n,
                        chunk["manifest"]["raw"][view]["frame_bgr_sha256"][offset],
                    )
                    timings["raw_decode_seconds"] += time.perf_counter() - t
                    t = time.perf_counter()
                    packed, decoded = native.select_packed_frame(
                        masks[view], offset, n, counts[n, view] or 0
                    )
                    timings["mask_decode_seconds"] += time.perf_counter() - t
                    available = decoded is not None
                    all_masks = decoded if available else np.empty((0, 1200, 1200), bool)
                    require(
                        (len(all_masks) if available else None) == counts[n, view],
                        "registered mask availability changed",
                    )
                    detections = nodes.at(view, n).copy()
                    t = time.perf_counter()
                    png = output / "raw" / f"raw_{n}_{view}.png"
                    require(cv2.imwrite(str(png), image), "lossless PNG write failed")
                    record(png)
                    restored = cv2.imread(str(png), cv2.IMREAD_COLOR)
                    require(
                        restored is not None and signature(restored) == signature(image),
                        "lossless PNG roundtrip differs",
                    )
                    obsname = f"observations/observation_{n}_{view}.npz"
                    obs = dict(
                        masks=all_masks,
                        mask_available=np.asarray(available),
                        detections=detections,
                        **{"packed_" + k: v for k, v in packed.items()},
                    )
                    dump(obsname, obs)
                    timings["payload_save_seconds"] += time.perf_counter() - t
                    native_rows.append(
                        dict(
                            frame_idx=n,
                            view=view,
                            mask_count=counts[n, view],
                            bgr_sha256=signature(image)["sha256"],
                            mask_sha256=signature(all_masks)["sha256"],
                            detection_signature=signature(detections),
                            chunk_index=chunk["record"]["index"],
                            native_offset=offset,
                            manifest_sha256=chunk["record"]["manifest_sha256"],
                            png_path=png.relative_to(output).as_posix(),
                            png_sha256=assets.sha(png),
                            observation_path=obsname,
                            observation_sha256=outputs[obsname]["sha256"],
                            observation_array_signatures={k: signature(v) for k, v in obs.items()},
                        )
                    )
                    for index, mask in enumerate(all_masks):
                        t = time.perf_counter()
                        arrays, metadata = prepare_crop(image, mask)
                        timings["crop_seconds"] += time.perf_counter() - t
                        crop_id = f"f{n}_{view}_m{index:03d}"
                        name = f"crops/{crop_id}.npz"
                        t = time.perf_counter()
                        dump(name, arrays)
                        timings["payload_save_seconds"] += time.perf_counter() - t
                        metadata.update(
                            frame_idx=n,
                            view=view,
                            mask_row=index,
                            crop_id=crop_id,
                            path=name,
                            sha256=outputs[name]["sha256"],
                            native_png_sha256=assets.sha(png),
                            native_observation_npz_sha256=outputs[obsname]["sha256"],
                        )
                        crop_rows.append(metadata)
            print(
                json.dumps(
                    dict(
                        event="native_crops_prepared",
                        through=chunk["record"]["stop"],
                        crops=len(crop_rows),
                    )
                ),
                flush=True,
            )
        for capture in captures.values():
            capture.release()
        captures = {}
        require(
            [r["crop_id"] for r in crop_rows] == [r["crop_id"] for r in plan["crops"]]
            and [(r["frame_idx"], r["view"]) for r in native_rows]
            == [(n, v) for n in FRAMES for v in VIEWS],
            "full fixed observation/crop order differs",
        )
        write("crops.json", crop_rows)
        write("native_evidence.json", native_rows)
        prior.no_model_imports()
        require(code == source_code(), "loaded source set changed")
        t = time.perf_counter()
        prior.final_guard(evidence)
        for chunk in inputs["chunks"]:
            for view in VIEWS:
                info = chunk["record"]["raw_hardlinks"][view]
                require(
                    os.path.samefile(info["path"], inputs["videos"][view]),
                    "raw hardlink changed at end",
                )
        for row in outputs.values():
            require(assets.sha(row["path"]) == row["sha256"], "output changed before seal")
        timings["final_guard_seconds"] = time.perf_counter() - t
        timings["total_through_guards_seconds"] = time.perf_counter() - wall
        result = dict(
            schema=STAGE,
            status="ok",
            frames=list(FRAMES),
            views=list(VIEWS),
            native_frames=60,
            raw_video_delivered_images=180,
            raw_png_roundtrip_decodes=180,
            decoded_mask_stacks=180,
            decoder_internal_keyframe_work_not_counted=True,
            mask_crops=len(crop_rows),
            adjacent_time_pairs=59,
            requests=177,
            mask_pairs=len(plan["pairs"]),
            maximum_forward_calls=2 * len(crop_rows),
            ready_crops=sum(r["crop_status"] == "ready" for r in crop_rows),
            empty_crops=sum(r["crop_status"] == "empty_mask" for r in crop_rows),
            zero_eligible_token_crops=sum(r["eligible_token_count"] == 0 for r in crop_rows),
            eligible_tokens=sum(r["eligible_token_count"] for r in crop_rows),
            all_token_rows=256 * len(crop_rows),
            model_forward_calls=0,
            model_loaded=False,
            GPU=False,
            candidate_arrays_read=False,
            geometry_join_performed=False,
            ID_or_review_changed=False,
            temporal_holdout=False,
            payload_bytes_before_summary=sum(r["size_bytes"] for r in outputs.values()),
            timings=timings,
            timing_scopes={
                "mask_archive_load_seconds": "18 NPZ full13 decompression/copy and schema checks",
                "mask_decode_seconds": "selected180 slice/validation/decode; excludes archive load",
                "SL_archive_load_seconds": "CachedNodes three archives and path checks",
                "payload_save_seconds": "PNG/NPZ save, output SHA, PNG roundtrip decode/SHA",
                "SHA_time": "included in enclosing timings; not separately measured for every SHA",
            },
        )
        write("summary.json", result)
        assets.write_json(
            output / "preparation.provenance.json",
            dict(
                stage=STAGE,
                status="ok",
                started_at=started,
                finished_at=assets.now(),
                inputs=evidence,
                outputs=outputs,
                environment=prior.runtime(),
            ),
        )
        return result
    except BaseException as exc:
        assets.write_json(
            output / "failure.json",
            dict(
                stage=STAGE,
                status="failed",
                error_type=type(exc).__name__,
                error=str(exc),
                inputs=evidence,
                completed_outputs=outputs,
                timings=timings,
            ),
        )
        raise
    finally:
        for capture in captures.values():
            capture.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    result = run(parser.parse_args().output)
    print(
        json.dumps(
            {k: result[k] for k in ("status", "mask_crops", "mask_pairs", "model_forward_calls")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
