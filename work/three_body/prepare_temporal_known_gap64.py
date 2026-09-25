"""Prepare the separately planned known-gap64 using legacy independent masks.

No legacy AVI read/hash, SL deserialization, model import, flow or GPU. Reuses
frozen stateless evidence/packed-mask helpers; never changes their globals.
Legacy generator runtime facts and unrecorded flags remain distinguishable.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import prepare_temporal_point64 as prior
import temporal_point_core as core
from source_sequence import validate_source_manifest

bind, header, member, members = prior.bind, prior.header, prior.member, prior.members
read_json, write_json, require = prior.read_json, prior.write_json, prior.require
sha, signature, native, now = prior.sha, prior.signature, prior.native, prior.now
REPO, WORK, VIEWS = prior.REPO, prior.WORK, prior.VIEWS
START, STOP, SOURCE_STOP = 333519, 333583, 333719
FRAMES = tuple(range(START, STOP))
STAGE = "temporal_known_gap64_preparation_v1"
PLAN = WORK / "TEMPORAL_KNOWN_GAP_64_PLAN.md"
PLAN_SHA = "3cd3c7ee8bd262daca3366a9e5d325bcb8bbec11f191e159f1557acbc726f3b2"
PRIOR_SHA = "42a5d6ca4e877eff0c61b7de7788136dbcba7008e23f13901db0fbb8eb8a27ba"
CORE_SHA = "3ec308ca7877338f8677c0703d6f180a9d5658c4858b9868451e1152b7d4e25b"
OLD = WORK / "inputs/transient_333519-333719"
OLD_SHA = "9ecee1b02e5e21e40cc2d70122ab2a18de69ec1372a283fc13e55b55f56d6bd1"
OLD_MANIFEST_SHA = "3912d249571326cf6b1902e3c0263cb1fbb78e4bd956378fd27ad25cd86dd6f5"
COMPACT = WORK / "inputs/transient_source_333519-333719"
COMPACT_SHA = "4353672bf891108e6477f49c0b2bcaa46f1cb8d676c934000ce41e62a40468e8"
MANIFEST_SHA = "7748cef2d9f8c5cb872ea8d356cefc0d1e06aed6b8286958cf933eb741953374"
AUDIT_SHA = "613f3e714cdafc4baf0afa72c45949b6261c86a42d460967970a31615abeee6b"
CURRENT = WORK / "temporal_point64_preparation_v1"
CURRENT_SHA = "9771ee33cf36ba320cb2cf5b055c3c44b59f906d835ffac346fccb65241769f3"
INITIAL_MASK_COUNTS, WINDOW_MASK_COUNTS = (3, 4, 3), (198, 221, 189)
UNRECORDED_FLAGS = (
    "cuda_matmul_allow_tf32",
    "cudnn_allow_tf32",
    "cudnn_benchmark",
    "autocast",
)


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")


def legacy_asset_roles(seal, manifest):
    """Old video roles have no SHA; neither they nor UNC model paths are opened."""
    for kind in ("top", "side"):
        for role in ("weights", "config"):
            item = seal["inputs"][kind + "_" + role]
            expected = manifest["models"][kind][role]
            require(
                Path(item["path"]) == Path(expected["path"])
                and item["sha256"] == expected["sha256"]
                and item["size"] == expected["bytes"],
                "legacy model role differs",
            )


def validate_metadata(old, compact, audit, old_seal):
    """Finite legacy edge; no claim that old generator declared current flags."""
    for value in (old, compact):
        require(
            value["experiment"] == native.EXPERIMENT
            and value["frames"] == list(range(START, SOURCE_STOP))
            and all(type(n) is int for n in value["frames"])
            and value["source_interval_half_open"] == [START, SOURCE_STOP]
            and value["fps"] == 166
            and value["image_shape_hw"] == [1200, 1200]
            and set(value["raw"]) == set(value["masks"]) == set(VIEWS)
            and value["mask_schema"] == "bbox_packbits_instances_v1",
            "legacy native schema differs",
        )
    require(
        old.get("raw_storage") is None
        and "FFV1" in old["raw_codec"]
        and compact.get("raw_storage") == "original_source_video"
        and compact["storage_qa_status"] == "verified",
        "storage adapter type differs",
    )
    require(
        old["models"] == compact["models"]
        and old["runtime"] == compact["runtime"]
        and old["code_sha256"] == compact["code_sha256"]
        and old_seal["params"]["source_code_sha256"] == old["code_sha256"]
        and old_seal["params"]["source_interval_half_open"] == [START, SOURCE_STOP]
        and old_seal["params"]["threshold"] == 0.5,
        "legacy detector condition differs",
    )
    require(
        audit["status"] == "verified"
        and audit["frame_count"] == 200
        and audit["images"] == 600
        and audit["code_and_inputs_unchanged"] is True
        and audit["source_video_full_sha256_before_after_verified"] is True
        and audit["compact_manifest_sha256"] == MANIFEST_SHA,
        "storage full200 audit differs",
    )
    for i, view in enumerate(VIEWS):
        hashes = old["raw"][view]["frame_bgr_sha256"]
        require(
            len(hashes) == 200
            and all(valid_sha(s) for s in hashes)
            and compact["raw"][view]["frame_bgr_sha256"] == hashes
            and compact["raw"][view]["source_offset_frames"] == START
            and compact["raw"][view]["source_total_frames"] == 348600,
            "relative0 versus native333519 mapping differs",
        )
        require(old["masks"][view] == compact["masks"][view], "copied mask descriptor differs")
        counts = old["masks"][view]["instances_per_frame"]
        require(
            len(counts) == 200
            and all(type(n) is int and n >= 0 for n in counts)
            and counts[0] == INITIAL_MASK_COUNTS[i]
            and sum(counts[:64]) == WINDOW_MASK_COUNTS[i],
            "fixed10/608 mask accounting differs",
        )
        require(
            all(audit["per_view"][view][k] == 200 for k in ("images", "masks", "detections")),
            "missing per-view full200 audit",
        )
    return dict(
        runtime_recorded=old["runtime"],
        threshold=0.5,
        source_code_sha256=old["code_sha256"],
        models=old["models"],
        generator="legacy prepare_three_body_sequence; no new D2 inference",
        unrecorded_runtime_fields=[k for k in UNRECORDED_FLAGS if k not in old["runtime"]],
        current_runtime_contract_imputed=False,
        biological_ID_truth=False,
    )


def source_identity(manifest, *, baseline=None):
    storage = manifest["storage_provenance"]
    cache = Path(storage["source_cache_path"]).resolve(strict=True)
    require(cache.is_relative_to(WORK), "source cache must be local workspace")
    videos, identities = {}, {}
    for view in VIEWS:
        info = manifest["raw"][view]
        source = native.checked_relative(cache, info["source_cache_relative_path"])
        linked = native.checked_relative(COMPACT, info["path"])
        record = storage["hardlinks"][view]
        stat = source.stat()
        require(
            os.path.samefile(source, linked)
            and info["storage_link"] == "hardlink"
            and record["same_file"] is True
            and record["inode"] == stat.st_ino
            and record["bytes"] == info["bytes"] == stat.st_size
            and record["sha256"] == info["sha256"],
            "canonical hardlink differs",
        )
        identities[view] = dict(
            source=str(source),
            linked=str(linked),
            size=stat.st_size,
            device=stat.st_dev,
            inode=stat.st_ino,
            mtime_ns=stat.st_mtime_ns,
        )
        videos[view] = source
    require(
        baseline is None or identities == baseline, "source identity changed during preparation"
    )
    return videos, identities


def metadata_inputs(evidence):
    old_seal = header(
        OLD / "staging.provenance.json", OLD_SHA, "prepare_three_body_sequence", evidence
    )
    compact_seal = header(
        COMPACT / "compact_sequence.provenance.json",
        COMPACT_SHA,
        "three_body_compact_source_sequence",
        evidence,
    )
    current_seal = header(
        CURRENT / "preparation.provenance.json",
        CURRENT_SHA,
        "temporal_point64_preparation_v1",
        evidence,
    )
    oo = members(old_seal, "outputs")
    ci, co = members(compact_seal, "inputs"), members(compact_seal, "outputs")
    ni, no = members(current_seal, "inputs"), members(current_seal, "outputs")
    for table, path, digest in (
        (oo, OLD / "manifest.json", OLD_MANIFEST_SHA),
        (ci, OLD / "manifest.json", OLD_MANIFEST_SHA),
        (co, COMPACT / "manifest.json", MANIFEST_SHA),
        (co, COMPACT / "decode_audit.json", AUDIT_SHA),
    ):
        member(table, path, digest)
        bind(path, digest, evidence)
    old, compact, audit = (
        read_json(p)
        for p in (OLD / "manifest.json", COMPACT / "manifest.json", COMPACT / "decode_audit.json")
    )
    detector = validate_metadata(old, compact, audit, old_seal)
    legacy_asset_roles(old_seal, old)
    storage = compact["storage_provenance"]
    require(
        storage["parent_manifest_sha256"] == OLD_MANIFEST_SHA
        and Path(storage["parent_sequence_path"]).resolve() == OLD.resolve(),
        "storage parent edge differs",
    )
    validate_source_manifest(COMPACT, compact)
    plan = CURRENT / "plan.json"
    current_plan = read_json(bind(plan, member(no, plan)["sha256"], evidence))
    shared = current_plan["shared_SL_calibration_sha256"]
    require(
        len(shared) == 7
        and sum(Path(p).suffix == ".npz" for p in shared) == 3
        and sum(Path(p).suffix == ".json" for p in shared) == 4,
        "current SL3/calibration4 required",
    )
    refs = []
    for path, digest in shared.items():
        refs.append(member(ni, path, digest))
        if Path(path).suffix == ".npz":
            member(ci, path, digest)
    for rel, digest in old["code_sha256"].items():
        bind(OLD / "code_snapshot" / Path(rel).name, digest, evidence)
    for path, digest in storage["code_sha256"].items():
        member(ci, path, digest)
        bind(path, digest, evidence)
    for kind in ("top", "side"):
        bind(
            OLD / old["models"][kind]["effective_config"],
            old["models"][kind]["effective_config_sha256"],
            evidence,
        )
    cache = Path(storage["source_cache_path"])
    for filename, digest in (
        ("source_cache_manifest.json", storage["source_cache_manifest_sha256"]),
        ("source_cache.provenance.json", None),
    ):
        path = cache / filename
        row = member(ci, path, digest)
        bind(path, row["sha256"], evidence)
    cache_manifest, cache_seal = (
        read_json(cache / p) for p in ("source_cache_manifest.json", "source_cache.provenance.json")
    )
    require(
        cache_manifest["status"] == "verified" and cache_seal["status"] == "ok",
        "successful canonical source cache required",
    )
    member(
        members(cache_seal, "outputs"),
        cache / "source_cache_manifest.json",
        storage["source_cache_manifest_sha256"],
    )
    videos, identities = source_identity(compact)
    for view in VIEWS:
        raw, masks = compact["raw"][view], compact["masks"][view]
        item = member(ci, videos[view], raw["sha256"])
        member(ni, videos[view], raw["sha256"])
        cached = cache_manifest["files"][raw["source_cache_relative_path"]]
        require(
            cached["local_full_sha256"] == cached["source_stream_sha256"] == raw["sha256"],
            "canonical cache SHA differs",
        )
        refs.append(item)
        path = native.checked_relative(COMPACT, masks["path"])
        old_path = native.checked_relative(OLD, old["masks"][view]["path"])
        member(oo, old_path, masks["sha256"])
        member(ci, old_path, masks["sha256"])
        require(path.stat().st_size == masks["bytes"], "copied packed size differs")
        refs.append(dict(path=str(path), sha256=masks["sha256"], size=masks["bytes"]))
    # Pin the reused decoder helper through the already completed current stage.
    for path, digest in ((Path(prior.__file__), PRIOR_SHA), (Path(core.__file__), CORE_SHA)):
        member(ni, path, digest)
        bind(path, digest, evidence)
    for module in (native, sys.modules["chunk_masks"]):
        path = Path(module.__file__)
        bind(path, member(ni, path)["sha256"], evidence)
    return dict(
        manifest=compact,
        old_manifest=old,
        references=refs,
        videos=videos,
        identities=identities,
        shared=shared,
        detector=detector,
    )


def read64(capture, hashes, *, shape=(1200, 1200)):
    """Absolute source native START; packed/legacy-clip relative index starts0."""
    require(len(hashes) == 64, "exact64 hashes required")
    native.validate_capture(capture, shape=shape)
    require(capture.set(cv2.CAP_PROP_POS_FRAMES, START), "absolute native seek failed")
    require(abs(capture.get(cv2.CAP_PROP_POS_FRAMES) - START) <= 1e-6, "native seek differs")
    for n, expected in zip(FRAMES, hashes, strict=True):
        ok, image = capture.read()
        require(
            ok
            and isinstance(image, np.ndarray)
            and image.dtype == np.uint8
            and image.shape == (*shape, 3),
            "native EOF/shape/dtype differs",
        )
        require(
            abs(capture.get(cv2.CAP_PROP_POS_FRAMES) - n - 1) <= 1e-6,
            "native decode position differs",
        )
        require(signature(image)["sha256"] == expected, "native BGR SHA differs")
        yield n, image


def observation(data, n, count):
    require(type(n) is int and START <= n < STOP, "native outside fixed64")
    packed, decoded = native.select_packed_frame(data, n - START, n, count)
    available = decoded is not None
    masks = decoded if available else np.empty((0, 1200, 1200), bool)
    return dict(
        frame_idx=np.asarray(n, np.int64),
        masks=masks,
        mask_available=np.asarray(available),
        **{"packed_" + k: v for k, v in packed.items()},
    )


def queries(image, obs):
    require(
        obs["frame_idx"].dtype == np.int64
        and obs["frame_idx"].shape == ()
        and int(obs["frame_idx"]) == START,
        "query native333519 scalar required",
    )
    original = {k: signature(v) for k, v in obs.items()}
    image_hash = signature(image)
    q = core.select_queries(image, obs["masks"] if bool(obs["mask_available"]) else None)
    require(
        original == {k: signature(v) for k, v in obs.items()} and image_hash == signature(image),
        "initial query selection changed inputs",
    )
    xy, origins, rows = q["xy"], q["mask_index"], q["rows"]
    require(
        q["available"] == bool(obs["mask_available"])
        and q["mask_count"] == (len(obs["masks"]) if q["available"] else None)
        and xy.dtype == np.float32
        and xy.shape == (len(origins), 2)
        and origins.dtype == np.int64
        and origins.ndim == 1
        and np.isfinite(xy).all(),
        "query schema differs",
    )
    indices, cursor = [], 0
    require(len(rows) == (q["mask_count"] or 0), "missing initial mask rows")
    for i, row in enumerate(rows):
        count = row["query_count"]
        require(
            row["mask_index"] == i
            and row["query_start"] == cursor
            and row["query_stop"] == cursor + count
            and 0 <= count <= 64
            and np.all(origins[cursor : cursor + count] == i),
            "query group order differs",
        )
        indices.extend(range(count))
        cursor += count
    require(cursor == len(xy), "query rows do not cover points")
    arrays = dict(
        xy=xy.copy(),
        mask_index=origins.copy(),
        corner_index=np.asarray(indices, np.int64),
        frame_idx=np.asarray(START, np.int64),
        available=np.asarray(q["available"]),
    )
    meta = {k: q[k] for k in ("rows", "mask_count", "available", "config")}
    unique = len(np.unique(xy, axis=0))
    meta.update(query_count=len(xy), unique_xy_count=unique, repeated_xy_count=len(xy) - unique)
    return arrays, meta


def source_code():
    result = prior.source_code()
    result[str(PLAN)] = PLAN_SHA
    result[str(Path(__file__).resolve())] = sha(__file__)
    # prior.source_code includes its own immutable plan because that helper is reused.
    return result


def run(output):
    output = Path(output).resolve()
    require(output.is_relative_to(WORK) and not output.exists(), "new workspace output required")
    prior.no_model_imports()
    require(
        sha(PLAN) == PLAN_SHA
        and sha(core.__file__) == CORE_SHA
        and sha(prior.__file__) == PRIOR_SHA,
        "frozen source/plan differs",
    )
    output.mkdir(parents=True)
    started, clock = now(), time.perf_counter()
    evidence, outputs, captures, timings = {}, {}, {}, {}
    code = source_code()

    def record(path):
        outputs[path.relative_to(output).as_posix()] = dict(
            path=str(path), sha256=sha(path), size_bytes=path.stat().st_size
        )

    def write(name, value):
        path = output / name
        write_json(path, value)
        record(path)

    def dump(name, value):
        path = output / name
        with path.open("xb") as f:
            np.savez_compressed(f, **value)
        record(path)

    try:
        for path, digest in code.items():
            bind(path, digest, evidence)
            dest = output / "code_snapshot" / Path(path).relative_to(REPO)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
            require(sha(dest) == digest, "source snapshot differs")
            record(dest)
        t = time.perf_counter()
        inputs = metadata_inputs(evidence)
        prior.verify_payloads(inputs, evidence)
        for rel, digest in inputs["old_manifest"]["code_sha256"].items():
            source = OLD / "code_snapshot" / Path(rel).name
            dest = output / "historical_detector_source" / Path(rel).name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
            require(sha(dest) == digest, "historical source snapshot differs")
            record(dest)
        timings["input_verification_seconds"] = time.perf_counter() - t
        write("legacy_detector_condition.json", inputs["detector"])
        write(
            "plan.json",
            dict(
                schema=STAGE,
                plan_sha256=PLAN_SHA,
                native_interval=[START, STOP],
                frames=list(FRAMES),
                views=list(VIEWS),
                source_hashes=code,
                query_core_sha256=CORE_SHA,
                shared_SL_calibration_sha256=inputs["shared"],
                model_intent=prior.MODEL_INTENT,
                compact_seal_sha256=COMPACT_SHA,
                compact_manifest_sha256=MANIFEST_SHA,
                storage_audit_sha256=AUDIT_SHA,
                legacy_D2_seal_sha256=OLD_SHA,
                legacy_manifest_sha256=OLD_MANIFEST_SHA,
                current_context_seal_sha256=CURRENT_SHA,
                expected_initial_masks=list(INITIAL_MASK_COUNTS),
                expected_window_masks=608,
                query_upper_bound=640,
                raw_images=192,
                observations=192,
                query_files=3,
                source_video_identity=inputs["identities"],
                no_reinitialization=True,
                temporal_holdout=False,
                known_3D_gap_is_occlusion_truth=False,
            ),
        )
        for folder in ("raw", "observations", "queries"):
            (output / folder).mkdir()
        native_rows, query_rows = [], []
        for index, view in enumerate(VIEWS):
            t = time.perf_counter()
            path = native.checked_relative(COMPACT, inputs["manifest"]["masks"][view]["path"])
            with np.load(path, allow_pickle=False) as z:
                require(len(z.files) == len(set(z.files)), "duplicate packed keys")
                data = {k: z[k].copy() for k in z.files}
            require(
                set(data) == set(native.SCHEMA_DTYPES)
                and all(data[k].dtype == dt for k, dt in native.SCHEMA_DTYPES.items())
                and np.array_equal(data["frame_idx"], np.arange(START, SOURCE_STOP))
                and np.array_equal(data["image_shape_hw"], [1200, 1200]),
                "legacy packed schema differs",
            )
            capture = cv2.VideoCapture(str(inputs["videos"][view]))
            captures[view] = capture
            for n, image in read64(
                capture, inputs["manifest"]["raw"][view]["frame_bgr_sha256"][:64]
            ):
                count = inputs["manifest"]["masks"][view]["instances_per_frame"][n - START]
                obs = observation(data, n, count)
                if n == START:
                    arrays, meta = queries(image, obs)
                    require(
                        meta["mask_count"] == INITIAL_MASK_COUNTS[index], "initial10 masks differ"
                    )
                    name = f"queries/{view}.npz"
                    dump(name, arrays)
                    query_rows.append(
                        dict(
                            view=view,
                            frame_idx=n,
                            path=name,
                            sha256=outputs[name]["sha256"],
                            array_signatures={k: signature(v) for k, v in arrays.items()},
                            **meta,
                        )
                    )
                pngname, obsname = (
                    f"raw/raw_{n}_{view}.png",
                    f"observations/observation_{n}_{view}.npz",
                )
                png = output / pngname
                require(cv2.imwrite(str(png), image), "PNG write failed")
                record(png)
                restored = cv2.imread(str(png), cv2.IMREAD_COLOR)
                require(
                    restored is not None and signature(restored) == signature(image),
                    "PNG roundtrip differs",
                )
                dump(obsname, obs)
                native_rows.append(
                    dict(
                        frame_idx=n,
                        view=view,
                        mask_count=count,
                        mask_available=bool(obs["mask_available"]),
                        bgr_sha256=signature(image)["sha256"],
                        mask_sha256=signature(obs["masks"])["sha256"],
                        native_offset=n - START,
                        png_path=pngname,
                        png_sha256=outputs[pngname]["sha256"],
                        observation_path=obsname,
                        observation_sha256=outputs[obsname]["sha256"],
                        manifest_sha256=MANIFEST_SHA,
                        observation_array_signatures={k: signature(v) for k, v in obs.items()},
                    )
                )
            capture.release()
            del captures[view]
            timings[view + "_prepare_seconds"] = time.perf_counter() - t
            print(
                json.dumps(
                    dict(
                        event="known_gap_view_prepared",
                        view=view,
                        frames=64,
                        queries=query_rows[-1]["query_count"],
                    )
                ),
                flush=True,
            )
        native_rows.sort(key=lambda r: (r["frame_idx"], VIEWS.index(r["view"])))
        require(
            [(r["frame_idx"], r["view"]) for r in native_rows]
            == [(n, v) for n in FRAMES for v in VIEWS]
            and sum(r["mask_count"] for r in native_rows) == 608
            and [r["mask_count"] for r in query_rows] == list(INITIAL_MASK_COUNTS)
            and sum(r["query_count"] for r in query_rows) <= 640,
            "fixed192/10/608 accounting differs",
        )
        write("native_evidence.json", native_rows)
        write("query_rows.json", query_rows)
        prior.no_model_imports()
        require(code == source_code(), "loaded source set differs")
        t = time.perf_counter()
        for path, digest in evidence.items():
            require(sha(path) == digest, "input changed at final guard: " + path)
        source_identity(inputs["manifest"], baseline=inputs["identities"])
        timings["final_input_guard_seconds"] = time.perf_counter() - t
        timings["total_before_summary_seconds"] = time.perf_counter() - clock
        result = dict(
            schema=STAGE,
            status="ok",
            frames=list(FRAMES),
            views=list(VIEWS),
            native_interval=[START, STOP],
            native_frames=64,
            raw_video_delivered_images=192,
            raw_png_roundtrip_decodes=192,
            decoded_mask_stacks=192,
            mask_decoder_calls=384,
            initial_mask_rows=10,
            total_masks=608,
            query_files=3,
            query_count=sum(r["query_count"] for r in query_rows),
            queries_per_view={r["view"]: r["query_count"] for r in query_rows},
            model_loaded=False,
            model_forward_calls=0,
            GPU=False,
            SL_arrays_decoded=False,
            candidate_arrays_read=False,
            legacy_AVI_payloads_read=False,
            new_D2_inference=False,
            ID_or_review_changed=False,
            temporal_holdout=False,
            known_3D_gap_is_occlusion_truth=False,
            decoder_internal_keyframe_work_not_counted=True,
            timings=timings,
            legacy_D2_condition=inputs["detector"],
            model_intent=prior.MODEL_INTENT,
        )
        write("summary.json", result)
        prior.output_guard(output, outputs)
        write_json(
            output / "preparation.provenance.json",
            dict(
                stage=STAGE,
                status="ok",
                started_at=started,
                finished_at=now(),
                inputs=evidence,
                outputs=outputs,
                environment=dict(
                    python=sys.version,
                    numpy=np.__version__,
                    opencv=cv2.__version__,
                    opencv_threads=cv2.getNumThreads(),
                    model_imported=False,
                ),
            ),
        )
        return result
    except BaseException as exc:
        write_json(
            output / "preparation.provenance.json",
            dict(
                stage=STAGE,
                status="failed",
                started_at=started,
                finished_at=now(),
                inputs=evidence,
                outputs=outputs,
                error_type=type(exc).__name__,
                error=str(exc),
                timings=timings,
            ),
        )
        raise
    finally:
        for capture in captures.values():
            capture.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    result = run(parser.parse_args().output)
    print(
        json.dumps({k: result[k] for k in ("status", "native_frames", "query_count")}), flush=True
    )


if __name__ == "__main__":
    main()
