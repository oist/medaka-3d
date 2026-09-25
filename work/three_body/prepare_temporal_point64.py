"""Freeze 192 native images/all masks and three anonymous first-frame query sets.

Model-free preparation for the fixed [66483,66547) diagnostic. Only selected
successful input evidence is consumed; candidate/checkpoint/model arrays are
never opened. The pinned earlier preparation supplies three first-frame checks,
not substitute frames. Actual decoding requires a separately reviewed launch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import temporal_point_core as core

REPO = Path(__file__).resolve().parents[2]
WORK = REPO / "work/three_body"
sys.path[:0] = [str(REPO / "src"), str(REPO / "labs/20260909_three_body")]
import long_final_native_samples as native  # noqa: E402
from source_sequence import validate_source_manifest  # noqa: E402

VIEWS = ("top_cam", "side1_cam", "side2_cam")
START, STOP, CHUNK_START, CHUNK_STOP = 66483, 66547, 66400, 68060
FRAMES = tuple(range(START, STOP))
STAGE = "temporal_point64_preparation_v1"
PLAN = WORK / "TEMPORAL_POINT_64_PLAN.md"
PLAN_SHA = "f72e37547e0f9078446927aa85ea15bdb825188b76d62eedeb8f7db4ace4990c"
CORE_SHA = "3ec308ca7877338f8677c0703d6f180a9d5658c4858b9868451e1152b7d4e25b"
PREP = WORK / "dino_minute_adjacent_crop_preparation_v1"
PREP_SHA = "292b0805d2a15bf04fd839cf68357332ab6dedd6b669477fb98b311aa5ee6914"
ORIGINAL = WORK / "long_remaining_supervisor_v1/phases/f066400-076360/tracking"
ORIGINAL_SHA = "95098e39bbd4d1e09d7ef98e3a7c244fe89ab2caf36a3650f360d35fc8289382"
CONSUMED_SHA = "8d5af23bfa3c42c3de4377a19234e20843632c6757199a579281e75c6c1ae905"
MANIFEST_SHA = "cb81c12a476bc4b4744b1d4c763090e1c5d288ef1b13eae6aef9320963bc7f8d"
CHUNK_SHA = "8ecc6d0c1b9b547b326f2bd55ae6256267b734de47525f533ba8e759ac106e0f"
MODEL_INTENT = dict(
    family="CoTracker3",
    variant="scaled_online",
    model_used_for_queries=False,
    assets_consumed=False,
    exact_asset_binding_stage="later inference stage",
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def now():
    return datetime.now(UTC).isoformat()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def signature(array):
    return dict(
        dtype=array.dtype.str,
        shape=list(array.shape),
        sha256=hashlib.sha256(array.tobytes()).hexdigest(),
    )


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def bind(path, digest, evidence):
    path = Path(path).resolve(strict=True)
    key = str(path)
    require(key not in evidence or evidence[key] == digest, "conflicting input SHA")
    if key not in evidence:
        require(sha(path) == digest, "input bytes changed: " + key)
        evidence[key] = digest
    return path


def members(record, group):
    """Normalize once, avoiding a quadratic full-path scan for every reference."""
    result = {}
    for key, row in record[group].items():
        if isinstance(row, str):
            path, digest, size = Path(key).resolve(), row, None
        else:
            path, digest = Path(row["path"]).resolve(), row["sha256"]
            size = row.get("size", row.get("size_bytes"))
        item = dict(path=str(path), sha256=digest, size=size)
        require(str(path) not in result or result[str(path)] == item, "conflicting evidence roles")
        result[str(path)] = item
    return result


def member(table, path, expected=None):
    key = str(Path(path).resolve())
    require(key in table, "required input/output membership absent: " + key)
    item = table[key]
    require(expected is None or expected == item["sha256"], "member SHA differs")
    return item


def header(path, digest, stage, evidence):
    value = read_json(bind(path, digest, evidence))
    require(
        value.get("stage") == stage and value.get("status") == "ok" and value.get("finished_at"),
        "successful pinned stage required",
    )
    return value


def selected_record(records):
    require(isinstance(records, list) and len(records) == 16, "original16 prefix required")
    for i, row in enumerate(records):
        require(
            type(row.get("index")) is int
            and row["index"] == i
            and type(row.get("start")) is int
            and row["start"] == 49800 + i * 1660
            and type(row.get("stop")) is int
            and row["stop"] == 49800 + (i + 1) * 1660,
            "consumed native order differs",
        )
    row = records[10]
    require(
        row["manifest_sha256"] == MANIFEST_SHA and row["provenance_sha256"] == CHUNK_SHA,
        "selected chunk changed",
    )
    return row


def first_rows(rows):
    require(
        [(r["frame_idx"], r["view"]) for r in rows]
        == [(66483 + 166 * i, v) for i in range(60) for v in VIEWS],
        "prior fixed60 native rows differ",
    )
    result = {r["view"]: r for r in rows[:3]}
    require(
        [result[v]["mask_count"] for v in VIEWS] == [4, 5, 4], "registered first13 mask rows differ"
    )
    return result


def raw_identity(root, row, manifest):
    videos = {}
    for view in VIEWS:
        record = row["raw_hardlinks"][view]
        source = Path(record["source_path"]).resolve(strict=True)
        linked = native.checked_relative(root, manifest["raw"][view]["path"])
        stat = source.stat()
        require(
            linked == Path(record["path"]).resolve()
            and os.path.samefile(source, linked)
            and [stat.st_size, stat.st_dev, stat.st_ino, stat.st_mtime_ns]
            == [record["bytes"], record["device"], record["inode"], record["mtime_ns"]]
            and record["sha256"] == manifest["raw"][view]["sha256"],
            "source hardlink identity differs",
        )
        videos[view] = source
    return videos


def metadata_inputs(evidence):
    """Small metadata/source only; return payload references without opening them."""
    prep = header(
        PREP / "preparation.provenance.json",
        PREP_SHA,
        "dino_minute_adjacent_crop_preparation_v1",
        evidence,
    )
    original = header(
        ORIGINAL / "continuation.provenance.json",
        ORIGINAL_SHA,
        "three_body_long_continuation",
        evidence,
    )
    pi, po = members(prep, "inputs"), members(prep, "outputs")
    oi, oo = members(original, "inputs"), members(original, "outputs")
    member(pi, ORIGINAL / "continuation.provenance.json", ORIGINAL_SHA)
    fixed = original["params"]["bindings"]["inputs"]
    require(original["params"]["through_frame"] == 76360, "original horizon differs")
    consumed = ORIGINAL / "consumed_chunks.json"
    member(oo, consumed, CONSUMED_SHA)
    member(pi, consumed, CONSUMED_SHA)
    row = selected_record(read_json(bind(consumed, CONSUMED_SHA, evidence)))
    root = native.checked_relative(fixed["input_stage_root"], row["attempt_path"])
    member(pi, root / "staging.provenance.json", CHUNK_SHA)
    chunk = header(
        root / "staging.provenance.json", CHUNK_SHA, "prepare_independent_mask_chunk", evidence
    )
    ci, co = members(chunk, "inputs"), members(chunk, "outputs")
    member(pi, root / "manifest.json", MANIFEST_SHA)
    member(co, root / "manifest.json", MANIFEST_SHA)
    manifest = read_json(bind(root / "manifest.json", MANIFEST_SHA, evidence))
    require(
        manifest.get("experiment") == native.EXPERIMENT
        and manifest.get("frames") == list(range(CHUNK_START, CHUNK_STOP))
        and manifest.get("fps") == 166
        and manifest.get("image_shape_hw") == [1200, 1200]
        and manifest.get("binding") == fixed["detector_binding"] == chunk["params"]["binding"]
        and manifest.get("tracking_performed") is False
        and manifest.get("human_truth_created") is False,
        "D2/source protocol differs",
    )
    validate_source_manifest(root, manifest)
    protocol = Path(fixed["protocol_path"])
    member(ci, protocol, fixed["protocol_sha256"])
    member(oi, protocol, fixed["protocol_sha256"])
    bind(protocol, fixed["protocol_sha256"], evidence)
    member(co, root / "protocol.json", fixed["protocol_sha256"])
    bind(root / "protocol.json", fixed["protocol_sha256"], evidence)
    for rel, digest in fixed["detector_binding"]["code_sha256"].items():
        bind(REPO / rel, digest, evidence)
        snapshot = root / "code_snapshot" / rel
        member(co, snapshot, digest)
        bind(snapshot, digest, evidence)
    plan_path = PREP / "plan.json"
    prep_plan = read_json(bind(plan_path, member(po, plan_path)["sha256"], evidence))
    shared = fixed["seed"]["SL_and_calibration_sha256"]
    require(
        shared == prep_plan["shared_SL_calibration_sha256"]
        and len(shared) == 7
        and sum(Path(p).suffix == ".npz" for p in shared) == 3
        and sum(Path(p).suffix == ".json" for p in shared) == 4,
        "SL3/calibration4 differ",
    )
    references = []
    for path, digest in shared.items():
        member(oi, path, digest)
        references.append(member(pi, path, digest))
    videos = raw_identity(root, row, manifest)
    for view in VIEWS:
        info = manifest["raw"][view]
        hashes = info["frame_bgr_sha256"]
        require(
            len(hashes) == 1660
            and all(
                isinstance(s, str) and len(s) == 64 and set(s) <= set("0123456789abcdef")
                for s in hashes
            ),
            "full chunk native BGR SHA records required",
        )
        for category, role in (("raw", view + "_raw"), ("masks", view + "_masks")):
            descriptor = manifest[category][view]
            path = native.checked_relative(root, descriptor["path"])
            recorded = row["output_hashes"][role]
            require(
                Path(recorded["path"]).resolve() == path
                and recorded["sha256"] == descriptor["sha256"],
                "consumed payload differs",
            )
            item = member(co, path, descriptor["sha256"])
            require(path.stat().st_size == item["size"], "selected payload size changed")
            if category == "raw":
                item = member(pi, videos[view], descriptor["sha256"])
                member(ci, videos[view], descriptor["sha256"])
            else:
                member(pi, path, descriptor["sha256"])
                member(oi, path, descriptor["sha256"])
            references.append(item)
        counts = manifest["masks"][view]["instances_per_frame"]
        require(
            len(counts) == 1660 and all(type(x) is int and x >= 0 for x in counts),
            "full mask count manifest required",
        )
    native_path = PREP / "native_evidence.json"
    first = first_rows(read_json(bind(native_path, member(po, native_path)["sha256"], evidence)))
    for view, item in first.items():
        require(
            item["manifest_sha256"] == MANIFEST_SHA
            and item["chunk_index"] == 10
            and item["native_offset"] == START - CHUNK_START
            and item["bgr_sha256"] == manifest["raw"][view]["frame_bgr_sha256"][83]
            and item["mask_count"] == manifest["masks"][view]["instances_per_frame"][83],
            "prior first frame source differs",
        )
        for key, digest_key in (
            ("png_path", "png_sha256"),
            ("observation_path", "observation_sha256"),
        ):
            path = native.checked_relative(PREP, item[key])
            references.append(member(po, path, item[digest_key]))
    # Historical pinning applies only to reused helpers, not unrelated producers.
    member(
        pi,
        Path(native.__file__),
        "f9c82e6fc7150edd20f424dd9fa75fda2ef321ad10f41076744ef7a7689e7791",
    )
    bind(Path(native.__file__), member(pi, Path(native.__file__))["sha256"], evidence)
    return dict(
        root=root,
        row=row,
        manifest=manifest,
        videos=videos,
        shared=shared,
        references=references,
        first=first,
    )


def verify_payloads(inputs, evidence):
    for row in inputs["references"]:
        path = bind(row["path"], row["sha256"], evidence)
        require(
            row["size"] is None or path.stat().st_size == row["size"],
            "selected payload size differs",
        )


def read64(capture, hashes, *, shape=(1200, 1200)):
    """One absolute seek then exactly64 sequential deliveries; no EOF probe."""
    require(len(hashes) == 64, "fixed64 hashes required")
    native.validate_capture(capture, shape=shape)
    require(capture.set(cv2.CAP_PROP_POS_FRAMES, START), "absolute source seek failed")
    require(
        abs(capture.get(cv2.CAP_PROP_POS_FRAMES) - START) <= 1e-6, "source seek position differs"
    )
    for n, expected in zip(FRAMES, hashes, strict=True):
        ok, image = capture.read()
        require(
            ok
            and isinstance(image, np.ndarray)
            and image.dtype == np.uint8
            and image.shape == (*shape, 3),
            "source EOF/shape/dtype differs",
        )
        require(
            abs(capture.get(cv2.CAP_PROP_POS_FRAMES) - n - 1) <= 1e-6,
            "source native decode position differs",
        )
        require(signature(image)["sha256"] == expected, "native BGR SHA differs")
        yield n, image


def observation(data, n, count):
    packed, masks = native.select_packed_frame(data, n - CHUNK_START, n, count)
    available = masks is not None
    return dict(
        frame_idx=np.asarray(n, np.int64),
        masks=masks if available else np.empty((0, 1200, 1200), bool),
        mask_available=np.asarray(available),
        **{"packed_" + k: v for k, v in packed.items()},
    )


def verify_first(image, obs, row):
    """Compare only reused arrays; detections remain unloaded SHA-only evidence."""
    prior_image = cv2.imread(str(PREP / row["png_path"]), cv2.IMREAD_COLOR)
    require(
        prior_image is not None and signature(prior_image) == signature(image),
        "first PNG pixels differ",
    )
    with np.load(PREP / row["observation_path"], allow_pickle=False) as z:
        require(len(z.files) == len(set(z.files)), "duplicate prior observation fields")
        for key, value in obs.items():
            if key != "frame_idx":
                require(
                    key in z.files
                    and signature(z[key])
                    == signature(value)
                    == row["observation_array_signatures"][key],
                    "first observation differs",
                )


def make_queries(image, obs):
    before = {k: signature(v) for k, v in obs.items()}
    image_before = signature(image)
    result = core.select_queries(image, obs["masks"] if bool(obs["mask_available"]) else None)
    require(
        image_before == signature(image) and before == {k: signature(v) for k, v in obs.items()},
        "query selection mutated native inputs",
    )
    xy, origins, rows = result["xy"], result["mask_index"], result["rows"]
    require(
        result["available"] == bool(obs["mask_available"])
        and result["mask_count"] == (len(obs["masks"]) if result["available"] else None),
        "query availability/count changed",
    )
    require(
        xy.dtype == np.float32
        and xy.shape == (len(origins), 2)
        and origins.dtype == np.int64
        and origins.ndim == 1
        and np.isfinite(xy).all(),
        "query schema differs",
    )
    require(len(rows) == (result["mask_count"] or 0), "missing initial mask rows")
    cursor, indices = 0, []
    for i, row in enumerate(rows):
        count = row["query_count"]
        require(
            row["mask_index"] == i
            and row["query_start"] == cursor
            and row["query_stop"] == cursor + count
            and 0 <= count <= 64
            and np.all(origins[cursor : cursor + count] == i),
            "query order differs",
        )
        indices.extend(range(count))
        cursor += count
    require(cursor == len(xy), "query rows do not cover all points")
    arrays = dict(
        xy=xy.copy(),
        mask_index=origins.copy(),
        corner_index=np.asarray(indices, np.int64),
        frame_idx=np.asarray(START, np.int64),
        available=np.asarray(result["available"]),
    )
    metadata = {k: result[k] for k in ("rows", "mask_count", "available", "config")}
    metadata.update(
        query_count=len(xy),
        unique_xy_count=len(np.unique(xy, axis=0)),
        repeated_xy_count=len(xy) - len(np.unique(xy, axis=0)),
    )
    return arrays, metadata


def no_model_imports():
    require(
        not any(
            k.split(".")[0] in {"torch", "cotracker", "dinov2", "detectron2"} for k in sys.modules
        ),
        "model/Torch import forbidden in preparation",
    )


def source_code():
    paths = {Path(__file__).resolve(), PLAN, Path(core.__file__).resolve()}
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.is_relative_to(REPO) and path.suffix == ".py" and ".venv" not in path.parts:
                paths.add(path)
    return {str(p): sha(p) for p in sorted(paths)}


def output_guard(output, outputs):
    for item in outputs.values():
        path = Path(item["path"])
        require(
            path.stat().st_size == item["size_bytes"] and sha(path) == item["sha256"],
            "output changed before seal",
        )
    require(
        {p.resolve() for p in output.rglob("*") if p.is_file()}
        == {Path(r["path"]) for r in outputs.values()},
        "unexpected output membership",
    )


def run(output):
    output = Path(output).resolve()
    require(output.is_relative_to(WORK) and not output.exists(), "new workspace output required")
    no_model_imports()
    require(
        sha(PLAN) == PLAN_SHA and sha(Path(core.__file__)) == CORE_SHA,
        "plan/core is not the reviewed frozen source",
    )
    output.mkdir(parents=True)
    evidence, outputs, timings, captures = {}, {}, {}, {}
    started, start = now(), time.perf_counter()
    code = source_code()

    def record(path):
        outputs[path.relative_to(output).as_posix()] = dict(
            path=str(path), sha256=sha(path), size_bytes=path.stat().st_size
        )

    def write(name, value):
        path = output / name
        write_json(path, value)
        record(path)

    def dump(name, arrays):
        path = output / name
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
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
        verify_payloads(inputs, evidence)
        timings["input_verification_seconds"] = time.perf_counter() - t
        write(
            "plan.json",
            dict(
                schema=STAGE,
                plan_sha256=PLAN_SHA,
                native_interval=[START, STOP],
                frames=list(FRAMES),
                views=list(VIEWS),
                model_intent=MODEL_INTENT,
                shared_SL_calibration_sha256=inputs["shared"],
                source_hashes=code,
                query_core_sha256=CORE_SHA,
                raw_images=192,
                observations=192,
                query_files=3,
                query_upper_bound=832,
                no_reinitialization=True,
                temporal_holdout=False,
            ),
        )
        for folder in ("raw", "observations", "queries"):
            (output / folder).mkdir()
        native_rows, query_rows = [], []
        for view in VIEWS:
            t = time.perf_counter()
            path = native.checked_relative(
                inputs["root"], inputs["manifest"]["masks"][view]["path"]
            )
            with np.load(path, allow_pickle=False) as z:
                require(len(z.files) == len(set(z.files)), "duplicate packed NPZ keys")
                data = {k: z[k].copy() for k in z.files}
            require(
                set(data) == set(native.SCHEMA_DTYPES)
                and all(data[k].dtype == dt for k, dt in native.SCHEMA_DTYPES.items())
                and np.array_equal(data["frame_idx"], np.arange(CHUNK_START, CHUNK_STOP))
                and np.array_equal(data["image_shape_hw"], [1200, 1200]),
                "packed schema differs",
            )
            capture = cv2.VideoCapture(str(inputs["videos"][view]))
            captures[view] = capture
            hashes = inputs["manifest"]["raw"][view]["frame_bgr_sha256"][83:147]
            for n, image in read64(capture, hashes):
                count = inputs["manifest"]["masks"][view]["instances_per_frame"][n - CHUNK_START]
                obs = observation(data, n, count)
                if n == START:
                    verify_first(image, obs, inputs["first"][view])
                    queries, metadata = make_queries(image, obs)
                    require(
                        metadata["mask_count"] == inputs["first"][view]["mask_count"],
                        "initial13 query mask denominator differs",
                    )
                    name = f"queries/{view}.npz"
                    dump(name, queries)
                    query_rows.append(
                        dict(
                            view=view,
                            frame_idx=n,
                            path=name,
                            sha256=outputs[name]["sha256"],
                            array_signatures={k: signature(v) for k, v in queries.items()},
                            **metadata,
                        )
                    )
                pngname = f"raw/raw_{n}_{view}.png"
                png = output / pngname
                require(cv2.imwrite(str(png), image), "PNG write failed")
                record(png)
                decoded = cv2.imread(str(png), cv2.IMREAD_COLOR)
                require(
                    decoded is not None and signature(decoded) == signature(image),
                    "lossless PNG roundtrip differs",
                )
                obsname = f"observations/observation_{n}_{view}.npz"
                dump(obsname, obs)
                native_rows.append(
                    dict(
                        frame_idx=n,
                        view=view,
                        mask_count=count,
                        mask_available=bool(obs["mask_available"]),
                        bgr_sha256=signature(image)["sha256"],
                        mask_sha256=signature(obs["masks"])["sha256"],
                        png_path=pngname,
                        png_sha256=outputs[pngname]["sha256"],
                        observation_path=obsname,
                        observation_sha256=outputs[obsname]["sha256"],
                        native_offset=n - CHUNK_START,
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
                        event="view_prepared",
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
            == [(n, v) for n in FRAMES for v in VIEWS],
            "all192 native order differs",
        )
        require(
            sum(r["mask_count"] for r in query_rows) == 13
            and sum(r["query_count"] for r in query_rows) <= 832,
            "initial query accounting",
        )
        write("native_evidence.json", native_rows)
        write("query_rows.json", query_rows)
        no_model_imports()
        require(code == source_code(), "loaded source set changed")
        t = time.perf_counter()
        for path, digest in evidence.items():
            require(sha(path) == digest, "input changed at final guard: " + path)
        raw_identity(inputs["root"], inputs["row"], inputs["manifest"])
        timings["final_input_guard_seconds"] = time.perf_counter() - t
        timings["total_before_summary_seconds"] = time.perf_counter() - start
        summary = dict(
            schema=STAGE,
            status="ok",
            frames=list(FRAMES),
            views=list(VIEWS),
            native_interval=[START, STOP],
            native_frames=64,
            raw_video_delivered_images=192,
            raw_png_roundtrip_decodes=192,
            decoded_mask_stacks=192,
            initial_mask_rows=13,
            mask_decoder_calls=384,
            query_count=sum(r["query_count"] for r in query_rows),
            queries_per_view={r["view"]: r["query_count"] for r in query_rows},
            query_files=3,
            model_loaded=False,
            model_forward_calls=0,
            GPU=False,
            SL_arrays_decoded=False,
            candidate_arrays_read=False,
            ID_or_review_changed=False,
            temporal_holdout=False,
            decoder_internal_keyframe_work_not_counted=True,
            selected_mask_validation=(
                "192 selected frame stacks; unrelated chunk frames not decoded"
            ),
            source_scope=(
                "selected sealed inputs only; no whole producer candidate/checkpoint rehash"
            ),
            first_observation_scope=(
                "masks/availability/packed13 exact; detections never deserialized"
            ),
            timings=timings,
            model_intent=MODEL_INTENT,
        )
        write("summary.json", summary)
        output_guard(output, outputs)
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
        return summary
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
