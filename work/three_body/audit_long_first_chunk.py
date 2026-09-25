"""Independent CPU audit of the first completed 1660-frame long-input chunk.

Decode packed masks without the producer decoder. Hash every canonical raw file
once, verify its hardlink aliases, and guard its size/mtime/file identity at exit.
No detector, pose, SLEAP, calibration, human label, or tracking output is loaded.
"""

import hashlib
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

from medaka3d.provenance import provenance

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "labs/20260909_three_body"
ROOT = REPO / "work/three_body/inputs/long_source_masks_49800-348600_v1"
CHUNK = ROOT / "chunks/f049800-051460/attempt_0001"
OUT = REPO / "work/three_body/long_first_chunk_independent_audit_v1"
BACKGROUND_AUDIT = REPO / "work/three_body/long_background_audit"
VIEWS = ("top_cam", "side1_cam", "side2_cam")
FRAMES = np.arange(49800, 51460, dtype=np.int64)
PROBES = (49800, 50630, 51459)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def array_sha(value):
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stat_key(path):
    value = Path(path).stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


class Evidence:
    def __init__(self):
        self.paths = {}
        self.files = {}
        self.raw_identities = set()

    def add(self, path, expected=None, *, raw=False):
        path = Path(path).resolve(strict=True)
        before = stat_key(path)
        identity = before[:2]
        if identity in self.files:
            old = self.files[identity]
            require(os.path.samefile(path, old["path"]), "file identity collision")
            require(before == old["stat"], "bound file changed")
            actual = old["sha256"]
        else:
            actual = sha(path)
            require(stat_key(path) == before, "file changed during hash")
            self.files[identity] = {"path": str(path), "stat": before, "sha256": actual}
        require(expected is None or actual == expected, "SHA differs: " + str(path))
        if raw:
            self.raw_identities.add(identity)
        self.paths[str(path)] = {"sha256": actual, "stat": before}
        return actual

    def final_guard(self):
        for path, item in self.paths.items():
            require(stat_key(path) == item["stat"], "evidence stat changed: " + path)
        # Raw full files are deliberately read only once, including all aliases.
        for identity, item in self.files.items():
            if identity not in self.raw_identities:
                require(sha(item["path"]) == item["sha256"], "non-raw evidence changed")


def check_masks(path, details):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    types = {
        "image_shape_hw": "int32",
        "frame_idx": "int64",
        "frame_offsets": "int64",
        "mask_boxes": "int32",
        "mask_byte_offsets": "int64",
        "mask_bits": "uint8",
        "scores": "float32",
        "pred_boxes": "float32",
        "classes": "int64",
        "instance_id": "int64",
        "source_track_id": "int64",
        "view_available": "bool",
        "frame_mask_sha256": "S64",
    }
    require(set(data) == set(types), "packed schema differs")
    for key, dtype in types.items():
        require(data[key].dtype == np.dtype(dtype), "packed dtype differs: " + key)
    require(np.array_equal(data["image_shape_hw"], [1200, 1200]), "mask image shape")
    require(np.array_equal(data["frame_idx"], FRAMES), "mask native frames")
    count = len(data["scores"])
    shapes = {
        "frame_offsets": (1661,),
        "mask_boxes": (count, 4),
        "mask_byte_offsets": (count + 1,),
        "scores": (count,),
        "pred_boxes": (count, 4),
        "classes": (count,),
        "instance_id": (count,),
        "source_track_id": (count,),
        "view_available": (1660,),
        "frame_mask_sha256": (1660,),
    }
    for key, shape in shapes.items():
        require(data[key].shape == shape, "packed shape differs: " + key)
    offsets, byte_offsets = data["frame_offsets"], data["mask_byte_offsets"]
    require(offsets[0] == 0 and offsets[-1] == count, "frame offset endpoints")
    require(np.all(np.diff(offsets) >= 0), "frame offset ordering")
    require(byte_offsets[0] == 0 and byte_offsets[-1] == len(data["mask_bits"]), "bit endpoints")
    require(np.all(np.diff(byte_offsets) >= 0), "bit offset ordering")
    counts = np.diff(offsets)
    require(counts.tolist() == details["instances_per_frame"], "manifest counts differ")
    require(details["roundtrip_verified_frames"] == 1660, "producer roundtrip coverage")
    require(np.all(data["source_track_id"] == -1), "persistent IDs found in masks")
    require(np.all(data["view_available"]), "unexpected unavailable producer view")
    require(np.isfinite(data["scores"]).all(), "nonfinite scores")
    require(np.all((data["scores"] >= 0.5) & (data["scores"] <= 1)), "score threshold")
    require(np.isfinite(data["pred_boxes"]).all(), "nonfinite detector boxes")
    require(np.all(data["classes"] >= 0), "negative detector class")
    empty_instances, foreground_pixels = 0, 0
    for time_index in range(1660):
        lo, hi = map(int, offsets[time_index : time_index + 2])
        require(np.array_equal(data["instance_id"][lo:hi], np.arange(hi - lo)), "local order")
        full = np.zeros((hi - lo, 1200, 1200), dtype=bool)
        for local, index in enumerate(range(lo, hi)):
            x0, y0, x1, y1 = map(int, data["mask_boxes"][index])
            require(0 <= x0 <= x1 <= 1200 and 0 <= y0 <= y1 <= 1200, "mask bounds")
            a, b = map(int, byte_offsets[index : index + 2])
            n = (x1 - x0) * (y1 - y0)
            require(b - a == (n + 7) // 8, "packed bit count")
            packed = data["mask_bits"][a:b]
            bits = np.unpackbits(packed, bitorder="little")
            require(not bits[n:].any(), "nonzero packed padding")
            crop = bits[:n].reshape(y1 - y0, x1 - x0).astype(bool)
            require(
                np.array_equal(np.packbits(crop.reshape(-1), bitorder="little"), packed),
                "bit roundtrip differs",
            )
            if n:
                require(
                    crop[0].any() and crop[-1].any() and crop[:, 0].any() and crop[:, -1].any(),
                    "box is not tight",
                )
            else:
                require([x0, y0, x1, y1] == [0, 0, 0, 0], "empty mask box differs")
                empty_instances += 1
            full[local, y0:y1, x0:x1] = crop
        require(
            array_sha(full) == data["frame_mask_sha256"][time_index].decode(),
            "independent full-image decoded mask SHA differs",
        )
        foreground_pixels += int(full.sum())
    timing = np.asarray(details["inference_s_by_frame"])
    require(
        timing.shape == (1660,) and np.isfinite(timing).all() and np.all(timing > 0),
        "inference timing records differ",
    )
    return {
        "frames": 1660,
        "instances": count,
        "min_instances": int(counts.min()),
        "max_instances": int(counts.max()),
        "zero_detection_frames": int((counts == 0).sum()),
        "empty_instances": empty_instances,
        "classes": np.unique(data["classes"]).tolist(),
        "available_frames": int(data["view_available"].sum()),
        "all_source_track_id_minus1": True,
        "independent_decoded_sha_matches": 1660,
        "sum_instance_foreground_pixels": foreground_pixels,
        "inference_seconds_sum": float(timing.sum()),
        "inference_seconds_median": float(np.median(timing)),
        "inference_seconds_p95": float(np.percentile(timing, 95)),
        "array_records": {
            k: {"shape": list(v.shape), "dtype": v.dtype.str, "sha256": array_sha(v)}
            for k, v in data.items()
        },
    }


def read_native(capture, frame):
    require(capture.set(cv2.CAP_PROP_POS_FRAMES, frame), "native seek failed")
    require(capture.get(cv2.CAP_PROP_POS_FRAMES) == frame, "native pre-read index differs")
    ok, value = capture.read()
    require(ok and value.dtype == np.uint8 and value.shape == (1200, 1200, 3), "raw decode")
    require(capture.get(cv2.CAP_PROP_POS_FRAMES) == frame + 1, "native post-read index differs")
    return value


def check_images(view, manifest, old_background, evidence):
    raw = manifest["raw"][view]
    capture = cv2.VideoCapture(str(CHUNK / raw["path"]))
    require(capture.isOpened(), "cannot open native raw")
    try:
        require(capture.get(cv2.CAP_PROP_FPS) == 166, "native fps")
        require(capture.get(cv2.CAP_PROP_FRAME_COUNT) == 348600, "native source count")
        require(capture.get(cv2.CAP_PROP_FRAME_WIDTH) == 1200, "native width")
        require(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) == 1200, "native height")
        probes = []
        for frame in PROBES:
            image = read_native(capture, frame)
            digest = array_sha(image)
            require(digest == raw["frame_bgr_sha256"][frame - 49800], "raw probe pixel SHA")
            probes.append({"frame": frame, "bgr_sha256": digest})
        old = {v["frame"]: v for v in old_background["views"][view]["frames"]}
        samples, sample_records = [], []
        for frame in manifest["background_frames"]:
            value = read_native(capture, frame)
            digest = array_sha(value)
            require(digest == old[frame]["bgr_sha256"], "prior native background SHA differs")
            samples.append(value)
            sample_records.append({"frame": frame, "bgr_sha256": digest})
        median = np.median(np.stack(samples), axis=0).astype(np.uint8)
        saved = cv2.imread(str(CHUNK / manifest["backgrounds"][view]["path"]))
        require(saved is not None and np.array_equal(median, saved), "background median differs")
        require(
            array_sha(median) == old_background["views"][view]["median_bgr_sha256"],
            "old background median differs",
        )
        return {
            "fps": 166,
            "source_total_frames": 348600,
            "backend": capture.getBackendName(),
            "probes": probes,
            "background_samples": sample_records,
            "background_median_bgr_sha256": array_sha(median),
            "background_median_all_pixels_exact": True,
        }
    finally:
        capture.release()


def main():
    require(not OUT.exists(), "output already exists")
    evidence = Evidence()
    evidence.add(__file__)
    evidence.add(REPO / "src/medaka3d/provenance.py")
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    with provenance("independent_long_first_chunk_audit", OUT, name="audit") as record:
        record.param("gpu_used", False)
        record.param("tracking_or_ID_evaluation", False)
        seal_path = CHUNK / "staging.provenance.json"
        chunk_seal_sha = evidence.add(seal_path)
        seal = read_json(seal_path)
        require(
            seal["status"] == "ok" and seal["stage"] == "prepare_independent_mask_chunk",
            "chunk not successfully sealed",
        )
        for role, item in seal["inputs"].items():
            evidence.add(item["path"], item["sha256"], raw=role.startswith("video:"))
            require(Path(item["path"]).stat().st_size == item["size"], "input size")
        for role, item in seal["outputs"].items():
            path = Path(item["path"]).resolve()
            require(path.is_relative_to(CHUNK), "output outside chunk")
            evidence.add(path, item["sha256"], raw=role.endswith("_raw"))
            require(path.stat().st_size == item["size"], "output size")
        manifest = read_json(CHUNK / "manifest.json")
        protocol = read_json(CHUNK / "protocol.json")
        require(manifest["binding"] == seal["params"]["binding"], "binding differs")
        require(
            manifest["binding"]["input_sha256"]
            == {k: v["sha256"] for k, v in seal["inputs"].items()},
            "input SHA roles differ",
        )
        require(manifest["frames"] == FRAMES.tolist(), "native frame sequence differs")
        require(manifest["source_interval_half_open"] == [49800, 51460], "chunk range")
        require(protocol["source_frame_range"] == [49800, 348600], "full registered range")
        require(
            protocol["chunk_frames"] == 1660 and manifest["fps"] == protocol["fps"] == 166,
            "chunk/fps differs",
        )
        require(
            manifest["image_shape_hw"] == protocol["image_shape_hw"] == [1200, 1200],
            "native dimensions differ",
        )
        require(tuple(protocol["views"]) == VIEWS, "view order differs")
        require(manifest["runtime"] == manifest["binding"]["runtime"], "runtime binding")
        require(manifest["runtime"]["detector_contract"] == protocol["detector"], "detector config")
        require(manifest["mask_schema"] == "bbox_packbits_instances_v1", "mask schema name")
        require(
            manifest["tracking_performed"] is False and manifest["human_truth_created"] is False,
            "unexpected tracking/labels",
        )
        require(
            protocol["independence"]
            == {
                "sleap_input": False,
                "seed_or_pose_input": False,
                "identity_or_human_label_input": False,
                "temporal_subsampling": False,
                "resume_scope": "independent per-image detector preprocessing only; "
                "no tracking is performed",
            },
            "input independence contract differs",
        )
        for relative, digest in manifest["binding"]["code_sha256"].items():
            evidence.add(REPO / relative, digest)
            snapshot = CHUNK / "code_snapshot" / relative
            require(
                Path(seal["outputs"]["code:" + relative]["path"]).resolve() == snapshot,
                "source snapshot role mismatch",
            )
            evidence.add(snapshot, digest)
        expected_protocol = manifest["binding"]["protocol_sha256"]
        evidence.add(CHUNK / "protocol.json", expected_protocol)
        evidence.add(LAB / "long_input_protocol.json", expected_protocol)
        evidence.add(LAB / "LONG_INPUT_PROTOCOL.md")
        cache_path = Path(seal["inputs"]["source_cache_manifest"]["path"])
        cache = read_json(cache_path)
        cache_seal = read_json(seal["inputs"]["source_cache_provenance"]["path"])
        require(cache["status"] == "verified" and cache_seal["status"] == "ok", "source cache seal")
        require(len(cache["files"]) == 8, "expected eight source files")
        for item in cache["files"].values():
            path = cache_path.parent / item["local_path"]
            evidence.add(path, item["local_full_sha256"], raw=path.suffix == ".mp4")
            require(item["source_stream_sha256"] == item["local_full_sha256"], "cache stream SHA")
            require(item["source_pre"] == item["source_post"], "original cache source stat changed")
            if item.get("trusted_expected_sha256") is not None:
                require(item["trusted_expected_sha256"] == item["local_full_sha256"], "trusted SHA")
        for role, digest in protocol["model_assets_sha256"].items():
            require(seal["inputs"]["asset:" + role]["sha256"] == digest, "registered model SHA")
        for kind in ("top", "side"):
            source = yaml.safe_load((CHUNK / f"{kind}_source_config.yaml").read_text())
            effective = yaml.safe_load((CHUNK / f"{kind}_effective_config.yaml").read_text())
            require(
                effective["MODEL"]["WEIGHTS"] == manifest["models"][kind]["weights"]["path"],
                "effective weights are not retained weights",
            )
            source["MODEL"]["WEIGHTS"] = effective["MODEL"]["WEIGHTS"]
            require(source == effective, "config differs beyond retained local weights path")
            require(effective["MODEL"]["DEVICE"] == "cuda", "detector device differs")
            require(effective["MODEL"]["ROI_HEADS"]["SCORE_THRESH_TEST"] == 0.5, "threshold")
            for part in ("weights", "config"):
                item = manifest["models"][kind][part]
                evidence.add(item["path"], item["sha256"])
                require(item["asset_manifest_entry"]["sha256"] == item["sha256"], "asset binding")
        background_frames = list(range(5000, 45001, 5000))
        require(
            manifest["background_frames"] == protocol["background_frames"] == background_frames,
            "background frame rule differs",
        )
        require(
            all(v < 49800 - protocol["background_exclusion_frames"] for v in background_frames),
            "background inside full target exclusion",
        )
        bg_seal_path = BACKGROUND_AUDIT / "audit.provenance.json"
        evidence.add(bg_seal_path)
        bg_seal = read_json(bg_seal_path)
        require(bg_seal["status"] == "ok", "old background audit incomplete")
        evidence.add(
            BACKGROUND_AUDIT / "summary.json", bg_seal["outputs"]["summary.json"]["sha256"]
        )
        background = read_json(BACKGROUND_AUDIT / "summary.json")
        require(background["protocol_sha256"] == expected_protocol, "background protocol SHA")
        views = {}
        for view in VIEWS:
            raw = manifest["raw"][view]
            source_path = Path(seal["inputs"]["video:" + view]["path"])
            require(
                os.path.samefile(source_path, CHUNK / raw["path"]), "raw is not source hardlink"
            )
            require(raw["sha256"] == protocol["source_videos_sha256"][view], "video protocol SHA")
            require(
                raw["source_offset_frames"] == 49800 and raw["source_total_frames"] == 348600,
                "raw source offset/count metadata",
            )
            require(len(raw["frame_bgr_sha256"]) == 1660, "raw pixel SHA sequence length")
            for category, label in (
                ("raw", "raw"),
                ("masks", "masks"),
                ("backgrounds", "background"),
            ):
                info = manifest[category][view]
                item = seal["outputs"][view + "_" + label]
                require(
                    (CHUNK / info["path"]).resolve() == Path(item["path"]).resolve(),
                    "view payload role mismatch",
                )
                require(info["sha256"] == item["sha256"], "view manifest digest binding")
            require(
                manifest["backgrounds"][view]["frames"] == background_frames, "per-view bg frames"
            )
            views[view] = {
                "masks": check_masks(
                    CHUNK / manifest["masks"][view]["path"], manifest["masks"][view]
                ),
                "images": check_images(view, manifest, background, evidence),
            }
            print(
                json.dumps(
                    {
                        "view": view,
                        "mask_frames_checked": 1660,
                        "raw_probes": 3,
                        "background_probes": 9,
                    }
                ),
                flush=True,
            )
        parent = ROOT / "runs/run_0001"
        evidence.add(parent / "run.provenance.json")
        parent_seal = read_json(parent / "run.provenance.json")
        evidence.add(parent / "incomplete.json", parent_seal["outputs"]["incomplete"]["sha256"])
        incomplete = read_json(parent / "incomplete.json")
        require(
            parent_seal["status"] == "failed" and "PreparationPaused" in parent_seal["error"],
            "parent did not stop for registered chunk limit",
        )
        require(
            incomplete["status"] == "incomplete"
            and incomplete["reason"] == "requested_max_new_chunks"
            and incomplete["final_sequence_path"] is None,
            "parent completion semantics",
        )
        require(
            len(incomplete["chunks"]) == 1
            and incomplete["chunks"][0]["provenance_sha256"] == chunk_seal_sha
            and incomplete["chunks"][0]["manifest_sha256"] == sha(CHUNK / "manifest.json"),
            "parent completed chunk reference",
        )
        elapsed = (
            datetime.fromisoformat(seal["finished_at"]) - datetime.fromisoformat(seal["started_at"])
        ).total_seconds()
        files = [p for p in CHUNK.rglob("*") if p.is_file()]
        raw_bytes = sum(manifest["raw"][v]["bytes"] for v in VIEWS)
        mask_bytes = sum(manifest["masks"][v]["bytes"] for v in VIEWS)
        summary = {
            "status": "ok",
            "scope": "first completed independent detector chunk only",
            "source_range": [49800, 51460],
            "native_frames": 1660,
            "native_seconds": 10,
            "view_images": 4980,
            "full_registered_range": [49800, 348600],
            "full_preparation_complete": False,
            "tracking_or_ID_evaluation": False,
            "source_inputs": len(seal["inputs"]),
            "chunk_outputs": len(seal["outputs"]),
            "producer_code_files": len(manifest["binding"]["code_sha256"]),
            "source_cache_files": 8,
            "raw_full_hash_unique_files": len(evidence.raw_identities),
            "raw_full_hash_policy": "once per file identity; hardlink aliases checked by samefile; "
            "final size/mtime/file-identity guard, not a second raw full SHA",
            "protocol_sha256": expected_protocol,
            "chunk_seal_sha256": chunk_seal_sha,
            "chunk_manifest_sha256": sha(CHUNK / "manifest.json"),
            "views": views,
            "producer_runtime": manifest["runtime"],
            "timing": {
                "chunk_seal_wall_seconds": elapsed,
                "chunk_native_fps": 1660 / elapsed,
                "chunk_view_images_per_second": 4980 / elapsed,
                "detector_sum_seconds": sum(
                    v["masks"]["inference_seconds_sum"] for v in views.values()
                ),
                "wall_scope": "chunk context: model initialization, background decode, "
                "all raw read/predict/pack/verify/save/seal; excludes earlier "
                "imports, source preflight/full hashes and runtime construction",
                "detector_scope": "predictor call, CPU transfer and CUDA synchronize; "
                "excludes raw decode, mask packing, model initialization",
                "audit_seconds": time.perf_counter() - started,
            },
            "sizes": {
                "raw_logical_bytes": raw_bytes,
                "video_payload_bytes_copied": 0,
                "mask_npz_bytes": mask_bytes,
                "chunk_all_logical_bytes": sum(p.stat().st_size for p in files),
                "chunk_non_raw_logical_bytes": sum(p.stat().st_size for p in files) - raw_bytes,
                "note": "logical file sizes; hardlinks add directory entries, "
                "not another copy of the raw payload; disk allocation not measured",
            },
            "parent": {
                "status": "incomplete",
                "reason": incomplete["reason"],
                "seal_status": "failed",
                "exception": "PreparationPaused",
                "max_new_chunks": 1,
                "completed_chunks": 1,
                "no_final_sequence_published": not (ROOT / "manifest.json").exists(),
            },
            "limitations": [
                "Only nine target images were independently re-decoded; producer stored pixel "
                "hashes for all 4980. All 4980 packed-mask frames were independently decoded.",
                "All 27 fixed background images were independently decoded and their pixel "
                "SHA matched the earlier audit; the three medians match every saved pixel.",
                "Detector inference was not rerun. File integrity and valid instances do not "
                "establish segmentation quality, fish counts or identity accuracy.",
                "The local source cache matches its original-stream SHA records. Network "
                "originals were not re-opened; this audit used the sealed local cache.",
                "One ten-second chunk is complete. Remaining 179 chunks and long tracking "
                "are not certified by this result. This is not an identity holdout.",
            ],
        }
        for path, info in evidence.paths.items():
            record.input(path, path, sha256=info["sha256"])
        target = OUT / "audit_source.py"
        shutil.copyfile(__file__, target)
        require(
            sha(target) == evidence.paths[str(Path(__file__).resolve())]["sha256"],
            "own snapshot differs",
        )
        rows = [
            "| View | Instances | Min-max/frame | Zero frames | Detector seconds |",
            "|---|---:|---:|---:|---:|",
        ]
        for view in VIEWS:
            item = views[view]["masks"]
            rows.append(
                f"| {view} | {item['instances']} | "
                f"{item['min_instances']}-{item['max_instances']} | "
                f"{item['zero_detection_frames']} | {item['inference_seconds_sum']:.3f} |"
            )
        report = (
            "# First long-input chunk: independent CPU audit\n\n"
            "The completed chunk 49800:51460 contains all 1,660 native frames in three views "
            "(10 seconds, 4,980 images). Its input, code, model and output checks passed. "
            "This certifies input integrity only; no tracking or identity score was evaluated.\n\n"
            f"The chunk seal spans {elapsed:.3f} seconds ({1660 / elapsed:.3f} native fps). "
            "It includes initialization and chunk processing; preflight and imports are outside "
            "this clock. Synchronized predictor totals appear below.\n\n"
            + "\n".join(rows)
            + "\n\nAll 4,980 packed-mask frames were independently reconstructed in native "
            "coordinates and matched their saved full-mask hashes. All source_track_id values "
            "are -1; instance_id is within-frame ordering. "
            "Counts above are detections, not fish.\n\n"
            "Native raw probes at 49800, 50630 and 51459 matched all nine saved image hashes. "
            "All 27 background probes at 5000:5000:45000 matched the earlier audit; their "
            "pixelwise medians matched all three chunk PNGs exactly.\n\n"
            f"The masks occupy {mask_bytes:,} bytes. Non-raw chunk files total "
            f"{summary['sizes']['chunk_non_raw_logical_bytes']:,} logical bytes. The "
            f"{raw_bytes:,}-byte raw payload uses verified hardlinks and was not copied. "
            "Each of the three raw files was fully hashed once; final guards check its "
            "identity, size and modification time. Disk allocation was not measured.\n\n"
            "The parent stopped intentionally at max_new_chunks=1. Its seal records failed "
            "with PreparationPaused while incomplete.json records the requested limit. "
            "The individual chunk seal is ok. No complete 30-minute input was published; "
            "179 chunks remain outside this audit. Detector inference, biological identity, "
            "segmentation correctness and the network originals were not re-evaluated.\n"
        )
        (OUT / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
        )
        (OUT / "RESULT.md").write_text(report, encoding="utf-8")
        output_hashes = {}
        for path in (target, OUT / "summary.json", OUT / "RESULT.md"):
            output_hashes[str(path)] = sha(path)
            record.output(path.name, path, sha256=output_hashes[str(path)])
        evidence.final_guard()
        require(
            all(sha(path) == digest for path, digest in output_hashes.items()),
            "audit output changed",
        )
        record.param("all_non_raw_SHA_and_all_file_stat_guards_passed", True)
        record.param("raw_full_sha_count", 3)
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(OUT),
                "native_frames": 1660,
                "mask_frames": 4980,
                "native_raw_probes": 9,
                "background_probes": 27,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
