"""Fresh batch-one masks on local original-video hardlinks; no FFV1 copies.

Use the existing Detectron2 environment with -B. A completed manifest requires
staging.provenance.json status=ok and the matching manifest checksum. Explicit
windows retain the original fixed-nine background protocol; fewer than three
remaining samples fails before output creation or runtime/GPU initialization.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

LAB = Path(__file__).resolve().parent
REPO = LAB.parents[1]
WORKSPACE = REPO
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(LAB))
import prepare_sequence as original  # noqa: E402
from source_sequence import SOURCE_TOTAL_FRAMES, validate_source_manifest  # noqa: E402

from medaka3d.provenance import provenance  # noqa: E402


@dataclass
class InferenceRuntime:
    metadata: dict
    make_predictor: Callable
    synchronize: Callable


def load_runtime():
    """Same DefaultPredictor configuration and precision defaults as frozen staging."""
    import detectron2
    import torch
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    if not torch.cuda.is_available():
        raise RuntimeError("fresh mask preparation requires the existing CUDA Detectron2 runtime")

    def make_predictor(kind, assets, out):
        config_path = Path(assets[f"{kind}_config"]["path"])
        cfg = get_cfg()
        cfg.merge_from_file(str(config_path))
        cfg.MODEL.WEIGHTS = assets[f"{kind}_weights"]["path"]
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.DEVICE = "cuda"
        cfg.freeze()
        effective = out / f"{kind}_effective_config.yaml"
        with effective.open("x", encoding="utf-8") as stream:
            stream.write(cfg.dump())
        copy_exclusive(config_path, out / f"{kind}_source_config.yaml")
        predictor = DefaultPredictor(cfg)
        model = dict(
            weights=assets[f"{kind}_weights"],
            config=assets[f"{kind}_config"],
            effective_config=effective.name,
            effective_config_sha256=original.sha256(effective),
        )
        return predictor, model

    metadata = dict(
        python=sys.version,
        executable=sys.executable,
        detectron2=detectron2.__version__,
        detectron2_distribution=importlib.metadata.version("detectron2"),
        torch=torch.__version__,
        torchvision=importlib.metadata.version("torchvision"),
        opencv=cv2.__version__,
        numpy=np.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
        batch_size=1,
        precision="unchanged DefaultPredictor FP32 defaults; no autocast or backend changes",
    )
    return InferenceRuntime(metadata, make_predictor, torch.cuda.synchronize)


def copy_exclusive(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    with Path(source).open("rb") as reader, target.open("xb") as writer:
        shutil.copyfileobj(reader, writer, 4 * 1024 * 1024)


def contained_file(root, relative):
    text = str(relative).replace("\\", "/")
    value = PurePosixPath(text)
    if value.is_absolute() or ".." in value.parts or ":" in text or not value.parts:
        raise ValueError("source-cache paths must be relative and contained")
    path = (root / value).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("source-cache path escaped its root")
    return path


def source_stat(path):
    value = path.stat()
    return dict(bytes=value.st_size, mtime_ns=value.st_mtime_ns)


def validate_window(start, stop):
    if (
        type(start) is not int
        or type(stop) is not int
        or not 0 <= start < stop <= SOURCE_TOTAL_FRAMES
    ):
        raise ValueError("start/stop must define a nonempty native window within [0, 348600)")
    try:
        return original.select_background_frames(start, stop)
    except ValueError as exc:
        raise ValueError(
            "fixed-nine background protocol retains fewer than three samples after "
            "[start-332, stop+332) exclusion; no fallback sampling is permitted. "
            "A separately registered background protocol is required for this window."
        ) from exc


def checked_capture(capture, path):
    details = original.validate_capture(capture, path)
    if (
        capture.get(cv2.CAP_PROP_FRAME_COUNT) != SOURCE_TOTAL_FRAMES
        or not np.isclose(capture.get(cv2.CAP_PROP_FPS), original.FPS, rtol=0, atol=1e-6)
        or (capture.get(cv2.CAP_PROP_FRAME_HEIGHT), capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        != original.SHAPE
    ):
        raise ValueError("source metadata must be exactly 348600 frames, 166 fps, 1200 square")
    return details


def read_native(capture, frame, *, seek=False):
    image = original.read_frame(capture, frame, seek=seek)
    if not np.isclose(capture.get(cv2.CAP_PROP_POS_FRAMES), frame + 1, rtol=0, atol=1e-6):
        raise ValueError("source frame position is not an exact native index")
    return image


def preflight(source_cache, out, start, stop, capture_factory):
    samples = validate_window(start, stop)
    if out.exists() or out.is_symlink():
        raise FileExistsError("fresh mask output exists; no overwrite or resume")
    if not out.resolve().is_relative_to(WORKSPACE.resolve()):
        raise ValueError("new hardlinks must remain inside the workspace")
    if not source_cache.is_relative_to(WORKSPACE.resolve()):
        raise ValueError("source cache must be local and inside the workspace")
    manifest_path = source_cache / "source_cache_manifest.json"
    provenance_path = source_cache / "source_cache.provenance.json"
    cached = json.loads(manifest_path.read_text(encoding="utf-8"))
    seal = json.loads(provenance_path.read_text(encoding="utf-8"))
    if cached.get("status") != "verified" or seal.get("status") != "ok":
        raise ValueError("local source cache is not sealed and verified")
    if original.sha256(manifest_path) != seal["outputs"]["cache_manifest"]["sha256"]:
        raise ValueError("local source-cache manifest seal does not match")
    assets, asset_manifest = original.verify_assets(source_cache)
    hashes = {
        str(path): original.sha256(path)
        for path in (manifest_path, provenance_path, asset_manifest)
    }
    hashes.update({entry["path"]: entry["sha256"] for entry in assets.values()})
    sources, statistics = {}, {}
    for view in original.VIEWS:
        relative = f"medaka_202512/{original.EXPERIMENT}/{view}/{original.VIDEO_NAMES[view]}"
        source = contained_file(source_cache, relative)
        entry = cached["files"][relative]
        digest = original.sha256(source)
        if (
            digest != entry["local_full_sha256"]
            or digest != entry["source_stream_sha256"]
            or source.stat().st_size != entry["bytes"]
        ):
            raise ValueError(f"local original video differs from the verified cache: {view}")
        capture = capture_factory(str(source))
        try:
            checked_capture(capture, source)
        finally:
            capture.release()
        hashes[str(source)] = digest
        sources[view] = source
        statistics[view] = source_stat(source)
    return samples, assets, sources, hashes, statistics, asset_manifest


def process_view(source, view, start, stop, samples, predictor, synchronize, out, capture_factory):
    """Only current image and current full masks are retained; packed masks accumulate."""
    capture = capture_factory(str(source))
    try:
        details = checked_capture(capture, source)
        background = np.median(
            np.stack([read_native(capture, frame, seek=True) for frame in samples]), axis=0
        ).astype(np.uint8)
        background_path = out / f"background_{view}.png"
        if not cv2.imwrite(str(background_path), background):
            raise RuntimeError("background PNG write failed")
        decoded_background = cv2.imread(str(background_path))
        if decoded_background is None or not np.array_equal(background, decoded_background):
            raise ValueError("background PNG did not preserve exact BGR pixels")
        packed, raw_hashes, timings = original.PackedMasks(), [], []
        for frame in range(start, stop):
            image = read_native(capture, frame, seek=frame == start)
            raw_hashes.append(original.array_sha256(image))
            tic = time.perf_counter()
            pred = predictor(image)["instances"].to("cpu")
            synchronize()
            timings.append(time.perf_counter() - tic)
            packed.append(
                frame,
                pred.pred_masks.numpy(),
                pred.scores.numpy(),
                pred.pred_boxes.tensor.numpy(),
                pred.pred_classes.numpy(),
            )
            if (frame - start + 1) % 50 == 0 or frame == stop - 1:
                print(
                    json.dumps(
                        {"view": view, "completed": frame - start + 1, "frames": stop - start}
                    ),
                    flush=True,
                )
    finally:
        capture.release()
    arrays = packed.arrays()
    if not np.array_equal(arrays["frame_idx"], np.arange(start, stop, dtype=np.int64)):
        raise ValueError("packed masks do not cover every declared native frame")
    mask_path = out / "masks" / f"{view}.npz"
    with mask_path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with np.load(mask_path, allow_pickle=False) as archive:
        saved = {key: archive[key] for key in archive.files}
    for index, expected in enumerate(packed.mask_hashes):
        if original.array_sha256(original.decode_frame_masks(saved, index)) != expected:
            raise ValueError("saved packed-mask roundtrip checksum mismatch")
    return dict(
        source=details,
        raw_hashes=raw_hashes,
        masks=dict(
            path=mask_path.relative_to(out).as_posix(),
            sha256=original.sha256(mask_path),
            bytes=mask_path.stat().st_size,
            instances_per_frame=np.diff(saved["frame_offsets"]).tolist(),
            roundtrip_verified_frames=len(raw_hashes),
            inference_s_by_frame=timings,
        ),
        background=dict(
            path=background_path.name, sha256=original.sha256(background_path), frames=samples
        ),
    )


def verify_completion(out):
    """Check the completed seal; SourceSequenceReader separately checks payload bytes."""
    out = Path(out)
    seal = json.loads((out / "staging.provenance.json").read_text(encoding="utf-8"))
    if seal.get("status") != "ok":
        raise ValueError("fresh mask staging did not complete successfully")
    path = out / "manifest.json"
    if original.sha256(path) != seal["outputs"]["manifest"]["sha256"]:
        raise ValueError("fresh mask manifest does not match its completed provenance")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_source_manifest(out, manifest)
    for view in original.VIEWS:
        for category, name in (("raw", "raw"), ("masks", "masks"), ("backgrounds", "background")):
            info = manifest[category][view]
            if info["sha256"] != seal["outputs"][f"{view}_{name}"]["sha256"]:
                raise ValueError("view payload is not bound into the completed staging seal")
    return manifest


def run(source_cache, out, start, stop, *, capture_factory=None, runtime_factory=None):
    capture_factory = capture_factory or cv2.VideoCapture
    runtime_factory = runtime_factory or load_runtime
    source_cache, out = Path(source_cache).resolve(strict=True), Path(out).resolve()
    samples, assets, sources, input_hashes, source_stats, asset_manifest = preflight(
        source_cache, out, start, stop, capture_factory
    )
    code_paths = (
        Path(__file__),
        LAB / "prepare_sequence.py",
        LAB / "source_sequence.py",
        LAB / "observations.py",
        REPO / "src/medaka3d/provenance.py",
        REPO / "src/medaka3d/types.py",
    )
    code = {str(path): original.sha256(path) for path in code_paths}
    out.mkdir(parents=True, exist_ok=False)
    with provenance("prepare_three_body_source_masks", out, name="staging") as record:
        tic = time.perf_counter()
        for name in ("raw", "masks", "code_snapshot"):
            (out / name).mkdir()
        for path, digest in code.items():
            copy_exclusive(path, out / "code_snapshot" / Path(path).relative_to(REPO))
            record.input("code:" + path, path, sha256=digest)
        for path, digest in input_hashes.items():
            record.input(path, path, sha256=digest)
        record.param("source_interval_half_open", [start, stop])
        record.param("batch_size", 1)
        record.param("threshold", 0.5)
        record.param("background_samples", samples)
        record.param("video_payload_bytes_copied", 0)
        links = {}
        for view, source in sources.items():
            target = out / "raw" / original.VIDEO_NAMES[view]
            os.link(source, target)
            if not os.path.samefile(source, target):
                raise ValueError("original-video reference is not a hardlink")
            links[view] = dict(
                path=target.relative_to(out).as_posix(),
                bytes=source.stat().st_size,
                sha256=input_hashes[str(source)],
                source_offset_frames=start,
                source_total_frames=SOURCE_TOTAL_FRAMES,
                storage_link="hardlink",
                source_cache_relative_path=source.relative_to(source_cache).as_posix(),
                inode=target.stat().st_ino,
                hardlink_count_at_creation=target.stat().st_nlink,
            )
        runtime = runtime_factory()
        models, views = {}, {}
        for kind, names in (("top", ("top_cam",)), ("side", ("side1_cam", "side2_cam"))):
            predictor, models[kind] = runtime.make_predictor(kind, assets, out)
            try:
                for view in names:
                    views[view] = process_view(
                        sources[view],
                        view,
                        start,
                        stop,
                        samples,
                        predictor,
                        runtime.synchronize,
                        out,
                        capture_factory,
                    )
                    links[view]["frame_bgr_sha256"] = views[view].pop("raw_hashes")
                    views[view]["raw"] = links[view]
                    for part in ("raw", "masks", "background"):
                        item = views[view][part]
                        record.output(view + "_" + part, out / item["path"], sha256=item["sha256"])
            finally:
                del predictor
        if set(views) != set(original.VIEWS):
            raise ValueError("all three camera streams must finish before publication")
        if any(original.sha256(path) != value for path, value in input_hashes.items()):
            raise ValueError(
                "source video, checkpoint/config or input metadata changed during inference"
            )
        if any(source_stat(path) != source_stats[view] for view, path in sources.items()):
            raise ValueError("source video metadata changed during inference")
        if any(original.sha256(path) != value for path, value in code.items()):
            raise ValueError("staging code changed during inference")
        for view, data in views.items():
            for part in ("masks", "background"):
                info = data[part]
                if original.sha256(out / info["path"]) != info["sha256"]:
                    raise ValueError(f"saved payload changed before sealing {view}/{part}")
        for kind, model in models.items():
            effective = contained_file(out, model["effective_config"])
            if original.sha256(effective) != model["effective_config_sha256"]:
                raise ValueError("effective predictor config changed before sealing")
            record.output(kind + "_effective_config", effective, sha256=original.sha256(effective))
        manifest = dict(
            experiment=original.EXPERIMENT,
            window="explicit_native_interval",
            frames=list(range(start, stop)),
            source_interval_half_open=[start, stop],
            fps=original.FPS,
            image_shape_hw=list(original.SHAPE),
            raw_storage="original_source_video",
            raw_codec="original source MP4; no re-encoding",
            raw={v: views[v]["raw"] for v in original.VIEWS},
            masks={v: views[v]["masks"] for v in original.VIEWS},
            backgrounds={v: views[v]["background"] for v in original.VIEWS},
            source_videos={v: views[v]["source"] for v in original.VIEWS},
            mask_schema="bbox_packbits_instances_v1",
            mask_encoding="Crop flattened C order; packbits little endian; K+1 byte offsets",
            mask_boxes="K,4 int32 [x0,y0,x1,y1], exclusive upper bounds; empty mask [0,0,0,0]",
            view_available="true for every processed frame, including zero detections",
            identity_fields="instance_id: within-frame order; source_track_id: always -1",
            background_candidates=list(original.BACKGROUND_CANDIDATES),
            background_rule=(
                "offline median excluding [start-332, stop+332), minimum 3 samples; no fallback"
            ),
            models=models,
            runtime=runtime.metadata,
            code_sha256=code,
            reused_frozen_code=dict(
                path="labs/20260909_three_body/prepare_sequence.py",
                sha256=code[str(LAB / "prepare_sequence.py")],
                functions=[
                    "verify_assets",
                    "select_background_frames",
                    "validate_capture",
                    "read_frame",
                    "PackedMasks",
                    "decode_frame_masks",
                    "array_sha256",
                    "sha256",
                ],
                predictor_configuration=(
                    "same get_cfg/merge, retained weights override, score .5, CUDA, "
                    "freeze, batch-one DefaultPredictor; requires real parity QA"
                ),
            ),
            asset_manifest_sha256=original.sha256(asset_manifest),
            source_cache_manifest_sha256=input_hashes[
                str(source_cache / "source_cache_manifest.json")
            ],
            input_sha256_before=input_hashes,
            input_sha256_after=input_hashes,
            code_and_inputs_unchanged=True,
            video_payload_bytes_copied=0,
            staging_elapsed_s=time.perf_counter() - tic,
            elapsed_excludes="preflight input/model/video hashes and native metadata checks",
            completion_rule=(
                "requires staging.provenance.json status ok and matching manifest SHA256"
            ),
            limitations=[
                "No SLEAP, initial seed, 3D trajectory, identity or ground-truth input is read.",
                "All detections at fixed score threshold .5 are retained; no count filter.",
                "No temporal subsampling, model retraining or backend precision changes.",
                "Cross-camera timestamp synchronization is inherited from the registered videos.",
                "Parity with frozen staging still needs a real small-window comparison.",
                "Large windows can fail the background rule; no alternative is inferred.",
                "Hardlinks share source bytes and must be opened read-only.",
            ],
        )
        validate_source_manifest(out, manifest)
        path = out / "manifest.json"
        with path.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, allow_nan=False)
        record.output("manifest", path, sha256=original.sha256(path))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.source_cache, args.out, args.start, args.stop)


if __name__ == "__main__":
    main()
