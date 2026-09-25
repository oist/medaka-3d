"""Restartable independent batch-one mask chunks under an explicit input protocol.

Every chunk and assembly is an immutable attempt directory. Reuse requires the
same complete protocol/code/runtime/input binding and verified payload hashes.
Only detector preprocessing can resume; no tracking, seed, SLEAP or truth is read.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parent
REPO = LAB.parents[1]
WORKSPACE = REPO
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(LAB))
import prepare_source_masks as single  # noqa: E402
from chunk_masks import concatenate_packed_masks  # noqa: E402
from prepare_sequence import EXPERIMENT, SHAPE, VIDEO_NAMES, VIEWS, sha256  # noqa: E402
from source_sequence import SOURCE_TOTAL_FRAMES, validate_source_manifest  # noqa: E402

from medaka3d.provenance import provenance  # noqa: E402

BACKGROUND_RULE = (
    "pixelwise median of the same nine fixed native frames before the full target interval, "
    "no fallback or image-dependent selection"
)
DETECTOR = dict(
    batch_size=1,
    score_threshold=0.5,
    device="cuda",
    parameter_dtype="float32",
    autocast=False,
    cuda_matmul_allow_tf32=False,
    cudnn_allow_tf32=True,
    cudnn_benchmark=False,
    retained_detectron2_models=True,
    all_detections_retained=True,
)
INDEPENDENCE = dict(
    sleap_input=False,
    seed_or_pose_input=False,
    identity_or_human_label_input=False,
    temporal_subsampling=False,
    resume_scope="independent per-image detector preprocessing only; no tracking is performed",
)
PROTOCOL_KEYS = {
    "schema",
    "created_utc",
    "experiment",
    "source_frame_range",
    "source_total_frames",
    "fps",
    "image_shape_hw",
    "views",
    "chunk_frames",
    "background_frames",
    "background_exclusion_frames",
    "background_rule",
    "detector",
    "source_cache_manifest_sha256",
    "asset_manifest_sha256",
    "source_videos_sha256",
    "model_assets_sha256",
    "independence",
    "limitations",
}
ASSET_KEYS = {"top_weights", "top_config", "side_weights", "side_config"}


def exact_contract(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            exact_contract(actual[k], v) for k, v in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            exact_contract(a, b) for a, b in zip(actual, expected, strict=True)
        )
    return actual == expected


def validate_protocol(value):
    if not isinstance(value, dict) or set(value) != PROTOCOL_KEYS:
        raise ValueError("protocol requires exactly the registered schema fields")
    fixed = dict(
        schema="medaka_independent_mask_protocol_v1",
        experiment=EXPERIMENT,
        source_total_frames=SOURCE_TOTAL_FRAMES,
        fps=166,
        image_shape_hw=list(SHAPE),
        views=list(VIEWS),
        background_exclusion_frames=332,
        background_rule=BACKGROUND_RULE,
        detector=DETECTOR,
        independence=INDEPENDENCE,
    )
    if any(not exact_contract(value[k], expected) for k, expected in fixed.items()):
        raise ValueError(
            "protocol changes native geometry/order, detector or independence contract"
        )
    interval = value["source_frame_range"]
    if (
        not isinstance(interval, list)
        or len(interval) != 2
        or any(type(v) is not int for v in interval)
        or not 0 <= interval[0] < interval[1] <= SOURCE_TOTAL_FRAMES
    ):
        raise ValueError("protocol source range must be a valid native half-open interval")
    chunk = value["chunk_frames"]
    if type(chunk) is not int or not 1 <= chunk <= 1660:
        raise ValueError("chunk_frames must be a positive native count no greater than 1660")
    background = value["background_frames"]
    if (
        not isinstance(background, list)
        or len(background) != 9
        or any(type(v) is not int for v in background)
        or background != sorted(set(background))
        or not all(0 <= v < interval[0] - value["background_exclusion_frames"] for v in background)
    ):
        raise ValueError(
            "exactly nine ordered native backgrounds must precede the full-window guard"
        )
    try:
        created = datetime.fromisoformat(value["created_utc"].replace("Z", "+00:00"))
        if created.utcoffset() is None or created.utcoffset().total_seconds() != 0:
            raise ValueError("timestamp is not UTC")
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("created_utc must be an explicit UTC timestamp") from exc
    if (
        not isinstance(value["limitations"], list)
        or not value["limitations"]
        or any(not isinstance(v, str) or not v for v in value["limitations"])
    ):
        raise ValueError("protocol limitations must be explicit nonempty strings")
    for key, names in (("source_videos_sha256", set(VIEWS)), ("model_assets_sha256", ASSET_KEYS)):
        if not isinstance(value[key], dict) or set(value[key]) != names:
            raise ValueError("protocol SHA maps require exact model/view keys")
    digests = [
        value["source_cache_manifest_sha256"],
        value["asset_manifest_sha256"],
        *value["source_videos_sha256"].values(),
        *value["model_assets_sha256"].values(),
    ]
    if any(not isinstance(v, str) or re.fullmatch("[0-9a-f]{64}", v) is None for v in digests):
        raise ValueError("protocol SHA256 values must be lowercase hexadecimal")
    return copy.deepcopy(value)


def chunk_ranges(protocol):
    start, stop = protocol["source_frame_range"]
    size = protocol["chunk_frames"]
    return [(first, min(first + size, stop)) for first in range(start, stop, size)]


def strict_runtime(protocol):
    runtime = single.load_runtime()
    import torch

    def flags():
        return dict(
            autocast=bool(torch.is_autocast_enabled()),
            cuda_matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
            cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
            cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
        )

    expected = {name: protocol["detector"][name] for name in flags()}
    if flags() != expected:
        raise ValueError("current CUDA/autocast backend flags differ from the registered protocol")
    runtime.metadata["detector_contract"] = copy.deepcopy(protocol["detector"])
    make = runtime.make_predictor

    def checked_predictor(kind, assets, out):
        if flags() != expected:
            raise ValueError("backend flags changed before predictor construction")
        predictor, model = make(kind, assets, out)
        if any(p.dtype != torch.float32 for p in predictor.model.parameters()):
            raise ValueError("retained detector parameters are not all float32")
        return predictor, model

    runtime.make_predictor = checked_predictor
    return runtime


@dataclass
class Context:
    protocol: dict
    protocol_path: Path
    source_cache: Path
    assets: dict
    sources: dict
    input_records: dict
    code_records: dict
    source_stats: dict
    runtime: object
    capture_factory: object
    process_view: object

    @property
    def binding(self):
        return dict(
            protocol_sha256=self.input_records["protocol"]["sha256"],
            input_sha256={k: v["sha256"] for k, v in self.input_records.items()},
            code_sha256={k: v["sha256"] for k, v in self.code_records.items()},
            runtime=self.runtime.metadata,
        )


def prepare_context(
    protocol_path, source_cache, *, capture_factory=None, runtime_factory=None, process_view=None
):
    capture_factory = capture_factory or single.cv2.VideoCapture
    runtime_factory = runtime_factory or strict_runtime
    protocol_path, source_cache = (
        Path(protocol_path).resolve(strict=True),
        Path(source_cache).resolve(strict=True),
    )
    if not source_cache.is_relative_to(WORKSPACE.resolve()):
        raise ValueError("only a verified local source cache inside the workspace is supported")
    protocol = validate_protocol(json.loads(protocol_path.read_text(encoding="utf-8")))
    cache_path = source_cache / "source_cache_manifest.json"
    cache_seal_path = source_cache / "source_cache.provenance.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    seal = json.loads(cache_seal_path.read_text(encoding="utf-8"))
    if (
        cache.get("status") != "verified"
        or seal.get("status") != "ok"
        or sha256(cache_path) != protocol["source_cache_manifest_sha256"]
        or sha256(cache_path) != seal["outputs"]["cache_manifest"]["sha256"]
    ):
        raise ValueError("verified source-cache manifest does not match protocol/seal")
    assets, asset_path = single.original.verify_assets(source_cache)
    if sha256(asset_path) != protocol["asset_manifest_sha256"]:
        raise ValueError("asset manifest differs from the registered protocol")
    inputs = dict(
        protocol=protocol_path,
        source_cache_manifest=cache_path,
        source_cache_provenance=cache_seal_path,
        asset_manifest=asset_path,
    )
    for key, entry in assets.items():
        if entry["sha256"] != protocol["model_assets_sha256"][key]:
            raise ValueError("model/config digest differs from protocol")
        inputs["asset:" + key] = Path(entry["path"])
    sources = {}
    for view in VIEWS:
        relative = f"medaka_202512/{EXPERIMENT}/{view}/{VIDEO_NAMES[view]}"
        path = single.contained_file(source_cache, relative)
        digest = sha256(path)
        entry = cache["files"][relative]
        if (
            digest != protocol["source_videos_sha256"][view]
            or digest != entry["local_full_sha256"]
            or digest != entry["source_stream_sha256"]
            or path.stat().st_size != entry["bytes"]
        ):
            raise ValueError("source video differs from protocol/cache")
        cap = capture_factory(str(path))
        try:
            single.checked_capture(cap, path)
        finally:
            cap.release()
        sources[view] = path
        inputs["video:" + view] = path
    input_records = {key: dict(path=str(path), sha256=sha256(path)) for key, path in inputs.items()}
    files = (
        Path(__file__),
        LAB / "prepare_source_masks.py",
        LAB / "prepare_sequence.py",
        LAB / "source_sequence.py",
        LAB / "observations.py",
        LAB / "chunk_masks.py",
        REPO / "src/medaka3d/provenance.py",
        REPO / "src/medaka3d/types.py",
    )
    code_records = {
        path.relative_to(REPO).as_posix(): dict(path=str(path), sha256=sha256(path))
        for path in files
    }
    runtime = runtime_factory(protocol)
    # TorchVersion is a str subclass; normalize exactly as the stored JSON before
    # comparing types on a later invocation, while keeping bool distinct from int.
    runtime.metadata = json.loads(json.dumps(runtime.metadata, allow_nan=False))
    if not exact_contract(runtime.metadata.get("detector_contract"), protocol["detector"]):
        raise ValueError("runtime does not attest the exact registered detector settings")
    return Context(
        protocol,
        protocol_path,
        source_cache,
        assets,
        sources,
        input_records,
        code_records,
        {view: single.source_stat(path) for view, path in sources.items()},
        runtime,
        capture_factory,
        process_view or single.process_view,
    )


def verify_frozen_inputs(context):
    for item in (*context.input_records.values(), *context.code_records.values()):
        if sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError("protocol, model, code or source bytes changed during preparation")
    if any(
        single.source_stat(path) != context.source_stats[view]
        for view, path in context.sources.items()
    ):
        raise ValueError("source-video metadata changed during preparation")


def new_directory(parent, prefix="attempt"):
    parent = Path(parent)
    if not parent.resolve().is_relative_to(WORKSPACE.resolve()):
        raise ValueError("attempt directory escaped the workspace")
    parent.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(re.escape(prefix) + r"_(\d{4,})$")
    numbers = [
        int(match[1]) for path in parent.iterdir() if (match := pattern.fullmatch(path.name))
    ]
    number = max(numbers, default=0) + 1
    while True:
        path = parent / f"{prefix}_{number:04d}"
        try:
            path.mkdir(exist_ok=False)
            return path
        except FileExistsError:
            number += 1


def register_context(context, out, record):
    for key, item in context.input_records.items():
        record.input(key, item["path"], sha256=item["sha256"])
    for relative, item in context.code_records.items():
        target = out / "code_snapshot" / relative
        single.copy_exclusive(item["path"], target)
        if sha256(target) != item["sha256"]:
            raise ValueError("code snapshot mismatch")
        record.output("code:" + relative, target, sha256=item["sha256"])
    single.copy_exclusive(context.protocol_path, out / "protocol.json")
    record.output("protocol", out / "protocol.json", sha256=context.binding["protocol_sha256"])
    record.param("binding", context.binding)


def raw_links(context, out, start, raw_hashes):
    (out / "raw").mkdir()
    raw = {}
    for view, path in context.sources.items():
        target = out / "raw" / VIDEO_NAMES[view]
        os.link(path, target)
        if not os.path.samefile(path, target):
            raise ValueError("source reference is not an original-video hardlink")
        raw[view] = dict(
            path=target.relative_to(out).as_posix(),
            bytes=path.stat().st_size,
            sha256=context.input_records["video:" + view]["sha256"],
            source_offset_frames=start,
            source_total_frames=SOURCE_TOTAL_FRAMES,
            storage_link="hardlink",
            frame_bgr_sha256=raw_hashes[view],
            source_cache_relative_path=path.relative_to(context.source_cache).as_posix(),
        )
    return raw


def manifest_base(context, start, stop, raw, masks, backgrounds, models):
    return dict(
        experiment=EXPERIMENT,
        frames=list(range(start, stop)),
        source_interval_half_open=[start, stop],
        fps=166,
        image_shape_hw=list(SHAPE),
        raw_storage="original_source_video",
        raw_codec="original source MP4; no re-encoding",
        raw=raw,
        masks=masks,
        backgrounds=backgrounds,
        models=models,
        runtime=context.runtime.metadata,
        binding=context.binding,
        protocol_sha256=context.binding["protocol_sha256"],
        background_frames=context.protocol["background_frames"],
        background_rule=BACKGROUND_RULE,
        background_scope="the same nine pre-full-window frames in every chunk",
        mask_schema="bbox_packbits_instances_v1",
        identity_fields="within-frame order only; source_track_id=-1",
        video_payload_bytes_copied=0,
        tracking_performed=False,
        human_truth_created=False,
        completion_rule=(
            "requires staging.provenance.json status ok and matching manifest/payload hashes"
        ),
    )


def seal_payload(context, out, manifest, record):
    validate_source_manifest(out, manifest)
    verify_frozen_inputs(context)
    for category, label in (("raw", "raw"), ("masks", "masks"), ("backgrounds", "background")):
        for view, info in manifest[category].items():
            path = single.contained_file(out, info["path"])
            if category == "raw" and (
                not os.path.samefile(path, context.sources[view])
                or info["sha256"] != context.input_records["video:" + view]["sha256"]
            ):
                raise ValueError("raw hardlink changed before sealing")
            if category != "raw" and sha256(path) != info["sha256"]:
                raise ValueError("output payload changed before sealing")
            record.output(f"{view}_{label}", path, sha256=info["sha256"])
    for kind, model in manifest["models"].items():
        effective = single.contained_file(out, model["effective_config"])
        if sha256(effective) != model["effective_config_sha256"]:
            raise ValueError("effective model config changed before sealing")
        record.output(kind + "_effective_config", effective, sha256=sha256(effective))
        source = out / f"{kind}_source_config.yaml"
        if sha256(source) != context.assets[kind + "_config"]["sha256"]:
            raise ValueError("source config copy differs from verified asset")
        record.output(kind + "_source_config", source, sha256=sha256(source))
    path = out / "manifest.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
    record.output("manifest", path, sha256=sha256(path))


def prepare_chunk(context, out, start, stop):
    if (start, stop) not in chunk_ranges(context.protocol):
        raise ValueError("chunk range does not match the registered plan")
    if any(out.iterdir()):
        raise FileExistsError("chunk attempt must be a new empty directory")
    with provenance("prepare_independent_mask_chunk", out, name="staging") as record:
        register_context(context, out, record)
        (out / "masks").mkdir()
        views, models = {}, {}
        for kind, names in (("top", ("top_cam",)), ("side", ("side1_cam", "side2_cam"))):
            predictor, models[kind] = context.runtime.make_predictor(kind, context.assets, out)
            try:
                for view in names:
                    views[view] = context.process_view(
                        context.sources[view],
                        view,
                        start,
                        stop,
                        context.protocol["background_frames"],
                        predictor,
                        context.runtime.synchronize,
                        out,
                        context.capture_factory,
                    )
            finally:
                del predictor
        raw = raw_links(context, out, start, {v: views[v]["raw_hashes"] for v in VIEWS})
        manifest = manifest_base(
            context,
            start,
            stop,
            raw,
            {v: views[v]["masks"] for v in VIEWS},
            {v: views[v]["background"] for v in VIEWS},
            models,
        )
        manifest["kind"] = "independent_detector_chunk"
        seal_payload(context, out, manifest, record)
    return manifest


def validate_attempt(
    context, out, start, stop, kind="independent_detector_chunk", expected_entries=None
):
    if out.is_symlink():
        raise ValueError("attempt directory cannot be a symlink")
    manifest = single.verify_completion(out)
    if (
        not exact_contract(manifest.get("binding"), context.binding)
        or manifest.get("kind") != kind
        or manifest["frames"] != list(range(start, stop))
        or manifest["source_interval_half_open"] != [start, stop]
        or manifest.get("background_frames") != context.protocol["background_frames"]
    ):
        raise ValueError("completed attempt binding/range/background does not match this run")
    seal = json.loads((out / "staging.provenance.json").read_text(encoding="utf-8"))
    if not exact_contract(seal["params"].get("binding"), context.binding):
        raise ValueError("attempt provenance binding differs from current inputs")
    if not exact_contract(manifest.get("runtime"), context.runtime.metadata):
        raise ValueError("attempt runtime metadata differs from its binding")
    for key, item in context.input_records.items():
        recorded = seal["inputs"].get(key, {})
        if recorded.get("sha256") != item["sha256"] or Path(recorded.get("path", "")) != Path(
            item["path"]
        ):
            raise ValueError("attempt input seals differ from the registered binding")
    required = {"manifest", "protocol", *("code:" + name for name in context.code_records)}
    required.update(f"{view}_{part}" for view in VIEWS for part in ("raw", "masks", "background"))
    required.update(
        f"{kind}_{part}_config" for kind in ("top", "side") for part in ("source", "effective")
    )
    if expected_entries is not None:
        required.add("chunk_index")
    if not required.issubset(seal["outputs"]):
        raise ValueError("completed attempt omits required code/model/input/output seals")
    for name, info in seal["outputs"].items():
        path = Path(info["path"]).resolve(strict=True)
        if not path.is_relative_to(out.resolve()):
            raise ValueError("sealed output path escaped its attempt")
        if name.endswith("_raw"):
            view = name.removesuffix("_raw")
            source = context.sources[view]
            if (
                not os.path.samefile(path, source)
                or info["sha256"] != context.input_records["video:" + view]["sha256"]
                or single.source_stat(source) != context.source_stats[view]
            ):
                raise ValueError("attempt original-video hardlink changed")
        elif sha256(path) != info["sha256"]:
            raise ValueError("completed attempt payload checksum changed")
    expected_snapshots = {"protocol": context.binding["protocol_sha256"]}
    expected_snapshots.update(
        {"code:" + key: item["sha256"] for key, item in context.code_records.items()}
    )
    if any(seal["outputs"][key]["sha256"] != digest for key, digest in expected_snapshots.items()):
        raise ValueError("attempt snapshot digests differ from the protocol/code binding")
    if set(manifest.get("models", {})) != {"top", "side"}:
        raise ValueError("attempt requires both registered detector models")
    for model_kind, model in manifest["models"].items():
        effective = single.contained_file(out, model["effective_config"])
        source = out / f"{model_kind}_source_config.yaml"
        if (
            sha256(effective) != model["effective_config_sha256"]
            or sha256(source) != context.assets[model_kind + "_config"]["sha256"]
            or not exact_contract(model.get("weights"), context.assets[model_kind + "_weights"])
            or not exact_contract(model.get("config"), context.assets[model_kind + "_config"])
        ):
            raise ValueError("attempt model/config metadata differs from sealed registered assets")
    for view in VIEWS:
        if len(manifest["raw"][view]["frame_bgr_sha256"]) != stop - start:
            raise ValueError("chunk native raw checksum count differs from its range")
        if manifest["backgrounds"][view]["frames"] != context.protocol["background_frames"]:
            raise ValueError("chunk substituted its own background sampling")
        with np.load(out / manifest["masks"][view]["path"], allow_pickle=False) as data:
            if (
                not np.array_equal(data["frame_idx"], np.arange(start, stop, dtype=np.int64))
                or not np.array_equal(data["image_shape_hw"], SHAPE)
                or not data["view_available"].all()
                or np.any(data["source_track_id"] != -1)
            ):
                raise ValueError("chunk mask frame/availability/independence schema mismatch")
    if expected_entries is not None:
        index_path = out / "chunks.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if (
            index.get("chunks") != expected_entries
            or index.get("protocol_sha256") != context.binding["protocol_sha256"]
            or manifest.get("chunk_index_sha256") != sha256(index_path)
            or index.get("schema") != "medaka_mask_chunk_index_v1"
            or not exact_contract(index.get("source_frame_range"), [start, stop])
            or not exact_contract(index.get("chunk_frames"), context.protocol["chunk_frames"])
            or Path(index.get("attempt_paths_relative_to", "")).resolve()
            != out.parent.parent.resolve()
        ):
            raise ValueError("existing assembly refers to different selected chunk attempts")
    return manifest


def select_attempt(
    context, parent, start, stop, kind="independent_detector_chunk", expected_entries=None
):
    rejected = []
    candidates = []
    if parent.exists():
        for path in parent.iterdir():
            match = re.fullmatch(r"attempt_(\d{4,})", path.name)
            if match and path.is_dir():
                candidates.append((int(match[1]), path))
    for _, path in sorted(candidates):
        try:
            manifest = validate_attempt(context, path, start, stop, kind, expected_entries)
            return path, manifest, rejected
        except (OSError, ValueError, KeyError, TypeError) as exc:
            rejected.append(dict(path=str(path), reason=f"{type(exc).__name__}: {exc}"))
    return None, None, rejected


def assemble(context, output_root, out, entries):
    plan = chunk_ranges(context.protocol)
    if [(entry["start"], entry["stop"]) for entry in entries] != plan:
        raise ValueError("final assembly requires every registered chunk exactly once in order")
    manifests, chunk_dirs = [], []
    for entry in entries:
        path = (output_root / entry["attempt_path"]).resolve(strict=True)
        if not path.is_relative_to(output_root) or path.is_symlink():
            raise ValueError("chunk index path escaped the preparation root")
        if (
            sha256(path / "manifest.json") != entry["manifest_sha256"]
            or sha256(path / "staging.provenance.json") != entry["provenance_sha256"]
        ):
            raise ValueError("chunk index checksum changed before assembly")
        manifests.append(validate_attempt(context, path, entry["start"], entry["stop"]))
        chunk_dirs.append(path)
    with provenance("assemble_independent_mask_chunks", out, name="staging") as record:
        register_context(context, out, record)
        chunk_inputs = []
        for number, (folder, manifest) in enumerate(zip(chunk_dirs, manifests, strict=True)):
            files = [folder / "manifest.json", folder / "staging.provenance.json"]
            files.extend(
                folder / manifest[category][view]["path"]
                for category in ("masks", "backgrounds")
                for view in VIEWS
            )
            for item in files:
                digest = sha256(item)
                chunk_inputs.append((item, digest))
                record.input(
                    f"chunk:{number}:{item.relative_to(folder).as_posix()}", item, sha256=digest
                )
        (out / "masks").mkdir()
        masks, backgrounds = {}, {}
        for view in VIEWS:
            arrays = concatenate_packed_masks(
                folder / manifest["masks"][view]["path"]
                for folder, manifest in zip(chunk_dirs, manifests, strict=True)
            )
            target = out / "masks" / f"{view}.npz"
            with target.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            masks[view] = dict(
                path=target.relative_to(out).as_posix(),
                sha256=sha256(target),
                bytes=target.stat().st_size,
                instances_per_frame=np.diff(arrays["frame_offsets"]).tolist(),
            )
            first = manifests[0]["backgrounds"][view]
            if any(m["backgrounds"][view]["sha256"] != first["sha256"] for m in manifests):
                raise ValueError("chunks disagree on the fixed background pixels/encoding")
            destination = out / f"background_{view}.png"
            single.copy_exclusive(chunk_dirs[0] / first["path"], destination)
            backgrounds[view] = dict(first, path=destination.name)
        start, stop = context.protocol["source_frame_range"]
        raw = raw_links(
            context,
            out,
            start,
            {
                view: [digest for m in manifests for digest in m["raw"][view]["frame_bgr_sha256"]]
                for view in VIEWS
            },
        )
        models = copy.deepcopy(manifests[0]["models"])
        for kind, model in models.items():
            for filename in (model["effective_config"], f"{kind}_source_config.yaml"):
                single.copy_exclusive(chunk_dirs[0] / filename, out / filename)
        index = dict(
            schema="medaka_mask_chunk_index_v1",
            protocol_sha256=context.binding["protocol_sha256"],
            source_frame_range=[start, stop],
            chunk_frames=context.protocol["chunk_frames"],
            chunks=entries,
            attempt_paths_relative_to=str(output_root),
        )
        path = out / "chunks.json"
        with path.open("x", encoding="utf-8") as stream:
            json.dump(index, stream, indent=2, allow_nan=False)
        record.output("chunk_index", path, sha256=sha256(path))
        manifest = manifest_base(context, start, stop, raw, masks, backgrounds, models)
        manifest.update(
            kind="assembled_independent_detector_chunks",
            chunks=len(entries),
            chunk_index_sha256=sha256(path),
            chunk_index="chunks.json",
        )
        if any(sha256(path) != digest for path, digest in chunk_inputs):
            raise ValueError("sealed chunk inputs changed during assembly")
        seal_payload(context, out, manifest, record)
    return manifest


class PreparationPaused(RuntimeError):
    """An explicit execution limit leaves sealed chunks but no global completion."""

    def __init__(self, summary):
        super().__init__("max_new_chunks reached; global input remains incomplete")
        self.summary = summary


def run(protocol_path, source_cache, out, *, max_new_chunks=None, **context_options):
    out = Path(out).resolve()
    if not out.is_relative_to(WORKSPACE.resolve()):
        raise ValueError("preparation output must remain inside the workspace")
    if max_new_chunks is not None and (type(max_new_chunks) is not int or max_new_chunks < 1):
        raise ValueError("max_new_chunks must be a positive integer execution limit")
    context = prepare_context(protocol_path, source_cache, **context_options)
    run_dir = new_directory(out / "runs", "run")
    entries, history, created = [], [], 0
    with provenance("restartable_independent_mask_preparation", run_dir, name="run") as record:
        record.param("binding", context.binding)
        for start, stop in chunk_ranges(context.protocol):
            parent = out / "chunks" / f"f{start:06d}-{stop:06d}"
            path, _manifest, rejected = select_attempt(context, parent, start, stop)
            reused = path is not None
            if path is None:
                if max_new_chunks is not None and created >= max_new_chunks:
                    incomplete = dict(
                        status="incomplete",
                        reason="requested_max_new_chunks",
                        final_sequence_path=None,
                        chunks=history,
                        protocol_sha256=context.binding["protocol_sha256"],
                        tracking_performed=False,
                    )
                    incomplete_path = run_dir / "incomplete.json"
                    with incomplete_path.open("x", encoding="utf-8") as stream:
                        json.dump(incomplete, stream, indent=2, allow_nan=False)
                    record.output("incomplete", incomplete_path, sha256=sha256(incomplete_path))
                    record.param("global_preparation_complete", False)
                    raise PreparationPaused(incomplete)
                path = new_directory(parent)
                prepare_chunk(context, path, start, stop)
                validate_attempt(context, path, start, stop)
                created += 1
            entry = dict(
                start=start,
                stop=stop,
                attempt_path=path.relative_to(out).as_posix(),
                manifest_sha256=sha256(path / "manifest.json"),
                provenance_sha256=sha256(path / "staging.provenance.json"),
            )
            entries.append(entry)
            history.append(dict(**entry, reused=reused, rejected_attempts=rejected))
            print(
                json.dumps(
                    {
                        "completed_chunks": len(entries),
                        "total_chunks": len(chunk_ranges(context.protocol)),
                        "start": start,
                        "stop": stop,
                        "reused": reused,
                    }
                ),
                flush=True,
            )
        start, stop = context.protocol["source_frame_range"]
        assembled, _manifest, rejected = select_attempt(
            context,
            out / "assemblies",
            start,
            stop,
            "assembled_independent_detector_chunks",
            entries,
        )
        if assembled is None:
            assembled = new_directory(out / "assemblies")
            assemble(context, out, assembled, entries)
            validate_attempt(
                context, assembled, start, stop, "assembled_independent_detector_chunks", entries
            )
        verify_frozen_inputs(context)
        result = dict(
            status="complete",
            tracking_performed=False,
            protocol_sha256=context.binding["protocol_sha256"],
            final_sequence_path=assembled.relative_to(out).as_posix(),
            final_manifest_sha256=sha256(assembled / "manifest.json"),
            final_provenance_sha256=sha256(assembled / "staging.provenance.json"),
            chunks=history,
            rejected_assemblies=rejected,
            completed_native_frames=stop - start,
            video_payload_bytes_copied=0,
        )
        path = run_dir / "summary.json"
        with path.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        record.output("summary", path, sha256=sha256(path))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--max-new-chunks",
        type=int,
        help="stop with exit 2 and incomplete status after this many new chunks",
    )
    args = parser.parse_args(argv)
    try:
        run(args.protocol, args.source_cache, args.out, max_new_chunks=args.max_new_chunks)
    except PreparationPaused as exc:
        print(json.dumps(exc.summary), flush=True)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
