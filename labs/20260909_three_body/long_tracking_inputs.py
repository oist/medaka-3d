"""Verify a prefix of sealed detector chunks for continuous long-window tracking.

No detector, GPU, baseline coordinate array or human annotation is opened here.
The detector runtime below is its frozen *recorded* metadata, not a runtime loaded
into this process. Native DLLs were not hashed by the detector producer. Each
SourceSequenceReader still checks decoded native pixels when the driver consumes
its selected chunk. Successful chunks do not certify an ID or a completed track.
"""

from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import prepare_chunked_source_masks as staging
from chunk_masks import validate_packed_masks
from node_geometry import RayObservations, supported_nodes
from observations import CachedNodes
from run_experiment import (
    CALIBRATION_DIR,
    EXPERIMENT,
    resolve_input,
    sha256,
    skeleton_from_sources,
    staging_catalog,
    verified_input,
)

from medaka3d.calib import load_rig
from medaka3d.types import VIEWS

LAB = Path(__file__).resolve().parent
REPO = LAB.parents[1]
WORKSPACE = REPO
INPUT_ROOT = REPO / "work/generative_tracking_inputs"
SOURCE_CACHE = REPO / "work/three_body/source_cache"
PROTOCOL_PATH = LAB / "long_input_protocol.json"
PROTOCOL_SHA256 = "78e9b832393acbe0297a6fc9122cbbed24d07e779f23542919a97e719cc19330"
SEED_SHA256 = "34da64c0c61a4c4e589b3e38a01627baff0e934b1d1a4325a280645e8a132b40"
FULL_RANGE = (49800, 348600)
STAGER_CODE = {
    "labs/20260909_three_body/prepare_chunked_source_masks.py":
        "a081c47e6ed67b4b61c02635e00c97c967803b3528fe107f016ab23b5ae34f16",
    "labs/20260909_three_body/prepare_source_masks.py":
        "b6e78bc70ef4191fe57dc4a5ec391a42f2060c51aae62e2e715412f3ae344262",
    "labs/20260909_three_body/prepare_sequence.py":
        "959e862cc58c5840a00a763d8a63b635de134492a637ac2f1e8a7131f25178b8",
    "labs/20260909_three_body/source_sequence.py":
        "9db0e992768d54369d1431f82280c1b1b22d324d7a66daa35aa94dc4296a5cfe",
    "labs/20260909_three_body/observations.py":
        "f1efcd27b31e63ce693229c9e765152acfc7984b3b8839f1792bcc2efd9cb5df",
    "labs/20260909_three_body/chunk_masks.py":
        "9e08248ea593b4ccb105a6ff351fd85c6db24aa6754413b2df766404cdfe4b81",
    "src/medaka3d/provenance.py":
        "1e2c260c959be0f240f244d064ec0f2b4c7f82cd7ecf0d9f29a82bdb1b2150c1",
    "src/medaka3d/types.py":
        "71b002e8dbfe7efb139520bc387c9cc0aa097ce55efb24aadd1a6080da71e33a",
}
RECORDED_RUNTIME = {
    "python": "3.11.15 (main, Mar 10 2026, 18:12:25) [MSC v.1944 64 bit (AMD64)]",
    "executable": str(REPO.parent / "medaka-mating-pipeline/.venv-detectron2/Scripts/python.exe"),
    "detectron2": "0.6", "detectron2_distribution": "0.6+fd27788pt2.10.0cu130",
    "torch": "2.10.0+cu130", "torchvision": "0.25.0+cu130", "opencv": "4.13.0",
    "numpy": "2.4.4", "cuda": "13.0", "gpu": "NVIDIA GeForce RTX 5070 Ti", "batch_size": 1,
    "precision": "unchanged DefaultPredictor FP32 defaults; no autocast or backend changes",
    "detector_contract": copy.deepcopy(staging.DETECTOR),
}


@dataclass(frozen=True)
class LongInputs:
    initial_pose_xyz: np.ndarray
    rig: object
    protocol: dict
    full_frame_range: tuple[int, int]
    frame_idx: np.ndarray
    chunk_roots: tuple[Path, ...]
    input_root: Path
    fixed_bindings: dict
    verified_hashes: dict[str, str]
    chunk_records: list[dict]
    full_interval_covered: bool
    global_assembly_verified: bool = False


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _root(path):
    path = Path(path)
    require(not path.is_symlink(), "input root cannot be a symbolic link")
    path = path.resolve(strict=True)
    require(path.is_dir() and path.is_relative_to(WORKSPACE.resolve()), "foreign input root")
    return path


def _record_file(info, expected_path, hashes):
    path = Path(info.get("path", "")).resolve(strict=True)
    require(path == expected_path.resolve(strict=True), "sealed output has a foreign path")
    digest = sha256(path)
    require(digest == info.get("sha256"), f"sealed output SHA differs: {path}")
    hashes[str(path)] = digest
    return digest


def _verify_seed(seed_root, input_root):
    """Use sealed first-frame outputs and present SL/calibration; never reopen C34."""
    registration_path = seed_root / "registration.json"
    require(sha256(registration_path) == SEED_SHA256, "wrong frozen seed registration SHA")
    registration = read_json(registration_path)
    seal_path = seed_root / "registration.provenance.json"
    seal = read_json(seal_path)
    require(
        seal.get("status") == "ok" and seal.get("stage") == "three_body_long_seed_registration",
        "seed registration did not complete",
    )
    require(
        registration["schema"] == "three_body_first_frame_seed_registration_v1"
        and registration["experiment"] == EXPERIMENT and registration["frame"] == FULL_RANGE[0]
        and registration["source_frame_range"] == list(FULL_RANGE)
        and registration["views"] == list(VIEWS)
        and registration["nodes"] == ["mouth", "pectoral", "tail"]
        and registration["fps"] == 166,
        "seed registration has a different native experiment",
    )
    hashes = {str(seal_path): sha256(seal_path)}
    expected_roles = {
        "registration.json", "initial_pose_xyz.npy", "seed.npz",
        "input_snapshot/prior_audit.json", "input_snapshot/audit_long_start.py",
        "input_snapshot/staging_manifest.json",
    }
    expected_roles.update("code_snapshot/" + key for key in registration["code_hashes"])
    expected_roles.update(
        "input_snapshot/calibration/" + name for name in (
            "calibration_manifest.json", "cam0_bundle_refractive.json",
            "cam1_bundle_refractive.json", "cam2_bundle_refractive.json",
        )
    )
    require(set(seal["outputs"]) == expected_roles, "seed seal lacks required bound snapshots")
    for role, info in seal["outputs"].items():
        target = resolve_input(seed_root, role)
        _record_file(info, target, hashes)
    for relative, expected in registration["code_hashes"].items():
        path = resolve_input(REPO, relative)
        require(
            sha256(path) == expected
            == hashes[str(resolve_input(seed_root, "code_snapshot/" + relative))],
            "seed source or its snapshot changed",
        )
        hashes[str(path)] = expected
    for key in ("initial_pose", "seed"):
        entry = registration[key]
        path = resolve_input(seed_root, entry["path"])
        require(hashes[str(path)] == entry["sha256"], "seed data has inconsistent output binding")
    catalog, catalog_path = staging_catalog(input_root)
    require(
        sha256(catalog_path) == registration["input_hashes"].get(str(catalog_path)),
        "foreign input catalog differs from seed registration",
    )
    hashes[str(catalog_path)] = sha256(catalog_path)
    current = {}
    for relative in catalog:
        if relative.startswith(CALIBRATION_DIR + "/") or relative.startswith(
            f"work/c01_s16_predictions/bottomup/{EXPERIMENT}__"
        ):
            path = verified_input(input_root, relative, catalog)
            digest = sha256(path)
            require(
                digest == registration["input_hashes"].get(str(path)),
                "current SL/calibration differs from the registered initial evidence",
            )
            current[str(path)] = digest
            if relative.startswith(CALIBRATION_DIR + "/"):
                snapshot = resolve_input(seed_root, "input_snapshot/calibration/" + path.name)
                require(hashes[str(snapshot)] == digest, "seed calibration snapshot mismatch")
    require(len(current) == 7, "seed requires exactly three SL caches and four calibration files")
    hashes.update(current)
    with np.load(seed_root / "seed.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    required = {
        "frame_idx", "pose_xyz", "source_rows", "sources", "observed", "projected_uv",
        "selected_points_uv", "ray_distance_mm", "reprojection_error_px", "direct_view",
        "segment_lengths_mm", *("sleap_points_" + view for view in VIEWS),
    }
    require(set(arrays) == required, "seed NPZ has unregistered arrays")
    pose = np.load(seed_root / "initial_pose_xyz.npy", allow_pickle=False)
    require(
        pose.shape == (3, 3, 3) and pose.dtype == np.float64 and np.isfinite(pose).all()
        and np.array_equal(pose, arrays["pose_xyz"])
        and np.array_equal(pose, registration["pose_xyz"]),
        "registered initial nine points changed",
    )
    require(
        np.all(pose >= [0, 0, -6]) and np.all(pose <= [192, 192, 186])
        and np.all(np.linalg.norm(np.diff(pose, axis=1), axis=-1) >= 0.1),
        "seed fails unchanged tracker bounds or body length checks",
    )
    frames = np.array([FULL_RANGE[0]], np.int64)
    require(np.array_equal(arrays["frame_idx"], frames), "seed contains a different frame")
    nodes = CachedNodes(input_root, EXPERIMENT, frames)
    rig = load_rig(input_root / CALIBRATION_DIR)
    require(tuple(rig.views) == tuple(VIEWS), "seed rig order changed")
    source = arrays["source_rows"]
    require(source.shape == (3, 3) and source.dtype.kind in "iu", "invalid seed source rows")
    points = {view: nodes.at(view, FULL_RANGE[0]) for view in VIEWS}
    for v, view in enumerate(VIEWS):
        require(
            np.all(source[:, v] >= 0) and np.all(source[:, v] < len(points[view]))
            and len(np.unique(source[:, v])) == 3,
            "seed SL source is missing or nonexclusive",
        )
        require(
            np.isfinite(points[view][source[:, v]]).all()
            and np.array_equal(points[view], arrays["sleap_points_" + view], equal_nan=True),
            "seed first-frame SL pixels changed",
        )
    reconstructed, triangulated = skeleton_from_sources(rig, nodes, frames, source[None])
    require(np.array_equal(reconstructed[0], pose) and triangulated.all(), "seed geometry changed")
    sources, observed, _ = supported_nodes(
        pose, np.repeat(source[..., None], 3, axis=-1),
        RayObservations.from_points(rig, points), rig, maximum_ray_mm=2, maximum_pixel_error=12,
    )
    require(
        observed.all() and np.array_equal(observed, arrays["observed"])
        and np.array_equal(sources, arrays["sources"]),
        "seed first-frame direct support no longer matches",
    )
    binding = {
        "registration_path": str(registration_path), "registration_sha256": SEED_SHA256,
        "registration_seal_sha256": hashes[str(seal_path)],
        "initial_pose_sha256": registration["initial_pose"]["sha256"],
        "seed_npz_sha256": registration["seed"]["sha256"],
        "input_root": str(input_root), "catalog_sha256": hashes[str(catalog_path)],
        "SL_and_calibration_sha256": current,
        "optimistic_initial_correspondence": True,
        "baseline_or_GT_arrays_opened_by_loader": False,
    }
    return pose.copy(), rig, binding, hashes


def _make_context():
    # This object only supplies previously registered metadata to pure validators.
    # It has no predictor factory or CUDA synchronization method.
    return staging.prepare_context(
        PROTOCOL_PATH, SOURCE_CACHE,
        runtime_factory=lambda _: SimpleNamespace(metadata=copy.deepcopy(RECORDED_RUNTIME)),
    )


def _select_chunk(context, stage_root, start, stop):
    parent = stage_root / "chunks" / f"f{start:06d}-{stop:06d}"
    candidates = []
    if parent.is_dir():
        for path in parent.iterdir():
            match = re.fullmatch(r"attempt_(\d{4,})", path.name)
            if match and path.is_dir():
                candidates.append((int(match[1]), path))
    reasons = []
    for _, path in sorted(candidates):
        try:
            require(not path.is_symlink(), "chunk attempt cannot be a symbolic link")
            path = path.resolve(strict=True)
            require(path.is_relative_to(stage_root), "foreign chunk attempt")
            seal = read_json(path / "staging.provenance.json")
            require(
                seal.get("status") == "ok"
                and seal.get("stage") == "prepare_independent_mask_chunk",
                "chunk has no successful independent detector seal",
            )
            require(
                set(seal["inputs"]) == set(context.input_records), "foreign detector input role"
            )
            manifest = staging.validate_attempt(context, path, start, stop)
            require(
                manifest.get("experiment") == EXPERIMENT
                and manifest.get("fps") == 166
                and manifest.get("image_shape_hw") == [1200, 1200]
                and manifest.get("protocol_sha256") == context.binding["protocol_sha256"]
                and manifest.get("tracking_performed") is False
                and manifest.get("human_truth_created") is False,
                "chunk native protocol or independence metadata changed",
            )
            for view in VIEWS:
                require(
                    all(isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value)
                        for value in manifest["raw"][view]["frame_bgr_sha256"]),
                    "chunk contains malformed native pixel hashes",
                )
                with np.load(path / manifest["masks"][view]["path"], allow_pickle=False) as arrays:
                    require(len(arrays.files) == len(set(arrays.files)), "duplicate mask arrays")
                    packed = {key: arrays[key] for key in arrays.files}
                validate_packed_masks(packed)
            return path, manifest, seal
        except (OSError, ValueError, KeyError, TypeError) as exc:
            reasons.append(f"{path.name}: {type(exc).__name__}: {exc}")
    raise ValueError(f"missing completed chunk [{start},{stop}); rejected={reasons}")


def verify_prefix_records(previous, current):
    """A checkpoint's consumed chunk evidence must remain exactly the same prefix."""
    require(isinstance(previous, list) and isinstance(current, list), "chunk records must be lists")
    require(len(previous) <= len(current), "previous consumed prefix became shorter")
    next_frame = FULL_RANGE[0]
    for index, record in enumerate(current):
        require(
            type(record.get("index")) is int and record["index"] == index
            and type(record.get("start")) is int
            and type(record.get("stop")) is int and record["start"] == next_frame
            and record["stop"] > record["start"],
            "chunk records are reordered or discontinuous",
        )
        next_frame = record["stop"]
    require(
        staging.exact_contract(previous, current[:len(previous)]),
        "previous consumed chunk evidence changed",
    )


def load_long_inputs(seed_root, input_stage_root, input_root=INPUT_ROOT, *, through_frame):
    """Return a verified [49800, through_frame) prefix; do not initialize a tracker.

    Select the lowest-numbered valid successful attempt for each registered chunk.
    Failed/changed attempts are never overwritten. Root drivers must call
    verify_prefix_records when restoring previous state, and keep their own tracker
    code/config binding. Future unconsumed chunks are not opened. ``verified_hashes``
    hashes original raw files once via canonical source-cache paths; hardlink
    identities and full per-chunk evidence remain in ``chunk_records``.
    """
    require(
        not isinstance(through_frame, (bool, np.bool_))
        and isinstance(through_frame, (int, np.integer)),
        "through_frame must be a native integer exclusive endpoint",
    )
    require(sha256(PROTOCOL_PATH) == PROTOCOL_SHA256, "wrong frozen long protocol SHA")
    protocol = staging.validate_protocol(read_json(PROTOCOL_PATH))
    require(tuple(protocol["source_frame_range"]) == FULL_RANGE, "foreign long protocol range")
    plan = staging.chunk_ranges(protocol)
    require(
        int(through_frame) in {stop for _, stop in plan}, "through_frame is not a chunk endpoint"
    )
    seed_root, stage_root, input_root = map(_root, (seed_root, input_stage_root, input_root))
    pose, rig, seed_binding, hashes = _verify_seed(seed_root, input_root)
    context = _make_context()
    require(staging.exact_contract(context.protocol, protocol), "context protocol changed")
    require(
        {key: item["sha256"] for key, item in context.code_records.items()} == STAGER_CODE,
        "detector source differs from the frozen eight-file implementation",
    )
    require(
        staging.exact_contract(context.runtime.metadata, RECORDED_RUNTIME),
        "recorded detector runtime differs from the registered environment",
    )
    for item in (*context.input_records.values(), *context.code_records.values()):
        hashes[str(Path(item["path"]).resolve())] = item["sha256"]
    chunk_roots, records, background_hashes = [], [], None
    for index, (start, stop) in enumerate(plan):
        if stop > through_frame:
            break
        path, manifest, seal = _select_chunk(context, stage_root, start, stop)
        backgrounds = {view: manifest["backgrounds"][view]["sha256"] for view in VIEWS}
        if background_hashes is None:
            background_hashes = backgrounds
        require(
            backgrounds == background_hashes, "chunk substituted different fixed background pixels"
        )
        output_hashes, raw_links = {}, {}
        for role, info in seal["outputs"].items():
            target = Path(info["path"]).resolve(strict=True)
            output_hashes[role] = dict(path=str(target), sha256=info["sha256"])
            if role.endswith("_raw"):
                view = role.removesuffix("_raw")
                original = context.sources[view]
                require(
                    os.path.samefile(target, original), "chunk raw input is not the source hardlink"
                )
                stat = target.stat()
                raw_links[view] = dict(
                    path=str(target), source_path=str(original), sha256=info["sha256"],
                    bytes=stat.st_size, device=stat.st_dev, inode=stat.st_ino,
                    mtime_ns=stat.st_mtime_ns,
                )
            else:
                hashes[str(target)] = info["sha256"]
        manifest_path, seal_path = path / "manifest.json", path / "staging.provenance.json"
        hashes[str(manifest_path)] = sha256(manifest_path)
        hashes[str(seal_path)] = sha256(seal_path)
        records.append(dict(
            index=index, start=start, stop=stop,
            attempt_path=path.relative_to(stage_root).as_posix(),
            manifest_sha256=hashes[str(manifest_path)], provenance_sha256=hashes[str(seal_path)],
            output_hashes=output_hashes, raw_hardlinks=raw_links,
        ))
        chunk_roots.append(path)
    verify_prefix_records([], records)
    # Prepare-context and validate-attempt have checked every payload. Recheck
    # canonical files once at the end, without rereading each raw hardlink.
    for path, expected in hashes.items():
        require(sha256(path) == expected, f"long input changed during validation: {path}")
    for record in records:
        for link in record["raw_hardlinks"].values():
            stat = Path(link["path"]).stat()
            require(
                os.path.samefile(link["path"], link["source_path"])
                and (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
                == (link["device"], link["inode"], link["bytes"], link["mtime_ns"]),
                "chunk hardlink changed during validation",
            )
    fixed = dict(
        schema="three_body_long_fixed_inputs_v1", full_frame_range=list(FULL_RANGE),
        seed=seed_binding, protocol_path=str(PROTOCOL_PATH.resolve()),
        protocol_sha256=PROTOCOL_SHA256,
        detector_binding=context.binding,
        detector_inputs={key: dict(value) for key, value in context.input_records.items()},
        background_sha256=background_hashes,
        input_stage_root=str(stage_root),
        verification_limitations=[
            "Detector runtime metadata is recorded, not re-executed; native DLLs were not hashed.",
            "Individual chunks are complete; no global assembly seal is claimed by this loader.",
            "Native frame decoding and per-image SHA verification occur in SourceSequenceReader.",
            "No baseline/GT arrays are inputs; the registered correspondence is optimistic.",
        ],
    )
    return LongInputs(
        pose, rig, protocol, FULL_RANGE,
        np.arange(FULL_RANGE[0], int(through_frame), dtype=np.int64),
        tuple(chunk_roots), input_root, fixed, hashes, records, int(through_frame) == FULL_RANGE[1],
    )
