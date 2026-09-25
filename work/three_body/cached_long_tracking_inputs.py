"""Receipt-backed execution of the unchanged long-input verification contract.

Only packed-mask NPZ loading/semantic validation is delegated to ReceiptStore.
The frozen detector attempt validator still runs for every candidate attempt,
including cache hits. SourceSequenceReader remains responsible for raw decoding.
This module does not start a detector, tracker or GPU or monkeypatch old helpers.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "labs/20260909_three_body"
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))

from long_tracking_inputs import (  # noqa: E402
    EXPERIMENT,
    FULL_RANGE,
    INPUT_ROOT,
    PROTOCOL_PATH,
    PROTOCOL_SHA256,
    RECORDED_RUNTIME,
    STAGER_CODE,
    VIEWS,
    LongInputs,
    _make_context,
    _root,
    _verify_seed,
    read_json,
    require,
    sha256,
    staging,
    verify_prefix_records,
)
from mask_validation_receipts import ReceiptStore  # noqa: E402


def _select_chunk(context, stage_root, start, stop, receipt_store):
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
                receipt_store.validate(
                    path / manifest["masks"][view]["path"],
                    scope={
                        "view": view,
                        "frame_range": [start, stop],
                        "image_shape_hw": [1200, 1200],
                        "manifest_path": str(path / "manifest.json"),
                        "manifest_sha256": sha256(path / "manifest.json"),
                        "seal_path": str(path / "staging.provenance.json"),
                        "seal_sha256": sha256(path / "staging.provenance.json"),
                        "detector_binding": context.binding,
                        "expected_npz_sha256": seal["outputs"][view + "_masks"]["sha256"],
                    },
                )
            return path, manifest, seal
        except (OSError, ValueError, KeyError, TypeError) as exc:
            reasons.append(f"{path.name}: {type(exc).__name__}: {exc}")
    raise ValueError(f"missing completed chunk [{start},{stop}); rejected={reasons}")


def load_cached_long_inputs(
    seed_root, input_stage_root, input_root=INPUT_ROOT, *,
    receipt_store: ReceiptStore, through_frame,
):
    """Verify the same native prefix with separately recorded mask-validation reuse.

    The caller owns ``receipt_store`` and seals its ``uses`` separately; receipt
    records never alter LongInputs, fixed bindings or consumed chunk records.
    All original file hashes, detector seals, seed checks and final guards remain.
    Canonical raw files are still hashed at the original orchestration points;
    this helper only avoids repeated packed-mask decompression/validation.
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
        path, manifest, seal = _select_chunk(context, stage_root, start, stop, receipt_store)
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
    receipt_store.final_guard()
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
