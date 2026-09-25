"""Pose-blind fixed120 spatial-unused samples for the same two-minute images.

Reuse the sealed first60 samples. Decode only the next60 native D2 masks and
reconstruct the deterministic fit sampler to exclude its coordinates. No video,
pose, ID, fit, renderer invocation, GPU initialization or detector inference.
Later scoring must bind both tracking arms to these source masks and config.
Reconstructed fit coordinates are not a saved-run diagnostic comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from types import SimpleNamespace

import compare_temporal_point64_v2 as proof
import numpy as np
import run_fast_tail_two_minutes as producer
import score_directional_minute_pixels as old
import torch

WORK, REPO, require = producer.WORK, producer.REPO, proof.require
START, HALF, STOP, CHUNK = 66400, 76360, 86320, 1660
FRAMES = tuple(range(START + 83, STOP, 166))
STAGE = "fast_tail_two_minute_unused_pixel_preparation_v1"
ORIGINAL = WORK / "long_remaining_receipted_supervisor_v1/phases/f076360-086320/tracking"
ORIGINAL_SHA = "f605619d94c59e2bd3313c00fdff32d8fa8a8e029319637d2a4d71961c80e763"
PREPARED = WORK / "directional_minute_unused_pixel_samples_v1"
PREPARED_SHA = "a219bbad9980acfc1e4d17928829fad0a39792c97be6bd9e14abc5e8938cfa4e"
PINS = {
    "work/three_body/run_fast_tail_two_minutes.py": (
        "8efbba7abb6cdedd0b077b8dc953a7a1d10c268b8aab7844b17f25cc337862af"
    ),
    "work/three_body/score_directional_minute_pixels.py": (
        "ef0e2a9ac06d582347b14832c5f344d0b29c637a5aa2294d54a66a1ecf102855"
    ),
    "work/three_body/score_temporal_disjoint_pixels.py": (
        "8a86b3942d93b5ed506ac85cb3520fa9ccd51a2a11c55201fd1884e4e5c31d47"
    ),
    "labs/20260909_three_body/observations.py": (
        "f1efcd27b31e63ce693229c9e765152acfc7984b3b8839f1792bcc2efd9cb5df"
    ),
    "labs/20260909_three_body/prepare_sequence.py": (
        "959e862cc58c5840a00a763d8a63b635de134492a637ac2f1e8a7131f25178b8"
    ),
}


def mask_input(seal, first, view):
    chunk = f"f{first:06d}-{first + CHUNK:06d}"
    rows = [
        r
        for r in seal["inputs"].values()
        if chunk in Path(r["path"]).parts and Path(r["path"]).name == view + ".npz"
    ]
    require(len(rows) == 1, "unique consumed mask chunk required")
    path = Path(rows[0]["path"]).resolve(strict=True)
    require(path.is_relative_to(WORK / "inputs"), "mask input escaped workspace input root")
    return path, rows[0]["sha256"]


def metadata(evidence):
    original = producer.read_seal(
        ORIGINAL,
        "continuation.provenance.json",
        "three_body_long_continuation",
        ORIGINAL_SHA,
        evidence,
    )
    previous = producer.read_seal(
        PREPARED, "preparation.provenance.json", old.PREP_STAGE, PREPARED_SHA, evidence
    )
    config = proof.read_json(producer.consumed(original, "checkpoint/metadata.json", evidence))[
        "context"
    ]["tracker_config"]
    summary = proof.read_json(producer.consumed(previous, "summary.json", evidence))
    require(
        original["seal"]["params"]["through_frame"] == STOP, "same86320 original horizon required"
    )
    require(
        summary["status"] == "ok"
        and summary["scored_frames"] == list(FRAMES[:60])
        and summary["sampler_config"] == config
        and summary["counts"] == [96, 96, 64]
        and summary["seed_offset"] == 8803010
        and summary["fitted_coordinate_overlap"] == 0
        and summary["pose_values_read"] is False,
        "first60 preparation differs",
    )
    require(
        config["mode"] == "detectron2"
        and config["width_ratio"] == 0.06
        and [config[k] for k in ("positive_samples", "near_samples", "global_samples")]
        == [384, 384, 256],
        "fixed sampling configuration differs",
    )
    # Sampler implementations were actually bound by the earlier preparation.
    for rel, digest in PINS.items():
        evidence.add(REPO / rel, digest)
        if "run_fast_tail_two_minutes" not in rel:
            require(
                previous["seal"]["inputs"].get(str(REPO / rel), {}).get("sha256") == digest,
                "earlier preparation used a different sampler/decoder",
            )
        if rel.startswith("labs/"):
            require(
                original["seal"]["inputs"].get(str(REPO / rel), {}).get("sha256") == digest,
                "original tracking used a different sampler/decoder",
            )
    sources = {}
    for first in range(START, STOP, CHUNK):
        for view in old.VIEWS:
            path, digest = mask_input(original["seal"], first, view)
            sources[str(path)] = digest
            if first < HALF:
                require(
                    summary["source_mask_sha256"].get(str(path)) == digest,
                    "reused sample mask differs from two-minute tracking input",
                )
    return original, previous, summary, config, sources


def validate_samples(arrays):
    schema = {
        "fit_pixels": ((3, 1024, 2), np.float64),
        "fit_target": ((3, 1024), np.float32),
        "fit_weights": ((3, 1024), np.float32),
        "fit_foreground": ((3, 1024), np.bool_),
        "pixels": ((3, 256, 2), np.float64),
        "target": ((3, 256), np.float32),
        "weights": ((3, 256), np.float32),
        "reused_fitted_coordinate": ((3, 256), np.bool_),
    }
    require(set(arrays) == set(schema), "fixed sample fields differ")
    for key, (shape, dtype) in schema.items():
        a = arrays[key]
        require(
            a.shape == shape and a.dtype == dtype and np.isfinite(a).all(),
            "sample schema/finite failure: " + key,
        )
    require(not arrays["reused_fitted_coordinate"].any(), "saved overlap flag")
    for vi in range(3):
        ids = []
        for key in ("fit_pixels", "pixels"):
            xy = arrays[key][vi]
            require(
                np.array_equal(xy, np.rint(xy)) and (xy >= 0).all() and (xy < 1200).all(),
                "native integer pixel centers required",
            )
            ids.append(xy[:, 1].astype(np.int64) * 1200 + xy[:, 0].astype(np.int64))
        require(not np.isin(ids[1], ids[0]).any(), "evaluation coordinate used by fit sampler")
    require(
        (arrays["weights"] >= 0).all() and (arrays["fit_weights"] >= 0).all(),
        "negative sample weights",
    )
    require(
        np.isin(arrays["target"], [0, 1]).all() and np.isin(arrays["fit_target"], [0, 1]).all(),
        "nonbinary sample target",
    )


def frame_from_masks(packed_by_view, first, native):
    require(first <= native < first + CHUNK, "native outside chunk")
    values = {}
    for view in old.VIEWS:
        packed = packed_by_view[view]
        require(
            np.array_equal(packed["frame_idx"], np.arange(first, first + CHUNK))
            and np.array_equal(packed["image_shape_hw"], [1200, 1200])
            and packed["view_available"].shape == (CHUNK,)
            and packed["view_available"].dtype == np.bool_,
            "native mask layout differs",
        )
        index = native - first
        decoded = old.decode_frame_masks(packed, index)
        require(
            hashlib.sha256(np.ascontiguousarray(decoded).tobytes()).hexdigest()
            == bytes(packed["frame_mask_sha256"][index]).decode(),
            "decoded mask SHA differs",
        )
        values[view] = decoded if packed["view_available"][index] else None
    return SimpleNamespace(
        frame_idx=native,
        masks=values,
        images_bgr={v: SimpleNamespace(shape=(1200, 1200, 3)) for v in old.VIEWS},
    )


def run(args):
    require(args.execute, "explicit preparation execution required")
    destination = args.output.resolve()
    require(
        destination.is_relative_to(WORK) and not destination.exists(), "new work output required"
    )
    evidence, output = proof.Evidence(), proof.Output(destination)
    started, begin, completed = proof.utc(), time.perf_counter(), []
    try:
        code = producer.local_code()
        for path, digest in code.items():
            evidence.add(path, digest)
            output.write(
                "code_snapshot/" + Path(path).relative_to(REPO).as_posix(), Path(path).read_bytes()
            )
        original, previous, prior_summary, config, sources = metadata(evidence)
        require(not torch.cuda.is_initialized(), "CPU-only preparation required")
        torch.set_num_threads(2)
        membership = proof.read_json(producer.consumed(previous, "membership.json", evidence))
        membership = [r for r in membership if r["frame"] in FRAMES[:60]]
        require(
            [(r["frame"], r["view"]) for r in membership]
            == [(f, v) for f in FRAMES[:60] for v in old.VIEWS],
            "reused membership order",
        )
        for native in FRAMES[:60]:
            path = producer.consumed(previous, f"frame_{native:06d}.npz", evidence)
            with np.load(path, allow_pickle=False) as z:
                arrays = {k: z[k].copy() for k in z.files}
            validate_samples(arrays)
            # Recompress unchanged sample arrays; first-half sampling is not repeated.
            output.write(f"frame_{native:06d}.npz", arrays)
            with np.load(output.root / f"frame_{native:06d}.npz", allow_pickle=False) as z:
                for k in arrays:
                    proof.same(arrays[k], z[k], "reused first60 array bytes")
            completed.append(native)
        for first in range(HALF, STOP, CHUNK):
            packed = {}
            for view in old.VIEWS:
                path, digest = mask_input(original["seal"], first, view)
                evidence.add(path, digest)
                with np.load(path, allow_pickle=False) as z:
                    packed[view] = {k: z[k].copy() for k in z.files}
            for native in (f for f in FRAMES[60:] if first <= f < first + CHUNK):
                frame = frame_from_masks(packed, first, native)
                arrays = old.fit_arrays(frame, config)
                pixels, target, weights, overlap, audit = old.unused_samples(
                    frame, arrays["fit_pixels"]
                )
                arrays.update(
                    pixels=pixels, target=target, weights=weights, reused_fitted_coordinate=overlap
                )
                validate_samples(arrays)
                output.write(f"frame_{native:06d}.npz", arrays)
                membership.extend(audit)
                completed.append(native)
            del packed
            print(
                json.dumps(dict(event="two_minute_pixel_samples_prepared", through=first + CHUNK)),
                flush=True,
            )
        require(completed == list(FRAMES) and len(membership) == 360, "fixed120 completion")
        output.write("membership.json", membership)
        output.write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                native_interval=[START, STOP],
                scored_frames=list(FRAMES),
                reused_frames=list(FRAMES[:60]),
                newly_sampled_frames=list(FRAMES[60:]),
                sampler_config=config,
                counts=list(old.COUNTS),
                seed_offset=old.SEED_OFFSET,
                calibration=prior_summary["calibration"],
                source_mask_sha256=sources,
                original_seal_sha256=ORIGINAL_SHA,
                reused_preparation_sha256=PREPARED_SHA,
                native_decoded_mask_checks=180,
                reused_sample_arrays_BYTEEXACT=True,
                fitted_coordinate_overlap=0,
                pixel_draws_per_arm=120 * 3 * 256,
                fit_coordinates_reconstructed_deterministically=True,
                saved_run_fit_diagnostics_compared=False,
                pose_values_read=False,
                raw_BGR_frames_decoded=0,
                models=0,
                GPU_initialized=False,
                temporal_holdout=False,
                identity_truth=False,
                elapsed_seconds=time.perf_counter() - begin,
                runtime=dict(
                    python=platform.python_version(), numpy=np.__version__, torch=torch.__version__
                ),
            ),
        )
        require(code == producer.local_code(), "loaded source set changed")
        require(not torch.cuda.is_initialized(), "preparation initialized CUDA")
        evidence.final_guard()
        output.final_guard()
        seal = dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=output.records,
        )
    except BaseException as error:
        seal = dict(
            stage=STAGE,
            status="failed",
            started_at=started,
            finished_at=proof.utc(),
            completed_frames=completed,
            inputs=evidence.paths,
            outputs=output.records,
            error_type=type(error).__name__,
            error=str(error),
        )
        with (destination / "preparation.provenance.json").open("xb") as stream:
            stream.write(proof.json_bytes(seal))
        raise
    with (destination / "preparation.provenance.json").open("xb") as stream:
        stream.write(proof.json_bytes(seal))
    return seal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    run(parser.parse_args())
