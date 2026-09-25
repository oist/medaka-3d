"""Pose-blind fixed300 spatial-unused samples for five minutes of tracking images.

Reuse all sealed120 samples; decode only the next180 native D2 masks. The
deterministic fit sampler supplies 1024 excluded coordinates per view. No video,
pose, identity inference, fitter, renderer, CUDA or new detector inference runs.
This is a same-time spatial holdout, not temporal holdout or identity truth.
The later evaluator must bind both tracking arms to these masks and config.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import prepare_fast_tail_two_minute_pixels as prior

producer, proof, old, torch = prior.producer, prior.proof, prior.old, prior.torch
WORK, REPO, require = prior.WORK, prior.REPO, prior.require
START, REUSED_STOP, STOP, CHUNK = 66400, 86320, 116200, 1660
REUSED_COUNT = 120
FRAMES = tuple(range(START + 83, STOP, 166))
STAGE = "fast_tail_five_minute_unused_pixel_preparation_v1"
ORIGINAL = WORK / "long_remaining_receipted_supervisor_v1/phases/f106240-116200/tracking"
ORIGINAL_SHA = "e99be2c7ea160752e93c52df924a77decebcd9d39955594c1dfda6cc0bb53005"
PREPARED = WORK / "fast_tail_two_minute_unused_pixel_samples_v1"
PREPARED_SHA = "85436a3a012d352e48b246e2e9b372352db2b086e820dd060c22ef3453b7d347"
PRIOR_CODE_SHA = "25db5d375dc353e2c640b3c67c287bde4f0b3a5f3e5c7fd9829c1d04b4522f9f"
PINS = {
    **prior.PINS,
    "work/three_body/prepare_fast_tail_two_minute_pixels.py": PRIOR_CODE_SHA,
}
mask_input = prior.mask_input
validate_samples = prior.validate_samples
frame_from_masks = prior.frame_from_masks


def custom_seal(root, digest, evidence):
    """The reused preparation has size_bytes outputs and string input digests."""
    root = root.resolve(strict=True)
    evidence.add(root / "preparation.provenance.json", digest)
    seal = proof.read_json(root / "preparation.provenance.json")
    require(
        seal.get("stage") == prior.STAGE and seal.get("status") == "ok",
        "successful fixed120 preparation required",
    )
    index = {}
    for name, item in seal["outputs"].items():
        path = Path(item["path"]).resolve()
        require(
            not Path(name).is_absolute()
            and ".." not in Path(name).parts
            and path == (root / name).resolve()
            and path.is_relative_to(root)
            and proof.valid_sha(item["sha256"])
            and type(item["size_bytes"]) is int
            and item["size_bytes"] >= 0,
            "unsafe custom output member",
        )
        require(str(path) not in index, "custom output alias")
        index[str(path)] = item
    return dict(root=root, seal=seal, index=index)


def custom_member(bundle, name, evidence):
    path = (bundle["root"] / name).resolve(strict=True)
    row = bundle["index"].get(str(path))
    require(row is not None and path.stat().st_size == row["size_bytes"], "unsealed/resized sample")
    evidence.add(path, row["sha256"])
    return path


def validate_previous(summary, config):
    required = dict(
        schema=prior.STAGE,
        status="ok",
        native_interval=[START, REUSED_STOP],
        scored_frames=list(FRAMES[:REUSED_COUNT]),
        reused_frames=list(FRAMES[:60]),
        newly_sampled_frames=list(FRAMES[60:REUSED_COUNT]),
        sampler_config=config,
        counts=[96, 96, 64],
        seed_offset=8803010,
        original_seal_sha256=prior.ORIGINAL_SHA,
        reused_preparation_sha256=prior.PREPARED_SHA,
        native_decoded_mask_checks=180,
        reused_sample_arrays_BYTEEXACT=True,
        fitted_coordinate_overlap=0,
        pixel_draws_per_arm=92160,
        fit_coordinates_reconstructed_deterministically=True,
        saved_run_fit_diagnostics_compared=False,
        pose_values_read=False,
        raw_BGR_frames_decoded=0,
        models=0,
        GPU_initialized=False,
        temporal_holdout=False,
        identity_truth=False,
    )
    require(all(summary.get(k) == v for k, v in required.items()), "fixed120 protocol differs")
    require(
        config["mode"] == "detectron2"
        and config["width_ratio"] == 0.06
        and [config[k] for k in ("positive_samples", "near_samples", "global_samples")]
        == [384, 384, 256],
        "fixed sampling configuration differs",
    )


def bind_sources(original_seal, previous_seal, summary):
    sources = {}
    reused = {}
    for first in range(START, STOP, CHUNK):
        for view in old.VIEWS:
            path, digest = mask_input(original_seal, first, view)
            sources[str(path)] = digest
            if first < REUSED_STOP:
                reused[str(path)] = digest
                require(
                    summary["source_mask_sha256"].get(str(path)) == digest,
                    "reused sample mask differs from five-minute tracking input",
                )
                if first >= prior.HALF:
                    require(
                        previous_seal["inputs"].get(str(path)) == digest,
                        "reused second-minute mask consumption differs",
                    )
    require(summary["source_mask_sha256"] == reused, "reused mask source membership differs")
    return sources


def metadata(evidence):
    original = producer.read_seal(
        ORIGINAL,
        "continuation.provenance.json",
        "three_body_long_continuation",
        ORIGINAL_SHA,
        evidence,
    )
    previous = custom_seal(PREPARED, PREPARED_SHA, evidence)
    config = proof.read_json(producer.consumed(original, "checkpoint/metadata.json", evidence))[
        "context"
    ]["tracker_config"]
    summary = proof.read_json(custom_member(previous, "summary.json", evidence))
    require(original["seal"]["params"]["through_frame"] == STOP, "same116200 horizon required")
    validate_previous(summary, config)
    for rel, digest in PINS.items():
        evidence.add(REPO / rel, digest)
        require(
            previous["seal"]["inputs"].get(str(REPO / rel)) == digest,
            "earlier preparation used a different sampler/decoder",
        )
        if rel.startswith("labs/"):
            require(
                original["seal"]["inputs"].get(str(REPO / rel), {}).get("sha256") == digest,
                "original tracking used a different sampler/decoder",
            )
    require(
        previous["seal"]["inputs"].get(str(prior.PREPARED / "preparation.provenance.json"))
        == prior.PREPARED_SHA,
        "first-minute preparation lineage differs",
    )
    calibration = [
        str(Path(p).parent)
        for p in original["seal"]["params"]["bindings"]["inputs"]["seed"][
            "SL_and_calibration_sha256"
        ]
        if Path(p).name == "calibration_manifest.json"
    ]
    require(calibration == [summary["calibration"]], "same calibration lineage required")
    sources = bind_sources(original["seal"], previous["seal"], summary)
    return original, previous, summary, config, sources


def validate_membership_order(membership, frames):
    require(
        [(r["frame"], r["view"]) for r in membership]
        == [(f, v) for f in frames for v in old.VIEWS],
        "all-view native membership order differs",
    )


def reuse_samples(previous, evidence, output):
    membership = proof.read_json(custom_member(previous, "membership.json", evidence))
    validate_membership_order(membership, FRAMES[:REUSED_COUNT])
    completed = []
    for native in FRAMES[:REUSED_COUNT]:
        path = custom_member(previous, f"frame_{native:06d}.npz", evidence)
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k].copy() for k in z.files}
        validate_samples(arrays)
        output.write(f"frame_{native:06d}.npz", arrays)
        with np.load(output.root / f"frame_{native:06d}.npz", allow_pickle=False) as z:
            for k in arrays:
                proof.same(arrays[k], z[k], "reused first120 array bytes")
        completed.append(native)
    return membership, completed


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
        membership, completed = reuse_samples(previous, evidence, output)
        new_archive_checks = 0
        decoded_checks = 0
        for first in range(REUSED_STOP, STOP, CHUNK):
            packed = {}
            for view in old.VIEWS:
                path, digest = mask_input(original["seal"], first, view)
                evidence.add(path, digest)
                with np.load(path, allow_pickle=False) as z:
                    packed[view] = {k: z[k].copy() for k in z.files}
                new_archive_checks += 1
            for native in (f for f in FRAMES[REUSED_COUNT:] if first <= f < first + CHUNK):
                frame = frame_from_masks(packed, first, native)
                decoded_checks += len(old.VIEWS)
                arrays = old.fit_arrays(frame, config)
                pixels, target, weights, overlap, audit = old.unused_samples(
                    frame, arrays["fit_pixels"]
                )
                arrays.update(
                    pixels=pixels, target=target, weights=weights, reused_fitted_coordinate=overlap
                )
                validate_samples(arrays)
                validate_membership_order(audit, [native])
                output.write(f"frame_{native:06d}.npz", arrays)
                membership.extend(audit)
                completed.append(native)
            del packed
            print(
                json.dumps(dict(event="five_minute_pixel_samples_prepared", through=first + CHUNK)),
                flush=True,
            )
        require(completed == list(FRAMES), "fixed300 completion")
        validate_membership_order(membership, FRAMES)
        require(new_archive_checks == 54 and decoded_checks == 540, "new mask consumption differs")
        output.write("membership.json", membership)
        output.write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                native_interval=[START, STOP],
                scored_frames=list(FRAMES),
                reused_frames=list(FRAMES[:REUSED_COUNT]),
                newly_sampled_frames=list(FRAMES[REUSED_COUNT:]),
                sampler_config=config,
                counts=list(old.COUNTS),
                seed_offset=old.SEED_OFFSET,
                calibration=prior_summary["calibration"],
                source_mask_sha256=sources,
                original_seal_sha256=ORIGINAL_SHA,
                reused_preparation_sha256=PREPARED_SHA,
                newly_consumed_mask_archives=new_archive_checks,
                native_decoded_mask_checks=decoded_checks,
                reused_mask_payload_reads=0,
                reused_sample_arrays_BYTEEXACT=True,
                fitted_coordinate_overlap=0,
                pixel_draws_per_arm=len(FRAMES) * 3 * 256,
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
