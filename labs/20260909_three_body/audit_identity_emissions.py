"""Replay sealed GPU identity image scores on CPU from every native raw frame.

The same registered renderer, sampling and color primitive are reused. This
checks CPU/GPU arithmetic and raw input binding, not an independent image model
or biological identity truth. Neither DP selection nor geometry fitting runs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import evaluate_identity_dp as evidence
import numpy as np
import run_experiment as base
import scipy
import torch
from identity_emissions import IdentityEmissionConfig, score_identity_permutations
from observations import SequenceReader
from pixel_appearance_loss import PixelAppearanceConfig
from prepare_sequence import array_sha256

from medaka3d.provenance import provenance

STAGE = "three_body_cpu_identity_image_emission_audit"
ATOL = RTOL = 1e-12
IMAGE_FIELDS = {
    "frame_idx": ((), "int64"),
    "unary": ((6,), "float64"),
    "per_view_costs": ((6, 3), "float64"),
    "enabled_views": ((3,), "bool"),
    "sampled_pixels": ((3,), "int64"),
    "valid_pixels": ((3,), "int64"),
    "available_core_pixels": ((3,), "int64"),
    "visible_mass": ((3, 3), "float64"),
    "rendering_performed": ((), "bool"),
}
MOTION_FIELDS = {"transitions", "supported_pairs", "supported_pair_count", "squared_distance_mm2"}


def validate_image_arrays(arrays, frames, config):
    """Require the producer's full score schema, but audit only the image fields here."""
    frames = np.asarray(frames)
    if not evidence.equal_json(config, evidence.FIXED_CONFIG):
        raise ValueError("CPU replay requires the registered fixed identity configuration")
    if (
        frames.dtype != np.int64
        or frames.ndim != 1
        or len(frames) < 166
        or np.any(np.diff(frames) != 1)
    ):
        raise ValueError("all native frames and the complete enrollment prefix are required")
    if set(arrays) != set(IMAGE_FIELDS) | MOTION_FIELDS:
        raise ValueError("saved score arrays are missing or unexpected")
    for key, (tail, dtype) in IMAGE_FIELDS.items():
        value = arrays[key]
        if value.shape != (len(frames), *tail) or value.dtype != np.dtype(dtype):
            raise ValueError(f"saved image score shape/dtype mismatch: {key}")
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise ValueError(f"saved image score is nonfinite: {key}")
        if key != "frame_idx" and np.any(value < 0):
            raise ValueError(f"saved image score is negative: {key}")
    if not np.array_equal(arrays["frame_idx"], frames):
        raise ValueError("saved image score native frames differ from geometry")
    for key in IMAGE_FIELDS.keys() - {"frame_idx"}:
        if np.any(arrays[key][:166] != 0):
            raise ValueError(f"enrollment frames must not have image scores/rendering: {key}")


def compare_image_array(actual, expected, key):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError(f"CPU/GPU image score shape/dtype mismatch: {key}")
    if actual.dtype.kind == "f":
        if not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise ValueError(f"CPU/GPU image scores must be finite: {key}")
        if not np.allclose(actual, expected, rtol=RTOL, atol=ATOL, equal_nan=False):
            raise ValueError(f"CPU/GPU image score exceeds 1e-12 tolerance: {key}")
        return dict(
            shape=list(actual.shape),
            dtype=str(actual.dtype),
            maximum_absolute_error=float(np.max(np.abs(actual - expected), initial=0)),
            cpu_sha256=array_sha256(actual),
            gpu_sha256=array_sha256(expected),
            within_tolerance=True,
            comparison="atol=rtol=1e-12",
        )
    if not np.array_equal(actual, expected):
        raise ValueError(f"CPU/GPU image count/flag mismatch: {key}")
    return dict(
        shape=list(actual.shape),
        dtype=str(actual.dtype),
        exact=True,
        cpu_sha256=array_sha256(actual),
        gpu_sha256=array_sha256(expected),
    )


def replay_emissions(reader, rig, geometry, enrollment, saved_scores, config, *, progress=None):
    """Independent native-frame loop; only the registered single-frame math is reused.

    Reviews, source rows, the saved chosen ID path and GT do not enter rendering.
    The original raw geometry and frozen causal template identities are supplied
    exactly as before. Future frames cannot change the enrollment templates.
    """
    frames = np.asarray(geometry["frame_idx"])
    validate_image_arrays(saved_scores, frames, config)
    if not np.array_equal(reader.frames, frames):
        raise ValueError("raw reader and geometry native intervals differ")
    n = len(frames)
    arrays = {key: np.zeros((n, *tail), dtype=dtype) for key, (tail, dtype) in IMAGE_FIELDS.items()}
    arrays["frame_idx"][:] = frames
    emission = IdentityEmissionConfig(**config["emission"])
    pixel = PixelAppearanceConfig(**config["pixel"])
    processed, rendered, native_hashes = 0, 0, []
    iterator = iter(reader)
    start = time.perf_counter()
    try:
        for index, frame in enumerate(iterator):
            if index >= n or frame.frame_idx != int(frames[index]):
                raise ValueError("raw replay contains an extra, missing or reordered native frame")
            if index >= 166:
                result = score_identity_permutations(
                    rig,
                    frame,
                    geometry["pose_xyz"][index],
                    enrollment["templates"],
                    enrollment["enabled_views"],
                    config=emission,
                    pixel_config=pixel,
                    device="cpu",
                )
                arrays["unary"][index] = config["appearance_weight"] * result.costs
                for key in IMAGE_FIELDS.keys() - {"frame_idx", "unary"}:
                    arrays[key][index] = getattr(result, key)
                rendered += int(result.rendering_performed)
            native_hashes.append(
                {
                    "frame_idx": int(frame.frame_idx),
                    "bgr_sha256": {v: array_sha256(frame.images_bgr[v]) for v in base.VIEWS},
                    "mask_available": {v: frame.masks[v] is not None for v in base.VIEWS},
                    "decoded_masks_sha256": {
                        v: None if frame.masks[v] is None else array_sha256(frame.masks[v])
                        for v in base.VIEWS
                    },
                }
            )
            # Compare as frames arrive, so a failed audit does not continue unnecessary rendering.
            for key in IMAGE_FIELDS.keys() - {"frame_idx"}:
                compare_image_array(arrays[key][index], saved_scores[key][index], key)
            processed += 1
            if progress and (processed % 100 == 0 or processed == n):
                progress(
                    dict(
                        processed=processed,
                        requested=n,
                        source_frame=int(frame.frame_idx),
                        elapsed_seconds=time.perf_counter() - start,
                    )
                )
    finally:
        if hasattr(iterator, "close"):
            iterator.close()
    if processed != n:
        raise ValueError("raw replay ended before the complete native interval")
    comparisons = {
        key: compare_image_array(value, saved_scores[key], key) for key, value in arrays.items()
    }
    return arrays, comparisons, native_hashes, rendered


def code_paths():
    result = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == ".py" and (
                path.is_relative_to(base.LAB) or path.is_relative_to(base.REPO / "src")
            ):
                result.add(path)
    return sorted(result)


def runtime_versions():
    # Reading Torch's version string does not initialize CUDA or use its device.
    return dict(
        numpy=np.__version__,
        scipy=scipy.__version__,
        opencv=cv2.__version__,
        torch=str(torch.__version__),
    )


def assert_unchanged(hashes):
    for path, digest in hashes.items():
        if base.sha256(path) != digest:
            raise RuntimeError("an image audit input or source changed during replay")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run = evidence.local_path(args.run)
    output = args.output.resolve()
    if not output.is_relative_to(base.REPO) or output.exists() or output.is_symlink():
        raise ValueError("CPU image audit requires a new workspace output directory")
    sources = {str(path): base.sha256(path) for path in code_paths()}
    # verify_completed is the evaluator's read-only evidence gate; its annotation-reading
    # evaluation main is never called. Its summary contains model costs, not GT scores.
    summary, context, hashes = evidence.verify_completed(run)
    if len(context.frame_idx) not in (200, 666):
        raise ValueError("this bounded raw CPU audit supports full 200 or 666 native frames")
    versions = runtime_versions()
    producer = evidence.read_json(run / "experiment.provenance.json")
    if any(str(producer["params"]["package_versions"].get(k)) != v for k, v in versions.items()):
        raise ValueError("CPU replay requires the producer's NumPy/SciPy/OpenCV/Torch versions")
    assert_unchanged(sources)
    hashes = {**hashes, **sources}
    saved = evidence.npz(run / "scores.npz")
    validate_image_arrays(saved, context.frame_idx, summary["config"])
    output.mkdir(parents=True, exist_ok=False)
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with provenance(STAGE, output, name="audit") as record:
            record.param("run", str(run))
            record.param("output", str(output))
            record.param("device", "cpu")
            record.param("atol", ATOL)
            record.param("rtol", RTOL)
            record.param("package_versions", versions)
            record.param("torch_cpu_threads", torch.get_num_threads())
            record.param("ground_truth_or_reference_arrays_opened", False)
            record.param("geometry_fit_or_identity_optimization_performed", False)
            for path, digest in hashes.items():
                record.input(path, path, sha256=digest)
            snapshot_hashes = {}
            for path, digest in sources.items():
                target = output / "code_snapshot" / Path(path).relative_to(base.REPO)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                snapshot_hashes[str(target)] = digest
                record.output(
                    "code:" + str(Path(path).relative_to(base.REPO)), target, sha256=digest
                )
            assert_unchanged({**hashes, **snapshot_hashes})
            reader = SequenceReader(context.geometry_sequence_root, context.input_root)
            rig = base.load_rig(context.input_root / base.CALIBRATION_DIR)
            start = time.perf_counter()
            arrays, comparisons, frames, rendered = replay_emissions(
                reader,
                rig,
                context.geometry,
                context.enrollment,
                saved,
                summary["config"],
                progress=lambda event: print(json.dumps(event), flush=True),
            )
            path = output / "recomputed_image_scores.npz"
            np.savez_compressed(path, **arrays)
            record.output("recomputed_image_scores", path, sha256=base.sha256(path))
            qa = dict(
                status="complete_raw_cpu_image_replay",
                run=str(run),
                device="cpu",
                frame_range=[int(context.frame_idx[0]), int(context.frame_idx[-1] + 1)],
                native_frames=len(context.frame_idx),
                decoded_images=3 * len(context.frame_idx),
                prefix_frames_without_rendering=166,
                rendered_frames=rendered,
                all_six_states_recomputed=True,
                atol=ATOL,
                rtol=RTOL,
                comparisons=comparisons,
                frames=frames,
                elapsed_seconds_including_replay_and_npz_save=time.perf_counter() - start,
                source_scores_sha256=hashes[str(run / "scores.npz")],
                ground_truth_or_reference_arrays_opened=False,
                geometry_fit_or_identity_optimization_performed=False,
                identity_labels_verified=False,
                limitations=[
                    "This reuses the same sampling, renderer and color formulas on CPU.",
                    "CPU/GPU score agreement does not validate image modeling or identity truth.",
                    "The DP path and motion costs are outside this image replay audit.",
                ],
            )
            assert_unchanged({**hashes, **snapshot_hashes})
            record.param("inputs_and_sources_unchanged", True)
            path = output / "summary.json"
            path.write_text(json.dumps(qa, indent=2, allow_nan=False), encoding="utf-8")
            record.output("summary", path, sha256=base.sha256(path))
    finally:
        torch.set_num_threads(old_threads)
    return qa


if __name__ == "__main__":
    main()
