"""CPU raw-image replay of a sealed observation-memory identity producer.

Only the image schema/stage changed. The native replay loop originates from
audit_identity_emissions.py (recorded SHA below); its renderer, pixel sampler
and CPU/GPU comparison primitive are reused unchanged. No fabricated adjacent
motion arrays, identity optimization, geometry fitting, or annotations are used.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import audit_identity_emissions as common
import evaluate_identity_memory as evidence
import numpy as np
from identity_emissions import IdentityEmissionConfig, score_identity_permutations
from pixel_appearance_loss import PixelAppearanceConfig

from medaka3d.provenance import provenance

base, torch = common.base, common.torch
STAGE = "three_body_cpu_identity_memory_image_emission_audit"
LOOP_ORIGIN = "labs/20260909_three_body/audit_identity_emissions.py"
LOOP_ORIGIN_SHA256 = "085397a7c8cd022e2b841e52bca67ea3c387f39d9dea8b79346eee6cc413c1ac"
IMAGE_FIELDS = common.IMAGE_FIELDS


def validate_image_arrays(saved, frames, enrollment, config):
    if not common.evidence.equal_json(config, evidence.FIXED_CONFIG):
        raise ValueError("memory image replay requires the registered complete configuration")
    if (np.asarray(frames).dtype != np.int64 or np.asarray(frames).ndim != 1
            or len(frames) < 166 or np.any(np.diff(frames) != 1)):
        raise ValueError("a complete native interval and enrollment prefix are required")
    evidence.validate_image_scores(saved, frames, enrollment, config)
    for key, (_, dtype) in IMAGE_FIELDS.items():
        if saved[key].dtype != np.dtype(dtype):
            raise ValueError(f"registered image field dtype differs: {key}")


def replay_emissions(reader, rig, geometry, enrollment, saved, config, *, progress=None):
    """Same native loop/math as the frozen auditor, with the actual nine-field schema.

    Keeping this short loop separate avoids modifying the old validator or
    inventing four unused adjacent-motion arrays to satisfy its legacy schema.
    Raw SequenceReader performs native seek, frame-count and BGR SHA checks.
    """
    frames = geometry["frame_idx"]
    validate_image_arrays(saved, frames, enrollment, config)
    if not np.array_equal(reader.frames, frames):
        raise ValueError("raw reader and immutable geometry frame intervals differ")
    n = len(frames)
    arrays = {key: np.zeros((n, *tail), dtype=dtype)
              for key, (tail, dtype) in IMAGE_FIELDS.items()}
    arrays["frame_idx"][:] = frames
    emission, pixel = IdentityEmissionConfig(**config["emission"]), PixelAppearanceConfig(
        **config["pixel"]
    )
    processed, rendered, native_hashes = 0, 0, []
    iterator, start = iter(reader), time.perf_counter()
    try:
        for index, frame in enumerate(iterator):
            if index >= n or frame.frame_idx != int(frames[index]):
                raise ValueError("extra, missing or reordered native replay frame")
            if index >= 166:
                result = score_identity_permutations(
                    rig, frame, geometry["pose_xyz"][index], enrollment["templates"],
                    enrollment["enabled_views"], config=emission, pixel_config=pixel, device="cpu",
                )
                arrays["unary"][index] = config["appearance_weight"] * result.costs
                for key in IMAGE_FIELDS.keys() - {"frame_idx", "unary"}:
                    arrays[key][index] = getattr(result, key)
                rendered += int(result.rendering_performed)
            native_hashes.append({
                "frame_idx": int(frame.frame_idx),
                "bgr_sha256": {v: common.array_sha256(frame.images_bgr[v]) for v in base.VIEWS},
                "mask_available": {v: frame.masks[v] is not None for v in base.VIEWS},
                "decoded_masks_sha256": {
                    v: None if frame.masks[v] is None else common.array_sha256(frame.masks[v])
                    for v in base.VIEWS
                },
            })
            for key in IMAGE_FIELDS.keys() - {"frame_idx"}:
                common.compare_image_array(arrays[key][index], saved[key][index], key)
            processed += 1
            if progress and (processed % 100 == 0 or processed == n):
                progress(dict(processed=processed, requested=n, source_frame=int(frame.frame_idx),
                              elapsed_seconds=time.perf_counter()-start))
    finally:
        if hasattr(iterator, "close"):
            iterator.close()
    if processed != n:
        raise ValueError("raw replay ended before the entire native interval")
    comparisons = {key: common.compare_image_array(value, saved[key], key)
                   for key, value in arrays.items()}
    return arrays, comparisons, native_hashes, rendered


def code_paths():
    return sorted(set(common.code_paths()) | {Path(__file__).resolve(), base.REPO / LOOP_ORIGIN})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run, output = common.evidence.local_path(args.run), args.output.resolve()
    if not output.is_relative_to(base.REPO) or output.exists() or output.is_symlink():
        raise ValueError("memory image audit requires a new workspace output directory")
    sources = {str(path): base.sha256(path) for path in code_paths()}
    if base.sha256(base.REPO / LOOP_ORIGIN) != LOOP_ORIGIN_SHA256:
        raise ValueError("the frozen original image replay implementation changed")
    # This no-GT verifier binds evidence; neither evaluator main nor its DP replay is called.
    summary, context, hashes = evidence.verify_completed(run)
    if len(context.frame_idx) not in (200, 666):
        raise ValueError("the bounded image audit supports full 200/666 native frames")
    versions = common.runtime_versions()
    producer = common.evidence.read_json(run / "experiment.provenance.json")
    if any(str(producer["params"]["package_versions"].get(k)) != v for k, v in versions.items()):
        raise ValueError("CPU replay requires the producer NumPy/SciPy/OpenCV/Torch versions")
    common.assert_unchanged(sources)
    hashes = {**hashes, **sources}
    saved = common.evidence.npz(run / "scores.npz")
    validate_image_arrays(saved, context.frame_idx, context.enrollment, summary["config"])
    output.mkdir(parents=True, exist_ok=False)
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with provenance(STAGE, output, name="audit") as record:
            for key, value in dict(
                run=str(run), output=str(output), device="cpu", atol=common.ATOL, rtol=common.RTOL,
                package_versions=versions, torch_cpu_threads=torch.get_num_threads(),
                loop_origin=LOOP_ORIGIN, loop_origin_sha256=LOOP_ORIGIN_SHA256,
                schema_adaptation="actual nine image fields; no adjacent-motion placeholders",
                ground_truth_or_reference_arrays_opened=False,
                geometry_fit_or_identity_optimization_performed=False,
            ).items():
                record.param(key, value)
            for path, digest in hashes.items():
                record.input(path, path, sha256=digest)
            snapshots = {}
            for path, digest in sources.items():
                target = output / "code_snapshot" / Path(path).relative_to(base.REPO)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                snapshots[str(target)] = digest
                record.output(
                    "code:" + str(Path(path).relative_to(base.REPO)), target, sha256=digest
                )
            common.assert_unchanged({**hashes, **snapshots})
            reader = common.SequenceReader(context.geometry_sequence_root, context.input_root)
            rig = base.load_rig(context.input_root / base.CALIBRATION_DIR)
            start = time.perf_counter()
            arrays, comparisons, frames, rendered = replay_emissions(
                reader, rig, context.geometry, context.enrollment, saved, summary["config"],
                progress=lambda event: print(json.dumps(event), flush=True),
            )
            path = output / "recomputed_image_scores.npz"
            np.savez_compressed(path, **arrays)
            record.output("recomputed_image_scores", path, sha256=base.sha256(path))
            qa = dict(
                status="complete_raw_cpu_memory_image_replay", run=str(run), device="cpu",
                frame_range=[int(context.frame_idx[0]), int(context.frame_idx[-1]+1)],
                native_frames=len(context.frame_idx), decoded_images=3*len(context.frame_idx),
                prefix_frames_without_rendering=166, rendered_frames=rendered,
                all_six_states_recomputed=True, atol=common.ATOL, rtol=common.RTOL,
                comparisons=comparisons, frames=frames,
                elapsed_seconds_including_replay_and_npz_save=time.perf_counter()-start,
                source_scores_sha256=hashes[str(run / "scores.npz")],
                ground_truth_or_reference_arrays_opened=False,
                geometry_fit_or_identity_optimization_performed=False,
                identity_labels_verified=False,
                loop_origin=LOOP_ORIGIN, loop_origin_sha256=LOOP_ORIGIN_SHA256,
                limitations=[
                    "Same registered sampler, renderer and color formula are reused on CPU.",
                    "Numerical CPU/GPU agreement does not establish image-model or ID accuracy.",
                    "Memory motion and DP selection are outside this raw-image replay audit.",
                ],
            )
            path = output / "summary.json"
            path.write_text(json.dumps(qa, indent=2, allow_nan=False), encoding="utf-8")
            record.output("summary", path, sha256=base.sha256(path))
            outputs = {item["path"]: item["sha256"] for item in record.outputs.values()}
            common.assert_unchanged({**hashes, **snapshots, **outputs})
            record.param("inputs_sources_snapshots_outputs_unchanged", True)
            if common.runtime_versions() != versions:
                raise RuntimeError("CPU image audit runtime versions changed")
    finally:
        torch.set_num_threads(old_threads)
    return qa


if __name__ == "__main__":
    main()
