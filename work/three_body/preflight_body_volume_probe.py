"""CPU-only native input and candidate-count preflight; no fit or ID inference."""

import argparse
import dataclasses
import json
import shutil
import time
from pathlib import Path

import numpy as np
import run_body_volume_probe as probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(probe.REPO) or output.exists():
        raise FileExistsError("a new workspace preflight directory is required")
    output.mkdir(parents=True)
    with probe.provenance(
        "three_body_body_volume_preflight_cpu", output, name="preflight"
    ) as record:
        code = {str(p): probe.base.sha256(p) for p in [*probe.source_paths(), Path(__file__)]}
        for path, digest in code.items():
            target = output / "code_snapshot" / Path(path).relative_to(probe.REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            record.output("code:" + path, target, sha256=digest)
        started = time.perf_counter()
        run = probe._verify_run(probe.DEFAULT_RUN, appearance=False)
        hashes = {**run.hashes, **code}
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
        geometry = probe._npz(run.root / "candidate.npz")
        if set(geometry) != probe.GEOMETRY_ARRAYS:
            raise ValueError("not the fixed twenty-one-array geometry schema")
        probe.persistent.validate_arm_arrays(geometry, run.frames)
        nodes = probe.CachedNodes(run.input_root, probe.base.EXPERIMENT, run.frames)
        probe.base.validate_result(geometry, run.frames, nodes)
        parents = probe.replay_chosen_parents(geometry, run.initial)
        rig = probe.load_rig(run.root / "calibration_snapshot")
        grid = probe.MaskGrid(rig, probe.MaskGridConfig())
        proposer = probe.InstanceVolumeProposer(
            grid.xyz,
            grid.projected_uv,
            grid.grid_shape,
            image_shape_hw=grid.config.image_shape_hw,
        )
        rows = []
        for t, frame, evidence in probe.decoded_probes(run, nodes):
            pool = proposer.evaluate(frame.masks)
            usable = [i for i, c in enumerate(pool.components) if c.principal_axis_defined]
            starts = probe.volume_scene_starts(
                geometry["pose_xyz"][t],
                [
                    dict(centroid=pool.components[i].centroid_xyz, axes=pool.components[i].pca_axes)
                    for i in usable
                ],
            )
            random = probe.random_scene_starts(
                geometry["pose_xyz"][t],
                starts["component_slot_sign_roll"][1:, 1],
                seed=frame.frame_idx * 7 + 9101,
            )
            combined, maps = probe.combine_admissible(starts, random)
            fit = probe.sample_pixels(frame, mode="detectron2")
            held = probe.sample_body_evaluation(frame, fit.pixels)
            target_arrays = probe.target_arrays(fit, held)
            arrays = {
                f"{arm}_{key}": value
                for arm, item in [("mask", starts), ("random", random)]
                for key, value in item.items()
                if isinstance(value, np.ndarray)
            }
            arrays.update({f"{arm}_fit_index": value for arm, value in maps.items()})
            arrays.update(
                combined_admitted_pose_xyz=combined,
                predicted_pose_xyz=parents["predicted_pose_xyz"][t],
                parent_missing=parents["parent_missing"][t],
            )
            arrays.update(target_arrays)
            probe.save_arrays(
                record,
                output,
                f"prepared:{frame.frame_idx}",
                f"prepared_{frame.frame_idx}.npz",
                arrays,
            )
            rows.append(
                dict(
                    frame=frame.frame_idx,
                    evidence=evidence,
                    pool_components=len(pool.components),
                    oriented_components=len(usable),
                    nominal_per_arm=len(starts["pose_xyz"]),
                    mask_admitted=int(starts["admissible"].sum()),
                    random_admitted=int(random["admissible"].sum()),
                    shared_fit_count=len(combined),
                    component_diagnostics=probe.component_payload(pool)[1],
                    evaluation_pool_counts=held.pool_counts,
                    evaluation_sampled_counts=held.sampled_counts,
                )
            )
            print(
                json.dumps(
                    {key: rows[-1][key] for key in ["frame", "nominal_per_arm", "shared_fit_count"]}
                ),
                flush=True,
            )
        if [row["frame"] for row in rows] != list(probe.FRAMES):
            raise ValueError("not all thirteen registered probes were prepared")
        summary = dict(
            status="cpu_preflight_completed",
            gpu_used=False,
            fit_executed=False,
            total_seconds=time.perf_counter() - started,
            grid_config=dataclasses.asdict(grid.config),
            shared_fit_count=sum(row["shared_fit_count"] for row in rows),
            frames=rows,
        )
        probe.save_json(record, output, "summary", "summary.json", summary)
        probe.assert_hashes({**hashes, **{v["path"]: v["sha256"] for v in record.outputs.values()}})
        record.param("source_input_snapshot_output_sha_unchanged", True)
        record.param("gt_or_baseline_trajectories_loaded", False)
    return summary


if __name__ == "__main__":
    main()
