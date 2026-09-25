"""CPU audit of sealed local A outputs and the shared A/B pixel experiment."""

import argparse
import json
from pathlib import Path

import numpy as np
import run_body_volume_probe as p


def completed(root, stage, name="experiment"):
    path = root / f"{name}.provenance.json"
    seal = json.loads(path.read_text(encoding="utf-8"))
    if seal["status"] != "ok" or seal["stage"] != stage:
        raise ValueError("the requested completed producer seal is absent")
    hashes = {str(path): p.base.sha256(path)}
    for key in ("inputs", "outputs"):
        for entry in seal[key].values():
            if "sha256" not in entry or not Path(entry["path"]).is_file():
                raise ValueError("all recorded evidence must be an existing SHA-bound file")
            if entry["path"] in hashes and hashes[entry["path"]] != entry["sha256"]:
                raise ValueError("inconsistent input/output digest")
            hashes[entry["path"]] = entry["sha256"]
    p.assert_hashes(hashes)
    return seal, hashes


def merge_hashes(target, extra):
    for path, digest in extra.items():
        if path in target and target[path] != digest:
            raise ValueError("a shared file has different recorded source/input bytes")
        target[path] = digest


def require_output(path, seal):
    resolved = str(path.resolve())
    matches = [
        entry
        for entry in seal["outputs"].values()
        if str(Path(entry["path"]).resolve()) == resolved
    ]
    if len(matches) != 1 or p.base.sha256(path) != matches[0]["sha256"]:
        raise ValueError("the consumed array/summary lacks a unique sealed output binding")
    return path


def equal(first, second):
    if (
        first.dtype != second.dtype
        or first.shape != second.shape
        or first.tobytes() != second.tobytes()
    ):
        raise ValueError("expected equal dtype, shape and C-order array bytes")


def audit_frame(arrays, samples, summary, b_pixels, b_curve):
    """No reranking by evaluation loss; inspect stored choices and conservation."""
    count = len(arrays["fit_pose_xyz"])
    if count != summary["joint_admitted_fit_count"]:
        raise ValueError("candidate count differs")
    equal(arrays["fit_input_pose_xyz"][0], arrays["original_pose_xyz"])
    lower, upper = np.array([0.0, 0.0, -6.0]), np.array([192.0, 192.0, 186.0])
    for arm in ("mask", "random"):
        starts = arrays[arm + "_pose_xyz"]
        outside = (starts < lower) | (starts > upper)
        equal(outside, arrays[arm + "_outside_coordinates"])
        equal(~outside.any((1, 2, 3)), arrays[arm + "_admissible"])
        mapping = arrays[arm + "_fit_index"]
        if mapping[0] != 0 or not np.all(mapping[~arrays[arm + "_admissible"]] == -1):
            raise ValueError("invalid shared base or rejected-trial map")
        selected = mapping[mapping >= 0]
        equal(arrays["fit_input_pose_xyz"][selected], starts[mapping >= 0])
        expected = int(selected[np.argmin(arrays["fit_scores"][selected])])
        if summary["selected"]["fit_indices"][arm] != expected:
            raise ValueError("an arm winner was not selected by fitting score")
    for prefix in ("fit_", "fit_initial_"):
        source = arrays[prefix + "sources"]
        equal((source >= 0).sum(axis=2) >= 2, arrays[prefix + "observed"])
        for plan in source:
            for view in range(3):
                for node in range(3):
                    active = plan[:, view, node]
                    active = active[active >= 0]
                    if len(active) != len(np.unique(active)):
                        raise ValueError("two bodies use the same view/node source row")
    if np.any(arrays["fit_scores"] > arrays["fit_initial_scores"] + 1e-10):
        raise ValueError("fit discarded its lower-loss start")
    fitted = arrays["fit_pose_xyz"]
    if np.any((fitted < lower) | (fitted > upper)) or not np.isfinite(fitted).all():
        raise ValueError("fitted body is outside the engineering box")
    equal((fitted <= lower) | (fitted >= upper), arrays["fit_boundary_coordinates"])
    loss = (
        (arrays["evaluation_union"] - samples["evaluation_target"]) ** 2
        * samples["evaluation_weights"]
        * arrays["evaluation_ray_valid"]
    )
    parts = np.stack(
        [np.sum(loss * (samples["evaluation_strata"] == k), axis=-1) for k in range(3)], axis=-1
    )
    np.testing.assert_allclose(parts, arrays["evaluation_loss_by_stratum"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        parts.sum(-1).mean(-1), arrays["evaluation_mean_view_loss"], rtol=1e-12, atol=1e-12
    )
    # B used another driver and frozen zero-offset curved renderer. Pixels must
    # coincide exactly; the original numerical render is checked at 1e-12.
    for vi, view in enumerate(p.base.VIEWS):
        equal(samples["fit_pixels_" + view], b_pixels["fit_uv"][vi])
        equal(samples["evaluation_pixels_" + view], b_pixels[view + "_eval_uv"])
    for a_key, b_key in [
        ("fit_target", "fit_target"),
        ("fit_weights", "fit_weights"),
        ("evaluation_target", "eval_target"),
        ("evaluation_weights", "eval_weights"),
        ("evaluation_strata", "eval_strata"),
        ("evaluation_sampled_counts", "eval_counts"),
        ("evaluation_pool_counts", "eval_pool_counts"),
        ("evaluation_excluded_fit_unique_count", "eval_excluded_fit_unique_count"),
        ("evaluation_view_available", "eval_view_available"),
        ("evaluation_foreground_nonempty", "eval_foreground_nonempty"),
    ]:
        equal(samples[a_key], b_pixels[b_key])
    equal(arrays["original_pose_xyz"], b_curve["pose_xyz"])
    original = (
        (b_curve["original_fit_union"] - b_pixels["fit_target"]) ** 2
        * b_pixels["fit_weights"]
        * b_curve["fit_ray_valid"]
    ).sum() / 3
    np.testing.assert_allclose(
        original, arrays["fit_initial_loss_parts"][0, 4], rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        arrays["evaluation_union"][0], b_curve["original_eval_union"], rtol=1e-12, atol=1e-12
    )
    return dict(
        frame=summary["frame"],
        fitted=count,
        original_silhouette=float(original),
        original_silhouette_ab_abs_difference=float(
            abs(original - arrays["fit_initial_loss_parts"][0, 4])
        ),
        shared_pixels_dtype_shape_bytes_equal=True,
        all_arm_indices_sources_boundary_and_evaluation_replay=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--curve", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(p.REPO) or output.exists():
        raise FileExistsError("new workspace audit output required")
    output.mkdir(parents=True)
    with p.provenance("three_body_local_volume_output_audit_cpu", output, name="audit") as record:
        hashes = {str(path): p.base.sha256(path) for path in [*p.source_paths(), Path(__file__)]}
        a, a_hashes = completed(args.run.resolve(), "three_body_local_volume_start_probe")
        b, b_hashes = completed(
            args.curve.resolve(), "fixed_node_curve_spatial_diagnostic", "curve"
        )
        merge_hashes(hashes, a_hashes)
        merge_hashes(hashes, b_hashes)
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
        summary = json.loads(
            require_output(args.run / "summary.json", a).read_text(encoding="utf-8")
        )
        if summary["native_frame_probes"] != list(p.FRAMES):
            raise ValueError("not the thirteen registered native frames")
        rows = []
        for frame_summary in summary["frames"]:
            frame = frame_summary["frame"]
            root = args.run / f"frame_{frame}"
            rows.append(
                audit_frame(
                    p._npz(require_output(root / "candidates.npz", a)),
                    p._npz(require_output(root / "sampled_targets.npz", a)),
                    frame_summary,
                    p._npz(require_output(args.curve / f"f{frame}_pixels.npz", b)),
                    p._npz(require_output(args.curve / f"f{frame}_curve.npz", b)),
                )
            )
        if [row["frame"] for row in rows] != list(p.FRAMES):
            raise ValueError("incomplete or reordered frame summaries")
        result = dict(
            status="completed_output_conservation_and_ab_pixel_audit",
            gpu_used=False,
            biological_ids_verified=False,
            frames=rows,
            matched_producer_stages=[a["stage"], b["stage"]],
        )
        p.save_json(record, output, "summary", "summary.json", result)
        p.assert_hashes({**hashes, **{v["path"]: v["sha256"] for v in record.outputs.values()}})
        record.param("source_input_output_sha_unchanged", True)
    return result


if __name__ == "__main__":
    main()
