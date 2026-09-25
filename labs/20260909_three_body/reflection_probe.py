"""Fixed finite reflection-path response diagnostic; no fit or identity update."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BodyArrays:
    centers: np.ndarray
    tangents: np.ndarray
    axial_sigma_mm: np.ndarray
    radial_sigma_mm: np.ndarray

    @classmethod
    def from_torch(cls, bodies):
        return cls(
            **{
                key: getattr(bodies, key)[0].detach().cpu().numpy().copy()
                for key in cls.__dataclass_fields__
            }
        )


def finite_response(bodies, origins, directions, maximum_distance, valid, *, opacity=0.85):
    """Original Gaussian peak response, constrained to each finite water segment.

    Shapes: rays (...,3), distance/valid (...), bodies (3,K,3)/(3,K).
    Returns (...,3) per-body alpha. Invalid geometry may be NaN only if valid=False.
    Infinite maximum_distance is the original half-ray control, not a finite path.
    """
    origins, directions = np.asarray(origins, float), np.asarray(directions, float)
    limit, valid = np.asarray(maximum_distance, float), np.asarray(valid)
    if (
        origins.shape != directions.shape
        or origins.ndim < 2
        or origins.shape[-1] != 3
        or limit.shape != origins.shape[:-1]
        or valid.shape != limit.shape
        or valid.dtype != bool
    ):
        raise ValueError("ray origins/directions/limits/valid have inconsistent shapes")
    if (
        not np.isfinite(opacity)
        or not 0 < opacity < 1
        or np.any(np.isnan(limit[valid]))
        or np.any(limit[valid] < 0)
        or not np.isfinite(origins[valid]).all()
        or not np.isfinite(directions[valid]).all()
        or not np.allclose(np.linalg.norm(directions[valid], axis=-1), 1, rtol=0, atol=1e-10)
    ):
        raise ValueError("valid rays require unit finite directions and nonnegative limits")
    centers, tangent = np.asarray(bodies.centers), np.asarray(bodies.tangents)
    axial, radial = np.asarray(bodies.axial_sigma_mm), np.asarray(bodies.radial_sigma_mm)
    if (
        centers.ndim != 3
        or centers.shape[0] != 3
        or centers.shape[-1] != 3
        or tangent.shape != centers.shape
        or axial.shape != centers.shape[:-1]
        or radial.shape != axial.shape
        or min(axial.shape) < 1
        or not all(np.isfinite(x).all() for x in (centers, tangent, axial, radial))
        or np.any(axial <= 0)
        or np.any(radial <= 0)
        or not np.allclose(np.linalg.norm(tangent, axis=-1), 1, rtol=0, atol=1e-10)
    ):
        raise ValueError("body Gaussian shapes/scales/tangents are invalid")
    shape = limit.shape
    output = np.zeros((*shape, 3), float)
    locations = np.flatnonzero(valid)
    origins, directions, limit = origins.reshape(-1, 3), directions.reshape(-1, 3), limit.ravel()
    for start in range(0, len(locations), 4096):
        indices = locations[start : start + 4096]
        o, d = origins[indices, None, None], directions[indices, None, None]
        delta = centers - o
        dp = np.sum(delta * tangent, axis=-1)
        vp = np.sum(d * tangent, axis=-1)
        dperp, vperp = delta - dp[..., None] * tangent, d - vp[..., None] * tangent
        ia, ir = np.square(1 / axial), np.square(1 / radial)
        numerator = np.sum(vperp * dperp, axis=-1) * ir + vp * dp * ia
        denominator = np.sum(np.square(vperp), axis=-1) * ir + np.square(vp) * ia
        depth = np.minimum(np.maximum(numerator / denominator, 0), limit[indices, None, None])
        residual = delta - depth[..., None] * d
        rp = np.sum(residual * tangent, axis=-1)
        rr = residual - rp[..., None] * tangent
        mahalanobis = np.sum(np.square(rr), axis=-1) * ir + np.square(rp) * ia
        alpha = opacity * np.exp(-0.5 * mahalanobis)
        output.reshape(-1, 3)[indices] = 1 - np.prod(1 - alpha, axis=-1)
    if not np.isfinite(output).all():
        raise FloatingPointError("finite-response arithmetic became nonfinite")
    return output


def combine_branches(direct, inner, outer, inner_weight, outer_weight):
    """Symmetric geometric unions; weights are not calibrated image probabilities."""
    direct, inner, outer = (np.asarray(x, float) for x in (direct, inner, outer))
    wi, wo = np.asarray(inner_weight, float), np.asarray(outer_weight, float)
    if (
        direct.shape != inner.shape
        or direct.shape != outer.shape
        or direct.shape[-1] != 3
        or wi.shape != direct.shape[:-1]
        or wo.shape != wi.shape
        or not all(
            np.isfinite(x).all() and np.all((x >= 0) & (x <= 1))
            for x in (direct, inner, outer, wi, wo)
        )
    ):
        raise ValueError("branch alphas and weights must be bounded and aligned")
    zero = np.zeros_like(direct)

    def union(*x):
        return 1 - np.prod(1 - np.stack(x, axis=-1), axis=(-2, -1))

    return dict(
        finite_direct=union(direct, zero, zero),
        inner_unit=union(direct, inner, zero),
        outer_unit=union(direct, zero, outer),
        both_unit=union(direct, inner, outer),
        inner_fresnel=union(direct, inner * wi[..., None], zero),
        outer_fresnel=union(direct, zero, outer * wo[..., None]),
        both_fresnel=union(direct, inner * wi[..., None], outer * wo[..., None]),
    )


FRAMES = (79316, 79318, 79322, 79355, 79370, 79372, 79381, 79394, 79396, 79405, 79547, 82719, 82720)


def render_paths(rig, pixels, pose, config, paths_module):
    """Render all fixed alternatives on exactly the same calibrated pixels."""
    import dataclasses

    import torch
    from renderer import RefractiveFishRenderer

    engine = RefractiveFishRenderer(rig, pixels, config, dtype=torch.float64, device="cpu")
    with torch.no_grad():
        control = engine.render(pose)
    bodies = BodyArrays.from_torch(control.bodies)
    origins = engine.ray_origins.numpy()
    directions = engine.ray_directions.numpy()
    valid = engine.ray_valid.numpy()
    original = control.per_fish_alpha[0].numpy()
    baseline = finite_response(
        bodies, origins, directions, np.full(valid.shape, np.inf), valid, opacity=config.opacity
    )
    error = float(np.max(np.abs(baseline - original)))
    union = 1 - np.prod(1 - baseline, axis=-1)
    error = max(error, float(np.max(np.abs(union - control.union_alpha[0].numpy()))))
    if error > 1e-12:
        raise AssertionError(f"unbounded float64 response differs from original: {error}")
    arrays = dict(
        calibrated_valid=valid,
        unbounded_alpha=baseline,
        unbounded_direct=union,
        ray_origins=origins,
        ray_directions=directions,
    )
    direct, inner, outer, wi, wo, direct_valid = [], [], [], [], [], []
    traces = []
    for v, view in enumerate(rig.views):
        camera = rig.camera(view)
        trace = paths_module.trace_y192_reflections(
            np.where(valid[v, :, None], origins[v], np.nan),
            np.where(valid[v, :, None], directions[v], np.nan),
            index_water=camera.index_water,
            index_glass=camera.index_glass,
            index_air=camera.index_air,
        )
        traces.append(trace)
        direct_valid.append(trace.direct_valid)
        direct.append(
            finite_response(
                bodies,
                origins[v],
                directions[v],
                trace.first_distance_mm,
                trace.direct_valid,
                opacity=config.opacity,
            )
        )
        for destination, branch in ((inner, trace.inner), (outer, trace.outer)):
            destination.append(
                finite_response(
                    bodies,
                    branch.origins,
                    branch.directions,
                    branch.max_distance_mm,
                    branch.valid,
                    opacity=config.opacity,
                )
            )
        wi.append(trace.inner.weight)
        wo.append(trace.outer.weight)
        for field in dataclasses.fields(trace):
            value = getattr(trace, field.name)
            if dataclasses.is_dataclass(value):
                for child in dataclasses.fields(value):
                    arrays[f"{view}_{field.name}_{child.name}"] = getattr(value, child.name)
            else:
                arrays[f"{view}_{field.name}"] = value
    arrays.update(
        direct_alpha=np.stack(direct),
        inner_alpha=np.stack(inner),
        outer_alpha=np.stack(outer),
        inner_weight=np.stack(wi),
        outer_weight=np.stack(wo),
        direct_valid=np.stack(direct_valid),
    )
    arrays.update(
        combine_branches(
            arrays["direct_alpha"],
            arrays["inner_alpha"],
            arrays["outer_alpha"],
            arrays["inner_weight"],
            arrays["outer_weight"],
        )
    )
    for field in dataclasses.fields(bodies):
        arrays["gaussian_" + field.name] = getattr(bodies, field.name)
    return arrays, traces, error


def region_metrics(arrays, view_index, selection):
    """Full selected-pixel denominator; unsupported physical paths stay zero."""
    selection = np.asarray(selection, bool)
    count = int(selection.sum())
    metrics = dict(
        pixels=count,
        calibrated_valid=int(arrays["calibrated_valid"][view_index, selection].sum()),
        finite_direct_valid=int(arrays["direct_valid"][view_index, selection].sum()),
    )
    modes = (
        "unbounded_direct",
        "finite_direct",
        "inner_unit",
        "outer_unit",
        "both_unit",
        "inner_fresnel",
        "outer_fresnel",
        "both_fresnel",
    )
    metrics["mass"] = {name: float(arrays[name][view_index, selection].sum()) for name in modes}
    metrics["mean"] = {
        name: value / count if count else None for name, value in metrics["mass"].items()
    }
    metrics["unexplained_mass"] = {name: count - mass for name, mass in metrics["mass"].items()}
    metrics["new_mass_relative_to_finite_direct"] = {
        name: mass - metrics["mass"]["finite_direct"] for name, mass in metrics["mass"].items()
    }
    metrics["per_body_branch_mass"] = {
        name: arrays[name + "_alpha"][view_index, selection].sum(axis=0).tolist()
        for name in ("direct", "inner", "outer")
    }
    metrics["per_body_fresnel_branch_mass"] = {
        name: (
            arrays[name + "_alpha"][view_index, selection]
            * arrays[name + "_weight"][view_index, selection, None]
        )
        .sum(axis=0)
        .tolist()
        for name in ("inner", "outer")
    }
    return metrics


def path_metrics(trace, count, paths_module, selection=None):
    indices = np.arange(count) if selection is None else np.flatnonzero(selection)
    if np.any(indices >= count):
        raise ValueError("path region includes padded pixels")
    count = len(indices)

    def counts(values, enum):
        keys, values = np.unique(values[indices], return_counts=True)
        return {enum(int(k)).name: int(v) for k, v in zip(keys, values, strict=True)}

    result = dict(
        pixels=count,
        direct_valid=int(trace.direct_valid[indices].sum()),
        first_wall=counts(trace.first_wall, paths_module.Wall),
        first_reason=counts(trace.first_reason, paths_module.Reason),
    )
    for name in ("inner", "outer"):
        branch = getattr(trace, name)
        good = branch.valid[indices]
        weights = branch.weight[indices][good]
        result[name] = dict(
            valid=int(good.sum()),
            valid_fraction=float(good.mean()) if count else None,
            reasons=counts(branch.reason, paths_module.Reason),
            weight_min=float(weights.min()) if len(weights) else None,
            weight_max=float(weights.max()) if len(weights) else None,
            weight_mean=float(weights.mean()) if len(weights) else None,
        )
    return result


def check_saved_pixels(path, fit, evaluation, views):
    """Require the previous fixed spatial pixels, without reading fitted curves."""
    expected = dict(
        fit_uv=np.stack([fit.pixels[v] for v in views]),
        fit_target=fit.target,
        fit_weights=fit.weights,
        eval_target=evaluation.target,
        eval_weights=evaluation.weights,
        eval_strata=evaluation.strata,
        eval_counts=evaluation.sampled_counts,
        eval_pool_counts=evaluation.pool_counts,
        eval_excluded_fit_unique_count=evaluation.excluded_fit_unique_count,
        eval_view_available=evaluation.view_available,
        eval_foreground_nonempty=evaluation.foreground_nonempty,
        **{v + "_eval_uv": evaluation.pixels[v] for v in views},
    )
    with np.load(path, allow_pickle=False) as saved:
        if int(saved["frame_idx"]) != evaluation.frame_idx:
            raise ValueError("saved spatial pixels belong to another frame")
        for key, value in expected.items():
            other = saved[key]
            if (
                other.dtype != value.dtype
                or other.shape != value.shape
                or other.tobytes() != value.tobytes()
            ):
                raise ValueError(f"spatial sample differs: {key}")
    return expected


def visualize(rig, pose, config, masks, raw, output, paths_module):
    """Static 4-pixel display grid; quantitative sampling is separate and unchanged."""
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    x, y = np.meshgrid(np.arange(0, 1200, 4), np.arange(0, 1200, 4))
    uv = np.column_stack((x.ravel(), y.ravel())).astype(float)
    pixels = {v: uv if i == 0 else np.empty((0, 2)) for i, v in enumerate(rig.views)}
    response, _, error = render_paths(rig, pixels, pose, config, paths_module)
    d2 = masks.any(axis=0) if masks is not None and len(masks) else np.zeros((1200, 1200), bool)
    figure, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    images = [None, "finite_direct", "both_unit", "inner", "outer", "both_fresnel"]
    titles = [
        "Native raw + D2 contours",
        "Finite direct response",
        "Direct + both, unit",
        "Inner reflection, unit",
        "Outer reflection, unit",
        "Direct + both, Fresnel",
    ]
    for axis, mode, title in zip(axes.ravel(), images, titles, strict=True):
        axis.imshow(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
        axis.contour(d2, levels=[0.5], colors=["cyan"], linewidths=0.55)
        if mode is not None:
            values = (
                1 - np.prod(1 - response[mode + "_alpha"][0], axis=-1)
                if mode in ("inner", "outer")
                else response[mode][0]
            )
            values = values.reshape(x.shape)
            axis.imshow(
                values,
                extent=(-2, 1198, 1198, -2),
                origin="upper",
                vmin=0,
                vmax=1,
                cmap="magma",
                alpha=np.clip(values * 0.8, 0, 0.8),
                interpolation="nearest",
            )
            for level, style in ((0.1, ":"), (0.5, "-")):
                if values.min() < level < values.max():
                    axis.contour(
                        x,
                        y,
                        values,
                        levels=[level],
                        colors=["orange"],
                        linestyles=[style],
                        linewidths=0.7,
                    )
        axis.set(
            xlim=(0, 1199),
            ylim=(1199, 0),
            title=title,
            xlabel="native x / pixel",
            ylabel="native y / pixel",
        )
    figure.suptitle(
        "79381 top camera | fixed geometry, no fit | cyan=D2; orange=.1/.5 response\n"
        "Display rays every 4 native pixels; engineering response, not brightness or ID"
    )
    figure.savefig(output / "f79381_top_reflection.png", dpi=140)
    if masks is not None and len(masks) > 3 and masks[3].any():
        yy, xx = np.nonzero(masks[3])
        for axis in axes.ravel():
            axis.set_xlim(max(0, int(xx.min()) - 140), min(1199, int(xx.max()) + 140))
            axis.set_ylim(min(1199, int(yy.max()) + 140), max(0, int(yy.min()) - 140))
        figure.savefig(output / "f79381_top_primary_detail.png", dpi=140)
    plt.close(figure)
    np.savez_compressed(output / "f79381_top_display.npz", pixels=uv, **response)
    if not cv2.imwrite(str(output / "f79381_top_raw.png"), raw):
        raise OSError("raw PNG write failed")
    if cv2.imread(str(output / "f79381_top_raw.png")).tobytes() != raw.tobytes():
        raise AssertionError("visual raw PNG changed pixels")
    return dict(native_grid_step=4, sample_pixels=len(uv), unbounded_parity_max_abs=error)


def main(argv=None):
    import argparse
    import importlib
    import json
    import shutil
    import sys
    from pathlib import Path

    lab = Path(__file__).resolve().parent
    repo = lab.parents[1]
    sys.path[:0] = [str(lab), str(repo / "src")]
    import cv2
    import identity_inputs
    import reflection_paths
    import torch
    from prepare_sequence import sha256

    from medaka3d.provenance import provenance

    # Bind all measurement modules before verification/registration, even though
    # real measurement imports their public functions only after registration.
    for dependency in ("body_pixel_evaluation", "node_geometry", "renderer"):
        importlib.import_module(dependency)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        default=repo / "work/three_body/persistent_mouth_continuous_one_minute_v1",
    )
    parser.add_argument(
        "--pixels-run", type=Path, default=repo / "work/three_body/curve_body_spatial_13frames_v1"
    )
    parser.add_argument("--register-only", action="store_true")
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(repo):
        raise ValueError("output must be new and inside the workspace")
    protocol_path = lab / "REFLECTION_PROBE_PROTOCOL.md"
    code_paths = {Path(__file__).resolve(), protocol_path}
    for module in tuple(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name:
            path = Path(name).resolve()
            if path.suffix == ".py" and (
                path.is_relative_to(lab) or path.is_relative_to(repo / "src")
            ):
                code_paths.add(path)
    code = {str(p): sha256(p) for p in sorted(code_paths)}
    verified = identity_inputs._verify_run(args.run, appearance=False)
    if not np.array_equal(verified.frames, np.arange(76620, 86580)):
        raise ValueError("probe is registered only for the completed fixed minute")
    cfg = verified.params["tracker_config"]
    if (
        cfg["mode"] != "detectron2"
        or cfg["width_ratio"] != 0.06
        or [cfg[k] for k in ("positive_samples", "near_samples", "global_samples")]
        != [384, 384, 256]
    ):
        raise ValueError("fixed geometry/detector/sampling configuration differs")
    original_sources = [
        lab / name
        for name in ("renderer.py", "observations.py", "node_geometry.py", "prepare_sequence.py")
    ]
    original_sources += [repo / "src/medaka3d" / name for name in ("calib.py", "types.py")]
    for path in original_sources:
        if sha256(path) != verified.code[path.relative_to(repo).as_posix()]:
            raise ValueError("measurement source differs from the frozen geometry producer")
    pixels_root = args.pixels_run.resolve(strict=True)
    pixels_seal_path = pixels_root / "curve.provenance.json"
    pixels_seal = json.loads(pixels_seal_path.read_text())
    if pixels_seal["status"] != "ok":
        raise ValueError("previous fixed spatial sampling is not sealed")
    hashes = {**verified.hashes, str(pixels_seal_path): sha256(pixels_seal_path)}
    for frame in FRAMES:
        path = pixels_root / f"f{frame}_pixels.npz"
        entry = pixels_seal["outputs"][path.name]
        if Path(entry["path"]).resolve() != path or sha256(path) != entry["sha256"]:
            raise ValueError("previous fixed pixel arrays do not match the sealed source")
        hashes[str(path)] = entry["sha256"]
    plan = dict(
        schema="fixed_reflection_probe_v1",
        frames=list(FRAMES),
        run=str(verified.root),
        pixels_run=str(pixels_root),
        protocol_sha256=sha256(protocol_path),
        code_hashes=code,
        input_hashes=dict(hashes),
        no_fit=True,
        no_gt=True,
        device="cpu",
        primary_frame=79381,
        primary_view="top_cam",
        primary_instance_row=3,
        package_versions=dict(
            python=sys.version,
            numpy=np.__version__,
            torch=str(torch.__version__),
            opencv=cv2.__version__,
        ),
        torch_cpu_threads_during_measurement=2,
    )
    if not args.register_only:
        if args.registration is None:
            raise ValueError("a completed pre-render registration is required")
        registration = args.registration.resolve(strict=True)
        seal_path = registration / "registration.provenance.json"
        seal = json.loads(seal_path.read_text())
        plan_path = registration / "plan.json"
        if (
            seal["status"] != "ok"
            or sha256(plan_path) != seal["outputs"]["plan"]["sha256"]
            or json.loads(plan_path.read_text()) != plan
        ):
            raise ValueError("registered plan/input/source bindings differ")
        hashes.update({str(seal_path): sha256(seal_path), str(plan_path): sha256(plan_path)})
    output.mkdir(parents=True, exist_ok=False)
    name = "registration" if args.register_only else "probe"
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        with provenance("fixed_reflection_" + name, output, name=name) as record:
            for path, digest in {**hashes, **code}.items():
                record.input(path, path, sha256=digest)
            for path, digest in code.items():
                target = output / "code_snapshot" / Path(path).relative_to(repo)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                if sha256(target) != digest:
                    raise AssertionError("source snapshot changed")
                record.output(target.relative_to(output).as_posix(), target, sha256=digest)
            path = output / "plan.json"
            path.write_text(json.dumps(plan, indent=2, allow_nan=False), encoding="utf-8")
            record.output("plan", path, sha256=sha256(path))
            if args.register_only:
                print("Registered fixed reflection probe; no image rendering performed", flush=True)
            else:
                _run_measurements(verified, pixels_root, output, record, reflection_paths, cfg)
            for path, digest in {**hashes, **code}.items():
                if sha256(path) != digest:
                    raise AssertionError("input/source changed during reflection diagnostic")
            for entry in record.outputs.values():
                if sha256(entry["path"]) != entry["sha256"]:
                    raise AssertionError("output/snapshot changed during reflection diagnostic")
    finally:
        torch.set_num_threads(old_threads)


def _run_measurements(verified, pixels_root, output, record, paths_module, cfg):
    import dataclasses
    import json
    import time

    import cv2
    import identity_inputs
    from body_pixel_evaluation import sample_body_evaluation, summarize_pixel_error
    from node_geometry import RayObservations, supported_nodes
    from observations import CachedNodes, FrameObservation, sample_pixels
    from prepare_sequence import VIEWS, array_sha256, decode_frame_masks, sha256
    from renderer import RendererConfig

    from medaka3d.calib import load_rig

    started = time.perf_counter()
    original = identity_inputs._npz(verified.root / "candidate.npz")
    if set(original) != identity_inputs.GEOMETRY_ARRAYS:
        raise ValueError("expected the sealed persistent 21-array geometry schema")
    identity_inputs.persistent_runner.validate_arm_arrays(original, verified.frames)
    rig = load_rig(verified.root / "calibration_snapshot")
    if tuple(rig.views) != tuple(VIEWS):
        raise ValueError("calibration view order differs")
    nodes = CachedNodes(verified.input_root, verified.manifest["experiment"], verified.frames)
    packed = {
        v: identity_inputs._npz(verified.sequence / verified.manifest["masks"][v]["path"])
        for v in VIEWS
    }
    for data in packed.values():
        if not np.array_equal(data["frame_idx"], verified.frames):
            raise ValueError("D2 native frame binding differs")
    config = RendererConfig(width_ratio=cfg["width_ratio"])
    rows = []
    modes = (
        "unbounded_direct",
        "finite_direct",
        "inner_unit",
        "outer_unit",
        "both_unit",
        "inner_fresnel",
        "outer_fresnel",
        "both_fresnel",
    )

    def save_npz(path, arrays):
        np.savez_compressed(path, **arrays)
        record.output(path.name, path, sha256=sha256(path))

    for frame in FRAMES:
        t = int(frame - verified.frames[0])
        pose, sources, observed = (original[k][t] for k in ("pose_xyz", "sources", "observed"))
        points = {v: nodes.at(v, frame) for v in VIEWS}
        rays = RayObservations.from_points(rig, points)
        replay_sources, replay_observed, _ = supported_nodes(pose, sources, rays, rig)
        if not np.array_equal(sources, replay_sources) or not np.array_equal(
            observed, replay_observed
        ):
            raise ValueError("original node support no longer matches calibrated SLEAP gates")
        masks = {}
        for view in VIEWS:
            data = packed[view]
            decoded = decode_frame_masks(data, t)
            if array_sha256(decoded) != data["frame_mask_sha256"][t].decode():
                raise ValueError("decoded D2 mask hash differs")
            masks[view] = decoded if data["view_available"][t] else None
            if masks[view] is None and len(decoded):
                raise ValueError("unavailable D2 view contains detections")
        dummy = np.zeros((1200, 1200, 3), np.uint8)
        observation = FrameObservation(frame, points, dict.fromkeys(VIEWS, dummy), masks, {})
        fit = sample_pixels(observation, "detectron2", 384, 384, 256)
        evaluation = sample_body_evaluation(observation, fit.pixels)
        saved = check_saved_pixels(pixels_root / f"f{frame}_pixels.npz", fit, evaluation, VIEWS)
        save_npz(output / f"f{frame}_pixels.npz", saved)
        info = dict(
            frame_idx=frame,
            pose_xyz=pose.tolist(),
            observed=observed.tolist(),
            sources=sources.tolist(),
            original_review=original["identity_review"][t].tolist(),
            physical_box_outside_nodes=int(
                np.any((pose < [0, 0, 0]) | (pose > [192, 192, 186]), axis=-1).sum()
            ),
            paths={},
            foreground={},
            core={},
            evaluation={},
            fitting={},
        )
        for label, target in (("fitting", fit), ("evaluation", evaluation)):
            if max(map(len, target.pixels.values())) == 0:
                info[label] = dict(neutral=True, reason="all_views_missing_or_detector_empty")
                continue
            response, traces, error = render_paths(rig, target.pixels, pose, config, paths_module)
            save_npz(output / f"f{frame}_{label}_response.npz", response)
            values = {}
            for mode in modes:
                if label == "evaluation":
                    measured = summarize_pixel_error(
                        response[mode], response["calibrated_valid"], evaluation
                    )
                    values[mode] = {k: v.tolist() for k, v in measured.items()}
                else:
                    loss = (
                        (response[mode] - fit.target) ** 2
                        * fit.weights
                        * response["calibrated_valid"]
                    ).sum(axis=-1)
                    values[mode] = dict(
                        loss_by_view=loss.tolist(), mean_view_loss=float(loss.mean())
                    )
            info[label] = dict(neutral=False, unbounded_parity_max_abs=error, scores=values)
            info["paths"][label] = {
                v: path_metrics(traces[i], len(target.pixels[v]), paths_module)
                for i, v in enumerate(VIEWS)
            }
            if label == "fitting":
                old_loss = float(original["loss_parts"][t, 4])
                if abs(values["unbounded_direct"]["mean_view_loss"] - old_loss) > 1e-10:
                    raise ValueError("original fitting silhouette no longer matches producer")
            else:
                info["evaluation"]["stratum_response"] = {
                    view: {
                        str(k): region_metrics(response, v, evaluation.strata[v] == k)
                        for k in range(3)
                    }
                    for v, view in enumerate(VIEWS)
                }
        full, flats = {}, {}
        for view in VIEWS:
            mask = masks[view]
            union = (
                mask.any(axis=0) if mask is not None and len(mask) else np.zeros((1200, 1200), bool)
            )
            flats[view] = np.flatnonzero(union)
            full[view] = np.column_stack((flats[view] % 1200, flats[view] // 1200)).astype(float)
        if max(map(len, full.values())):
            dense, traces, error = render_paths(rig, full, pose, config, paths_module)
            save_npz(
                output / f"f{frame}_foreground_response.npz",
                {**dense, **{v + "_pixels": full[v] for v in VIEWS}},
            )
            info["paths"]["foreground"] = {
                v: path_metrics(traces[i], len(full[v]), paths_module) for i, v in enumerate(VIEWS)
            }
            info["foreground_parity_max_abs"] = error
            centers = dense["gaussian_centers"]
            info["physical_box_outside_gaussian_centers"] = int(
                np.any((centers < [0, 0, 0]) | (centers > [192, 192, 186]), axis=-1).sum()
            )
            for v, view in enumerate(VIEWS):
                mask = masks[view]
                mask = np.zeros((0, 1200, 1200), bool) if mask is None else mask
                selection = np.arange(dense["finite_direct"].shape[1]) < len(full[view])
                info["foreground"][view] = dict(
                    union=region_metrics(dense, v, selection), instances=[]
                )
                info["core"][view] = []
                first = int(packed[view]["frame_offsets"][t])
                for instance, m in enumerate(mask):
                    selected = np.zeros_like(selection)
                    selected[: len(full[view])] = m.ravel()[flats[view]]
                    row = region_metrics(dense, v, selected)
                    row.update(
                        instance_row=instance,
                        detector_score=float(packed[view]["scores"][first + instance]),
                        paths=path_metrics(
                            traces[v], len(full[view]), paths_module, selected[: len(full[view])]
                        ),
                    )
                    info["foreground"][view]["instances"].append(row)
                    core = cv2.erode(
                        m.astype(np.uint8),
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                        borderType=cv2.BORDER_CONSTANT,
                        borderValue=0,
                    ).astype(bool)
                    selected[: len(full[view])] = core.ravel()[flats[view]]
                    info["core"][view].append(
                        dict(instance_row=instance, **region_metrics(dense, v, selected))
                    )
        if frame == 79381:
            raw_info = verified.manifest["raw"][VIEWS[0]]
            capture = cv2.VideoCapture(str(verified.sequence / raw_info["path"]))
            try:
                expected_count = len(verified.frames)
                if (
                    not capture.isOpened()
                    or capture.get(cv2.CAP_PROP_FRAME_COUNT) != expected_count
                    or capture.get(cv2.CAP_PROP_FPS) != 166
                ):
                    raise ValueError("native FFV1 metadata differs")
                if (
                    not capture.set(cv2.CAP_PROP_POS_FRAMES, t)
                    or capture.get(cv2.CAP_PROP_POS_FRAMES) != t
                ):
                    raise ValueError("native visual seek differs")
                ok, raw = capture.read()
                if (
                    not ok
                    or raw is None
                    or raw.shape != (1200, 1200, 3)
                    or raw.dtype != np.uint8
                    or capture.get(cv2.CAP_PROP_POS_FRAMES) != t + 1
                    or array_sha256(raw) != raw_info["frame_bgr_sha256"][t]
                ):
                    raise ValueError("native raw frame does not match saved BGR SHA")
            finally:
                capture.release()
            info["visual"] = visualize(
                rig, pose, config, masks[VIEWS[0]], raw, output, paths_module
            )
            info["visual"]["native_bgr_sha256"] = array_sha256(raw)
            for filename in (
                "f79381_top_raw.png",
                "f79381_top_display.npz",
                "f79381_top_reflection.png",
            ):
                path = output / filename
                record.output(filename, path, sha256=sha256(path))
            detail = output / "f79381_top_primary_detail.png"
            if detail.exists():
                record.output(detail.name, detail, sha256=sha256(detail))
        rows.append(info)
        path = output / f"f{frame}_summary.json"
        path.write_text(json.dumps(info, indent=2, allow_nan=False), encoding="utf-8")
        record.output(path.name, path, sha256=sha256(path))
        print(json.dumps(dict(frame_idx=frame, complete=len(rows), total=len(FRAMES))), flush=True)
    values = {
        mode: [
            row["evaluation"]["scores"][mode]["mean_view_loss"][0]
            for row in rows
            if not row["evaluation"]["neutral"]
        ]
        for mode in modes
    }
    summary = dict(
        frames=list(FRAMES),
        mode="fixed_geometry_no_fit",
        frame_results=rows,
        mean_spatial_evaluation_loss={
            k: float(np.mean(v)) if v else None for k, v in values.items()
        },
        neutral_frame_count=sum(row["evaluation"]["neutral"] for row in rows),
        renderer_config=dataclasses.asdict(config),
        wall_box_lower=[0, 0, 0],
        wall_box_upper=[192, 192, 186],
        inner_y_mm=192,
        outer_y_mm=196,
        camera_indices={
            v: {k: getattr(rig.camera(v), k) for k in ("index_water", "index_glass", "index_air")}
            for v in VIEWS
        },
        native_raw_frames_decoded=1,
        pixel_content_used_for_sampling=False,
        gt_opened=False,
        geometry_or_identity_modified=False,
        device="cpu",
        elapsed_seconds=time.perf_counter() - started,
    )
    path = output / "summary.json"
    path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    record.output("summary", path, sha256=sha256(path))


if __name__ == "__main__":
    main()
