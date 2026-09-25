"""Registered CPU-only local fitting; three fixed mask models, no tracking or GT."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

LAB = Path(__file__).resolve().parent
REPO = LAB.parents[1]
sys.path[:0] = [str(LAB), str(REPO / "src")]

import identity_inputs  # noqa: E402
from body_local_fit import replay_chosen_parents  # noqa: E402
from body_pixel_evaluation import sample_body_evaluation, summarize_pixel_error  # noqa: E402
from node_geometry import RayObservations, assignment_plans, supported_nodes  # noqa: E402
from observations import CachedNodes, FrameObservation, sample_pixels  # noqa: E402
from prepare_sequence import VIEWS, array_sha256, decode_frame_masks, sha256  # noqa: E402
from reflection_local_fit import fit_local  # noqa: E402
from reflection_probe import FRAMES, check_saved_pixels  # noqa: E402
from reflection_renderer import ReflectionFishRenderer  # noqa: E402
from renderer import RendererConfig  # noqa: E402
from tracker import TrackerConfig  # noqa: E402

from medaka3d.calib import load_rig  # noqa: E402
from medaka3d.provenance import provenance  # noqa: E402

ARMS = ("direct", "unit", "fresnel")
LOWER, UPPER = np.array([0., 0., -6.]), np.array([192., 192., 186.])
PROTOCOL = REPO / "work/three_body/REFLECTION_LOCAL_FIT_PLAN.md"
PROTOCOL_SHA = "07c21ec5b75f206959491f381f085b1e45efd32363b6dee28e54bbff66db3cc5"
PINNED = {
    "reflection_local_fit.py": "d77df5e6cd44a5eafbd22de0f89d5bad3e5033ab5f90540aed57f05237fc7a03",
    "reflection_renderer.py": "4099adb877aaaa98f5e289bf4bc9d5652aa469da2c7fa6b421fbada367da584b",
    "reflection_paths.py": "e8f64a776d546a35c7eb10c0af30edee896b9ab75b388d1df289dc4dc3cc86e3",
    "body_local_fit.py": "4f2517346a11a3e85766b4367b17392cdc28f7645b78390e9e1311c102a20e93",
    "reflection_probe.py": "1df7df16c83a621da1a23e7c3c9c2f95432ae63104dc8cb8ec7035343f055be4",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def bind(path, expected, hashes):
    path = Path(path).resolve(strict=True)
    require(path.is_relative_to(REPO), "evidence must remain in the workspace")
    digest = sha256(path)
    require(expected is None or digest == expected, f"SHA differs: {path}")
    require(hashes.get(str(path), digest) == digest, f"conflicting input: {path}")
    hashes[str(path)] = digest
    return path


def unchanged(hashes, outputs):
    for path, digest in hashes.items():
        require(sha256(path) == digest, f"input/source changed: {path}")
    for entry in outputs.values():
        require(sha256(entry["path"]) == entry["sha256"], "output/snapshot changed")


def source_hashes():
    paths = {Path(__file__).resolve(), PROTOCOL}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == ".py" and (
                path.is_relative_to(LAB) or path.is_relative_to(REPO / "src")
            ):
                paths.add(path)
    code = {str(p): sha256(p) for p in sorted(paths)}
    require(code[str(PROTOCOL)] == PROTOCOL_SHA, "registered protocol changed")
    for name, digest in PINNED.items():
        require(code[str(LAB / name)] == digest, f"registered helper changed: {name}")
    return code


def verify_evidence(args, code):
    # This verifier hashes candidate bytes but never deserializes its coordinates.
    verified = identity_inputs._verify_run(args.run, appearance=False)
    require(np.array_equal(verified.frames, np.arange(76620, 86580)), "fixed full minute required")
    old_cfg = TrackerConfig(**verified.params["tracker_config"])
    require(asdict(old_cfg) == asdict(TrackerConfig(flow_proposals=True)),
            "producer configuration differs from the fixed experiment")
    cfg = replace(old_cfg, device="cpu")
    for name in ("renderer.py", "observations.py", "node_geometry.py", "prepare_sequence.py",
                 "tracker.py", "graph_fit.py"):
        path = LAB / name
        require(code[str(path)] == verified.code[path.relative_to(REPO).as_posix()],
                f"local source differs from the producer: {name}")
    for name in ("calib.py", "types.py"):
        path = REPO / "src/medaka3d" / name
        require(code[str(path)] == verified.code[path.relative_to(REPO).as_posix()],
                f"calibration source differs: {name}")
    hashes = dict(verified.hashes)
    pixels_root = args.pixels_run.resolve(strict=True)
    pixel_seal = bind(pixels_root / "curve.provenance.json", None, hashes)
    sealed = read_json(pixel_seal)
    require(sealed["status"] == "ok" and sealed["stage"] == "fixed_node_curve_spatial_diagnostic",
            "completed original spatial sample artifact required")
    for path, digest in verified.hashes.items():
        entry = sealed["inputs"].get(path)
        require(entry is not None and Path(entry["path"]).resolve() == Path(path)
                and entry["sha256"] == digest, "spatial artifact producer input mismatch")
    for frame in FRAMES:
        path = pixels_root / f"f{frame}_pixels.npz"
        entry = sealed["outputs"][path.name]
        require(Path(entry["path"]).resolve() == path, "spatial output role mismatch")
        bind(path, entry["sha256"], hashes)
    audit_root = args.renderer_audit.resolve(strict=True)
    audit = read_json(bind(audit_root / "audit.provenance.json", None, hashes))
    require(audit["status"] == "ok" and audit["stage"] == "saved_reflection_renderer_cpu_parity",
            "successful real-pixel renderer parity required")
    for category in ("inputs", "outputs"):
        for entry in audit[category].values():
            bind(entry["path"], entry["sha256"], hashes)
    for name in ("reflection_renderer.py", "reflection_paths.py"):
        require(hashes.get(str(LAB / name)) == code[str(LAB / name)], "parity source mismatch")
    primary_probe_source = REPO / (
        "work/three_body/reflection_probe_13frames_v1/code_snapshot/"
        "labs/20260909_three_body/reflection_probe.py")
    require(hashes.get(str(primary_probe_source)) == code[str(LAB / "reflection_probe.py")],
            "parity's independent NumPy source differs")
    summary_path = audit_root / "summary.json"
    require(any(Path(e["path"]).resolve() == summary_path for e in audit["outputs"].values()),
            "renderer summary not bound as output")
    parity = read_json(summary_path)
    require(parity["status"] == "passed" and parity["frames"] == list(FRAMES)
            and parity["device"] == "cpu" and parity["dtype"] == "float64",
            "renderer parity conditions differ")
    for row in parity["rows"]:
        require(all(np.isfinite(v) and 0 <= v <= 1e-12
                    for v in row["maximum_absolute_difference"].values()),
                "renderer real-pixel parity failed")
    require(len(parity["rows"]) == 78, "renderer parity incomplete")
    return verified, cfg, pixels_root, hashes


def verify_registration(root, plan, hashes):
    root = Path(root).resolve(strict=True)
    seal_path = bind(root / "registration.provenance.json", None, hashes)
    seal = read_json(seal_path)
    require(seal["status"] == "ok" and seal["stage"] == "reflection_local_registration",
            "successful local-fit registration required")
    for entry in seal["outputs"].values():
        bind(entry["path"], entry["sha256"], hashes)
    path = root / "plan.json"
    require(Path(seal["outputs"]["plan"]["path"]).resolve() == path,
            "registration plan output role differs")
    require(read_json(path) == plan, "registered input/source/config/runtime plan differs")
    expected = {**plan["input_hashes"], **plan["code_hashes"]}
    require({k: v["sha256"] for k, v in seal["inputs"].items()} == expected,
            "registration input set differs")
    for key, entry in seal["inputs"].items():
        require(Path(entry["path"]).resolve() == Path(key), "registration input role differs")
        bind(entry["path"], entry["sha256"], hashes)


def assert_exclusive(sources):
    value = np.asarray(sources)
    require(value.shape == (3, 3, 3) and np.issubdtype(value.dtype, np.integer)
            and np.all(value >= -1), "source shape or integer values differ")
    for v in range(3):
        for n in range(3):
            used = value[:, v, n]
            used = used[used >= 0]
            require(len(used) == len(np.unique(used)), "same view/node source consumed twice")


def validate_fit(result, start, plan, cfg, rays, rig):
    steps = cfg.opt_steps + 1
    require(result["tracking_initializations"] == 0 and result["identity_updated"] is False,
            "local fitter changed tracking/identity state")
    np.testing.assert_array_equal(result["input_pose_xyz"], start)
    np.testing.assert_array_equal(result["plan"], plan)
    for key, shape in (("iteration_scores", (steps,)), ("iteration_loss_parts", (steps, 5)),
                       ("iteration_pose_xyz", (steps, 3, 3, 3))):
        require(np.asarray(result[key]).shape == shape and np.isfinite(result[key]).all(),
                f"incomplete finite iteration history: {key}")
    best = int(np.argmin(result["iteration_scores"]))
    require(result["best_step"] == best, "wrong best iterate")
    np.testing.assert_array_equal(result["pose_xyz"], result["iteration_pose_xyz"][best])
    np.testing.assert_allclose(result["score"], result["iteration_scores"][best],
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(result["loss_parts"], result["iteration_loss_parts"][best],
                               rtol=0, atol=1e-12)
    iteration_sources, iteration_observed = [], []
    for pose in result["iteration_pose_xyz"]:
        require(np.all((pose >= LOWER) & (pose <= UPPER)), "iteration outside fixed bounds")
        source, observed, _ = supported_nodes(pose, plan, rays, rig)
        assert_exclusive(source)
        iteration_sources.append(source)
        iteration_observed.append(observed)
    np.testing.assert_array_equal(result["sources"], iteration_sources[best])
    np.testing.assert_array_equal(result["observed"], iteration_observed[best])
    np.testing.assert_array_equal(result["boundary_coordinates"],
                                  (result["pose_xyz"] <= LOWER) | (result["pose_xyz"] >= UPPER))
    return np.stack(iteration_sources), np.stack(iteration_observed)


def evaluate_poses(rig, cfg, poses, target):
    if target.target.shape[1] == 0:
        valid = np.empty((3, 0), bool)
        unions = {name: np.empty((len(poses), 3, 0)) for name in ARMS}
    else:
        engine = ReflectionFishRenderer(
            rig, target.pixels, RendererConfig(width_ratio=cfg.width_ratio),
            direct_mode="unbounded", device="cpu", dtype=torch.float64)
        with torch.no_grad():
            rendered = engine.render(poses)
        valid = rendered.ray_valid.numpy().copy()
        unions = {name: getattr(rendered, name + "_union").numpy().copy() for name in ARMS}
    metrics = {name: summarize_pixel_error(unions[name], valid, target) for name in ARMS}
    return unions, valid, metrics


def fit_arms(rig, initial, cfg, start, predicted, missing, rays, targets):
    """Exactly one common assignment; evaluation targets cannot enter this API."""
    plan = assignment_plans(start, rays, cfg.ray_sigma_mm, cfg.missing_cost, keep=1)[0][1]
    assert_exclusive(plan)
    results, iteration_sources, iteration_observed = [], [], []
    for arm in ARMS:
        result = fit_local(rig, initial, cfg, start.copy(), plan.copy(), predicted.copy(),
                           missing.copy(), rays, targets, arm)
        support = validate_fit(result, start, plan, cfg, rays, rig)
        results.append(result)
        iteration_sources.append(support[0])
        iteration_observed.append(support[1])
    return plan, results, np.stack(iteration_sources), np.stack(iteration_observed)


def screening(frames, observed, mean_losses):
    """Mean loss axes frame, fitted arm, scoring model; retain every scene-slot loss."""
    observed, losses = np.asarray(observed), np.asarray(mean_losses)
    require(observed.shape == (len(frames), 3, 3, 3) and observed.dtype == bool
            and losses.shape == (len(frames), 3, 3) and np.isfinite(losses).all(),
            "screening shape or finite data mismatch")
    recovery = np.isin(frames, (82719, 82720))
    rows = {}
    for a, arm in enumerate(ARMS):
        lost_pect = observed[:, 0, :, 1] & ~observed[:, a, :, 1]
        lost_mouth = observed[:, 0, :, 0] & ~observed[:, a, :, 0] & recovery[:, None]
        better = bool(losses[:, a, a].mean() < losses[:, 0, 0].mean() - 1e-10)
        better_same = bool(losses[:, a, a].mean() < losses[:, 0, a].mean() - 1e-10)
        records = lambda flags: [  # noqa: E731
            dict(frame_idx=int(frames[t]), slot=int(f)) for t, f in np.argwhere(flags)
        ]
        rows[arm] = dict(
            mean_chosen_mask_error=float(losses[:, a, a].mean()),
            mean_common_direct_error=float(losses[:, a, 0].mean()),
            cross_model_mean_errors=losses[:, a].mean(axis=0).tolist(),
            average_chosen_error_lower_than_direct=better,
            same_model_error_lower_than_direct_arm_pose=better_same,
            gate_error_tolerance=1e-10,
            lost_chest_scene_slots=records(lost_pect),
            lost_recovery_mouth_scene_slots=records(lost_mouth),
            gate_pass=bool(a != 0 and better and better_same
                           and not lost_pect.any() and not lost_mouth.any()),
            worse_chosen_mask_frames=[int(frames[t]) for t in np.flatnonzero(
                losses[:, a, a] > losses[:, 0, 0])],
            worse_common_direct_frames=[int(frames[t]) for t in np.flatnonzero(
                losses[:, a, 0] > losses[:, 0, 0])],
        )
    return rows


def run_fits(verified, cfg, pixels_root, output, record):
    started = time.perf_counter()
    original = identity_inputs._npz(verified.root / "candidate.npz")
    require(set(original) == identity_inputs.GEOMETRY_ARRAYS, "persistent 21-array schema required")
    identity_inputs.persistent_runner.validate_arm_arrays(original, verified.frames)
    parents = replay_chosen_parents(original, verified.initial)
    rig = load_rig(verified.root / "calibration_snapshot")
    require(tuple(rig.views) == tuple(VIEWS), "calibration view order differs")
    nodes = CachedNodes(verified.input_root, verified.manifest["experiment"], verified.frames)
    packed = {v: identity_inputs._npz(verified.sequence / verified.manifest["masks"][v]["path"])
              for v in VIEWS}
    for data in packed.values():
        require(np.array_equal(data["frame_idx"], verified.frames), "mask frame order differs")
    all_observed, all_losses, rows = [], [], []

    def save(name, arrays):
        path = output / name
        np.savez_compressed(path, **arrays)
        record.output(name, path, sha256=sha256(path))

    for frame in FRAMES:
        t = int(frame - verified.frames[0])
        start = original["pose_xyz"][t].copy()
        points = {v: nodes.at(v, frame) for v in VIEWS}
        rays = RayObservations.from_points(rig, points)
        old_source, old_observed, _ = supported_nodes(start, original["sources"][t], rays, rig)
        np.testing.assert_array_equal(old_source, original["sources"][t])
        np.testing.assert_array_equal(old_observed, original["observed"][t])
        masks = {}
        for view, data in packed.items():
            decoded = decode_frame_masks(data, t)
            require(array_sha256(decoded) == data["frame_mask_sha256"][t].decode(),
                    "decoded independent mask SHA differs")
            masks[view] = decoded if data["view_available"][t] else None
            require(masks[view] is not None or len(decoded) == 0, "missing mask view has instances")
        dummy = np.zeros((1200, 1200, 3), np.uint8)  # native shape only, no raw color input
        observation = FrameObservation(frame, points, dict.fromkeys(VIEWS, dummy), masks, {})
        fit = sample_pixels(observation, "detectron2", 384, 384, 256)
        evaluation = sample_body_evaluation(observation, fit.pixels)
        pixels = check_saved_pixels(pixels_root / f"f{frame}_pixels.npz", fit, evaluation, VIEWS)
        save(f"f{frame}_pixels.npz", dict(frame_idx=np.array(frame), **pixels))
        plan, results, iteration_sources, iteration_observed = fit_arms(
            rig, verified.initial, cfg, start, parents["predicted_pose_xyz"][t],
            parents["parent_missing"][t], rays, fit)
        arrays = {key: np.stack([r[key] for r in results]) for key in (
            "pose_xyz", "sources", "observed", "score", "loss_parts", "iteration_scores",
            "iteration_loss_parts", "iteration_pose_xyz", "boundary_coordinates", "best_step")}
        pose = arrays["pose_xyz"]
        arrays.update(
            frame_idx=np.array(frame), arms=np.asarray(ARMS), fixed_plan=plan,
            input_pose_xyz=start, predicted_pose_xyz=parents["predicted_pose_xyz"][t],
            parent_missing=parents["parent_missing"][t],
            parent_velocity=parents["parent_velocity"][t],
            original_sources=old_source, original_observed=old_observed,
            original_identity_review=original["identity_review"][t],
            iteration_sources=iteration_sources, iteration_observed=iteration_observed,
            body_segment_lengths_mm=np.linalg.norm(np.diff(pose, axis=2), axis=-1),
            chest_pair_distances_mm=np.stack([
                np.linalg.norm(pose[:, i, 1] - pose[:, j, 1], axis=-1)
                for i, j in ((0, 1), (0, 2), (1, 2))], axis=-1),
            predicted_chest_displacement_mm=pose[:, :, 1] - parents["predicted_pose_xyz"][t, :, 1],
        )
        unions, valid, metrics = evaluate_poses(rig, cfg, pose, evaluation)
        for scorer in ARMS:
            arrays["eval_" + scorer + "_union"] = unions[scorer]
            for key, value in metrics[scorer].items():
                arrays["eval_" + scorer + "_" + key] = value
        arrays["eval_calibrated_valid"] = valid
        means = np.stack([metrics[name]["mean_view_loss"] for name in ARMS], axis=-1)
        all_losses.append(means)
        all_observed.append(arrays["observed"])
        save(f"f{frame}_fits.npz", arrays)
        rows.append(dict(frame_idx=frame, mean_errors_fit_arm_by_scoring_model=means.tolist(),
                         observed=arrays["observed"].tolist(), scores=arrays["score"].tolist(),
                         loss_parts=arrays["loss_parts"].tolist(),
                         evaluation={name: {k: v.tolist() for k, v in metrics[name].items()}
                                     for name in ARMS}))
        print(json.dumps(dict(frame_idx=frame, completed_arms=3)), flush=True)
    gates = screening(np.asarray(FRAMES), np.stack(all_observed), np.stack(all_losses))
    summary = dict(schema="reflection_local_probe_v1", frames=list(FRAMES), arms=list(ARMS),
                   scoring_models=list(ARMS), scene_fits=39, loss_part_order=[
                       "nodes", "shape", "center_motion", "repulsion", "mask"],
                   rows=rows, screening=gates, elapsed_seconds=time.perf_counter() - started,
                   input_geometry_modified=False, tracking=False, identity_updated=False,
                   GT_read=False, native_colors_read=False,
                   gate_formula="For a=unit/fresnel: L[a,a]<L[direct,direct]-1e-10 AND "
                   "L[a,a]<L[direct,a]-1e-10, with no per-slot chest/recovery-mouth losses.",
                   scope="Opened development frames; fit-excluded pixels use the same D2 labels. "
                   "Passing these gates does not validate identity or physical geometry.")
    path = output / "summary.json"
    path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    record.output("summary", path, sha256=sha256(path))
    lines = ["Fixed 13-frame CPU local reflection comparison.", "",
             "No tracking, relabeling, acceptance updates or GT. The lower bound -6 mm remains "
             "an engineering constraint; optical floor 0 mm and walls are nominal hypotheses.", "",
             "Each arm's chosen-model error and the common direct-only error are both retained. "
             "A changed scoring definition is not evidence of better three-body geometry.", "",
             "Error gates require L[a,a] < L[direct,direct] - 1e-10 and "
             "L[a,a] < L[direct,a] - 1e-10 before the per-slot support gates.", "",
             "| Arm | Chosen mean | Common direct mean | Lost chest | Lost recovery mouth | Gate |",
             "|---|---:|---:|---:|---:|---|"]
    for arm, item in gates.items():
        lines.append(f"| {arm} | {item['mean_chosen_mask_error']:.9f} | "
                     f"{item['mean_common_direct_error']:.9f} | "
                     f"{len(item['lost_chest_scene_slots'])} | "
                     f"{len(item['lost_recovery_mouth_scene_slots'])} | {item['gate_pass']} |")
    lines += ["", "Every frame, including failures and 79372/82719/82720:", "",
              "| Frame | Arm | Chosen minus direct-arm error | Common-direct difference |",
              "|---|---|---:|---:|"]
    for frame, values in zip(FRAMES, all_losses, strict=True):
        for a, arm in enumerate(ARMS):
            lines.append(f"| {frame} | {arm} | {values[a,a]-values[0,0]:+.9f} | "
                         f"{values[a,0]-values[0,0]:+.9f} |")
    lines += ["", "All per-view/stratum results, source support and iteration histories are saved "
              "in summary.json and f*_fits.npz. Failed arms remain reported; no gate was tuned."]
    path = output / "RESULT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    record.output("report", path, sha256=sha256(path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=REPO /
                        "work/three_body/persistent_mouth_continuous_one_minute_v1")
    parser.add_argument("--pixels-run", type=Path, default=REPO /
                        "work/three_body/curve_body_spatial_13frames_v1")
    parser.add_argument("--renderer-audit", type=Path, default=REPO /
                        "work/three_body/reflection_renderer_saved_parity_v1")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register-only", action="store_true")
    mode.add_argument("--registration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    require(output.is_relative_to(REPO) and not output.exists(), "new workspace output required")
    code = source_hashes()  # Captured before any verification, candidate remains unread.
    verified, cfg, pixels_root, hashes = verify_evidence(args, code)
    plan = dict(schema="reflection_local_probe_registration_v1", frames=list(FRAMES),
                arms=list(ARMS), scoring_models=list(ARMS), protocol_sha256=PROTOCOL_SHA,
                run=str(verified.root), pixels_run=str(pixels_root), code_hashes=code,
                input_hashes=dict(hashes), tracker_config=asdict(cfg), threads=2,
                dtype="float64", direct_mode="unbounded", initializations=0, GT=False,
                gate_formula="L[a,a]<L[0,0]-1e-10 and L[a,a]<L[0,a]-1e-10; "
                "no lost chest at any scene-slot and no lost mouth at 82719/82720",
                runtime=dict(python=sys.version, numpy=np.__version__,
                             torch=str(torch.__version__)))
    if args.registration:
        verify_registration(args.registration, plan, hashes)
    old_threads = torch.get_num_threads()
    output.mkdir(parents=True, exist_ok=False)
    name = "registration" if args.register_only else "probe"
    try:
        torch.set_num_threads(2)
        with provenance("reflection_local_" + name, output, name=name) as rec:
            for path, digest in {**hashes, **code}.items():
                rec.input(path, path, sha256=digest)
            for path, digest in code.items():
                target = output / "code_snapshot" / Path(path).relative_to(REPO)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                require(sha256(target) == digest, "source snapshot mismatch")
                rec.output(target.relative_to(output).as_posix(), target, sha256=digest)
            path = output / "plan.json"
            path.write_text(json.dumps(plan, indent=2, allow_nan=False), encoding="utf-8")
            rec.output("plan", path, sha256=sha256(path))
            unchanged({**hashes, **code}, rec.outputs)
            if not args.register_only:
                run_fits(verified, cfg, pixels_root, output, rec)
            unchanged({**hashes, **code}, rec.outputs)
    finally:
        torch.set_num_threads(old_threads)
    print(json.dumps(dict(output=str(output), status="ok", mode=name)))


if __name__ == "__main__":
    main()
