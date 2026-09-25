"""Registered single-frame CPU/CUDA qualification; no temporal tracking or GT."""

import argparse
import dataclasses
import json
import platform
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import scipy
import torch

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "labs/20260909_three_body"
sys.path[:0] = [str(LAB), str(REPO / "src")]
import identity_inputs  # noqa: E402
from node_geometry import RayObservations, assigned_rays  # noqa: E402
from observations import CachedNodes, PixelTargets  # noqa: E402
from prepare_sequence import VIEWS, sha256  # noqa: E402
from reflection_renderer import ReflectionFishRenderer  # noqa: E402
from reflection_tracker import ReflectionPersistentNodeTracker  # noqa: E402
from renderer import RendererConfig  # noqa: E402
from tracker import Proposal, TrackerConfig  # noqa: E402

from medaka3d.calib import load_rig  # noqa: E402
from medaka3d.provenance import provenance  # noqa: E402

WORK = REPO / "work/three_body"
PRODUCER = WORK / "reflection_local_probe_13frames_v1"
AUDIT = WORK / "reflection_local_probe_independent_audit_v1"
PROTOCOL = WORK / "REFLECTION_TEMPORAL_PLAN.md"
FRAME, ARMS = 79381, ("direct", "unit", "fresnel")
TRACKER_SHA = "a6512910ea9fee3395fc33e30e0d1c1d7dd24565f40bfd37444c70751709d8df"
TOLERANCES = dict(
    response=1e-9,
    gradient=1e-9,
    optical_weight=1e-12,
    fit_pose_mm=1e-8,
    fit_score=1e-9,
    fit_parts=1e-9,
    flags="exact",
)
FLOAT_FIELDS = (
    "direct_alpha",
    "inner_alpha",
    "outer_alpha",
    "direct_union",
    "unit_union",
    "fresnel_union",
)
BOOL_FIELDS = ("ray_valid", "direct_valid", "inner_valid", "outer_valid")
WEIGHT_FIELDS = ("inner_weight", "outer_weight")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bind(path, expected, hashes):
    path = Path(path).resolve(strict=True)
    require(path.is_relative_to(REPO), "evidence must remain inside the workspace")
    actual = hashes.get(str(path)) or sha256(path)
    require(expected is None or actual == expected, "input SHA differs: " + str(path))
    hashes[str(path)] = actual
    return path


def unchanged(hashes):
    for path, expected in hashes.items():
        require(sha256(path) == expected, "input/source/snapshot/output changed: " + path)


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
    values = {str(path): sha256(path) for path in sorted(paths)}
    require(values[str(LAB / "reflection_tracker.py")] == TRACKER_SHA, "unqualified tracker source")
    return values


def runtime(device_name):
    # cuda build metadata is static: this function does not initialize CUDA.
    return dict(
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        opencv=cv2.__version__,
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        cuda_device_name=device_name,
        torch_cpu_threads=2,
    )


def verified_seal(root, name, stage, hashes):
    root = Path(root).resolve(strict=True)
    path = bind(root / name, None, hashes)
    seal = read(path)
    require(
        seal.get("status") == "ok" and seal.get("stage") == stage,
        "successful producer seal required",
    )
    outputs = {}
    for item in seal["outputs"].values():
        target = Path(item["path"]).resolve(strict=True)
        require(target.is_relative_to(root), "sealed output escaped its root")
        bind(target, item["sha256"], hashes)
        require(item.get("size") == target.stat().st_size, "sealed output size differs")
        outputs[str(target)] = item["sha256"]
    return seal, outputs


def load_inputs():
    hashes = source_hashes()  # Freeze live code before invoking any input verifier.
    code = hashes.copy()
    seal, outputs = verified_seal(
        PRODUCER, "probe.provenance.json", "reflection_local_probe", hashes
    )
    require(
        all(
            str(PRODUCER / name) in outputs
            for name in ("plan.json", "summary.json", f"f{FRAME}_fits.npz", f"f{FRAME}_pixels.npz")
        ),
        "local context is not bound in producer outputs",
    )
    local_plan = read(PRODUCER / "plan.json")
    require(
        local_plan["arms"] == list(ARMS) and local_plan["direct_mode"] == "unbounded",
        "saved local arm/direct-mode contract differs",
    )
    for path, digest in local_plan["code_hashes"].items():
        source = bind(path, digest, hashes)
        item = seal["inputs"].get(str(source), {})
        require(
            item.get("sha256") == digest and Path(item.get("path", "")).resolve() == source,
            "local source input binding differs",
        )
        copied = PRODUCER / "code_snapshot" / source.relative_to(REPO)
        require(outputs.get(str(copied)) == digest, "local source snapshot differs")
    verified = identity_inputs._verify_run(Path(local_plan["run"]), appearance=False)
    for path, digest in verified.hashes.items():
        item = seal["inputs"].get(path, {})
        require(
            item.get("sha256") == digest and Path(item.get("path", "")).resolve() == Path(path),
            "geometry/calibration/SL/seed binding differs from local experiment",
        )
        bind(path, digest, hashes)
    audit, _ = verified_seal(
        AUDIT, "audit.provenance.json", "independent_reflection_local_output_audit", hashes
    )
    for path in (
        PRODUCER / "probe.provenance.json",
        PRODUCER / f"f{FRAME}_fits.npz",
        PRODUCER / f"f{FRAME}_pixels.npz",
    ):
        require(
            any(
                Path(item["path"]).resolve() == path and item.get("sha256") == hashes[str(path)]
                for item in audit["inputs"].values()
            ),
            "successful audit does not bind chosen context",
        )
    for item in audit["inputs"].values():
        bind(item["path"], item["sha256"], hashes)
    cfg = TrackerConfig(**local_plan["tracker_config"])
    require(
        cfg.device == "cpu" and cfg.opt_steps == 14 and cfg.learning_rate_mm == 0.25,
        "fixed fourteen-update local configuration required",
    )
    expected_cfg = dataclasses.replace(
        TrackerConfig(**verified.params["tracker_config"]), device="cpu"
    )
    require(
        dataclasses.asdict(cfg) == dataclasses.asdict(expected_cfg),
        "local numerical configuration changed",
    )
    with np.load(PRODUCER / f"f{FRAME}_fits.npz", allow_pickle=False) as archive:
        saved = {
            key: archive[key].copy()
            for key in (
                "frame_idx",
                "arms",
                "pose_xyz",
                "fixed_plan",
                "predicted_pose_xyz",
                "parent_missing",
            )
        }
    with np.load(PRODUCER / f"f{FRAME}_pixels.npz", allow_pickle=False) as archive:
        pixels = {
            key: archive[key].copy() for key in ("frame_idx", "fit_uv", "fit_target", "fit_weights")
        }
    validate_context(saved, pixels)
    rig = load_rig(verified.root / "calibration_snapshot")
    nodes = CachedNodes(
        verified.input_root, verified.manifest["experiment"], np.array([FRAME], np.int64)
    )
    points = {view: nodes.at(view, FRAME).copy() for view in VIEWS}
    for view in VIEWS:
        require(str(nodes.paths[view].resolve()) in hashes, "actual SLEAP cache is not bound")
    rays = RayObservations.from_points(rig, points)
    assigned_rays(rays, saved["fixed_plan"][None])  # Reject invalid row indexes before execution.
    versions = verified.params["package_versions"]
    software = runtime(versions["cuda_device_name"])
    require(
        all(software[key] == value for key, value in versions.items()),
        "actual software differs from original geometry",
    )
    require(set(source_hashes()) == set(code), "late local source imports are not registered")
    return SimpleNamespace(
        hashes=hashes,
        code=code,
        saved=saved,
        pixels=pixels,
        rig=rig,
        initial=verified.initial.copy(),
        rays=rays,
        cfg=cfg,
        runtime=software,
        points=points,
        calibration_root=verified.root / "calibration_snapshot",
    )


def validate_context(saved, pixels):
    require(
        int(saved["frame_idx"]) == int(pixels["frame_idx"]) == FRAME, "fixed native frame required"
    )
    require(saved["arms"].tolist() == list(ARMS), "fixed start-arm order required")
    for key, shape in (("pose_xyz", (3, 3, 3, 3)), ("predicted_pose_xyz", (3, 3, 3))):
        value = saved[key]
        require(
            value.dtype == np.float64 and value.shape == shape and np.isfinite(value).all(),
            "finite float64 saved geometry required: " + key,
        )
    plan = saved["fixed_plan"]
    require(
        plan.shape == (3, 3, 3) and plan.dtype == np.int32 and np.all(plan >= -1),
        "fixed plan schema differs",
    )
    for v in range(3):
        for n in range(3):
            rows = plan[:, v, n][plan[:, v, n] >= 0]
            require(len(rows) == len(np.unique(rows)), "fixed plan reuses a same-view node row")
    require(
        saved["parent_missing"].shape == (3,)
        and saved["parent_missing"].dtype == np.int64
        and np.all(saved["parent_missing"] >= 0),
        "parent gap schema differs",
    )
    require(
        pixels["fit_uv"].shape == (3, 1024, 2)
        and pixels["fit_uv"].dtype == np.float64
        and np.isfinite(pixels["fit_uv"]).all(),
        "fixed 1024 native pixels per view required",
    )
    for key in ("fit_target", "fit_weights"):
        x = pixels[key]
        require(
            x.shape == (3, 1024)
            and x.dtype == np.float32
            and np.isfinite(x).all()
            and np.all(x >= 0),
            "pixel target/weight contract differs",
        )
    require(np.all(pixels["fit_target"] <= 1), "mask target outside [0,1]")


def make_targets(context):
    p = context.pixels
    return PixelTargets(
        dict(zip(VIEWS, p["fit_uv"], strict=True)),
        p["fit_target"].copy(),
        p["fit_weights"].copy(),
        p["fit_target"].astype(bool),
    )


def evaluate_device(context, device):
    target = make_targets(context)
    renderer = ReflectionFishRenderer(
        context.rig,
        target.pixels,
        RendererConfig(width_ratio=context.cfg.width_ratio),
        direct_mode="unbounded",
        device=device,
        dtype=torch.float64,
    )
    outputs = {}
    for arm in ARMS:
        pose = torch.tensor(
            context.saved["pose_xyz"], device=device, dtype=torch.float64, requires_grad=True
        )
        rendered = renderer.render(pose)
        truth = torch.tensor(target.target, device=device, dtype=torch.float64)
        weights = (
            torch.tensor(target.weights, device=device, dtype=torch.float64) * rendered.ray_valid
        )
        union = getattr(rendered, arm + "_union")
        # Mean across B=3 and V=3 of the original fixed-weight pixel sums.
        loss = ((union - truth).square() * weights).sum(dim=2).mean()
        (gradient,) = torch.autograd.grad(loss, pose)
        outputs["gradient_" + arm] = gradient.detach().cpu().numpy().copy()
        outputs["loss_" + arm] = loss.detach().cpu().numpy().copy()
        if arm == ARMS[0]:
            for key in (*FLOAT_FIELDS, *BOOL_FIELDS, *WEIGHT_FIELDS):
                outputs[key] = getattr(rendered, key).detach().cpu().numpy().copy()
    tracker = ReflectionPersistentNodeTracker(
        context.rig,
        context.initial.copy(),
        dataclasses.replace(context.cfg, device=device),
        mask_arm="fresnel",
    )
    proposals = [
        Proposal(
            p.copy(),
            context.saved["predicted_pose_xyz"].copy(),
            context.saved["fixed_plan"].copy(),
            0,
            "fixed_qualification",
        )
        for p in context.saved["pose_xyz"]
    ]
    states = [SimpleNamespace(missing=context.saved["parent_missing"].copy())]
    fitted = tracker._fit(proposals, states, context.rays, target)
    require(tracker.initialize_count == 0, "qualification unexpectedly initialized tracking")
    for key, value in zip(
        ("fit_pose_xyz", "fit_score", "fit_parts", "fit_direct_mask_support"), fitted, strict=True
    ):
        outputs[key] = value.copy()
    outputs["initial_lengths_mm"] = tracker.lengths.copy()
    return outputs


def compare_outputs(cpu, cuda):
    expected = {
        *FLOAT_FIELDS,
        *BOOL_FIELDS,
        *WEIGHT_FIELDS,
        *("gradient_" + arm for arm in ARMS),
        *("loss_" + arm for arm in ARMS),
        "fit_pose_xyz",
        "fit_score",
        "fit_parts",
        "fit_direct_mask_support",
        "initial_lengths_mm",
    }
    require(set(cpu) == set(cuda) == expected, "CPU/CUDA required output fields differ")
    errors, passed = {}, True
    for key in cpu:
        a, b = cpu[key], cuda[key]
        require(a.shape == b.shape and a.dtype == b.dtype, "CPU/CUDA dtype/shape differ: " + key)
        boolean = key in (*BOOL_FIELDS, "fit_direct_mask_support")
        require(
            a.dtype == (np.dtype(bool) if boolean else np.dtype(np.float64)),
            "qualification field dtype differs: " + key,
        )
        if boolean:
            equal = np.array_equal(a, b)
            errors[key] = dict(exact=bool(equal), mismatched_elements=int(np.count_nonzero(a != b)))
            passed &= bool(equal)
            continue
        require(np.isfinite(a).all() and np.isfinite(b).all(), "nonfinite qualification output")
        tolerance = (
            TOLERANCES["fit_pose_mm"]
            if key == "fit_pose_xyz"
            else TOLERANCES["fit_score"]
            if key == "fit_score"
            else TOLERANCES["fit_parts"]
            if key == "fit_parts"
            else TOLERANCES["optical_weight"]
            if key in WEIGHT_FIELDS
            else TOLERANCES["gradient"]
            if key.startswith("gradient_")
            else TOLERANCES["response"]
        )
        maximum = float(np.max(np.abs(a - b), initial=0))
        errors[key] = dict(
            maximum_absolute_difference=maximum, tolerance=tolerance, passed=maximum <= tolerance
        )
        passed &= maximum <= tolerance
    return bool(passed), errors


def plan_for(context):
    return dict(
        schema="reflection_CPU_CUDA_qualification_plan_v1",
        frame_idx=FRAME,
        batch=3,
        start_arms=list(ARMS),
        tolerances=TOLERANCES,
        code_hashes=context.code,
        input_hashes=context.hashes,
        runtime=context.runtime,
        tracker_config=dataclasses.asdict(context.cfg),
        direct_mode="unbounded",
        fitted_mask_arm="fresnel",
        opt_steps=14,
        source_run=str(PRODUCER),
        local_audit=str(AUDIT),
        calibration_root=str(context.calibration_root),
        gradient_formula="mean_B,V(sum_N(weight * ray_valid * (union-target)^2)); all 3 unions",
        no_temporal_tracking=True,
        no_new_image_decode=True,
        GT=False,
        cuda_device_verified_at_registration=False,
    )


def fresh(root):
    root = Path(root).resolve()
    require(not root.exists() and root.is_relative_to(WORK), "new workspace output required")
    root.mkdir(parents=True, exist_ok=False)
    return root


def save(record, output, name, data, outputs):
    path = output / name
    if isinstance(data, dict) and data and all(isinstance(v, np.ndarray) for v in data.values()):
        with path.open("xb") as stream:
            np.savez_compressed(stream, **data)
    else:
        path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    outputs[str(path)] = sha256(path)
    record.output(name, path, sha256=outputs[str(path)])


def snapshots(record, output, code, outputs):
    for source, digest in code.items():
        target = output / "code_snapshot" / Path(source).relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        outputs[str(target)] = digest
        record.output(str(target), target, sha256=digest)


def verify_registration(root, expected_sha, plan, hashes):
    path = Path(root).resolve(strict=True) / "plan.json"
    require(sha256(path) == expected_sha, "explicitly registered plan SHA differs")
    seal, outputs = verified_seal(
        path.parent,
        "registration.provenance.json",
        "register_reflection_CPU_CUDA_qualification",
        hashes,
    )
    require(str(path) in outputs and read(path) == plan, "registration conditions changed")
    require(
        {key: item["sha256"] for key, item in seal["inputs"].items()} == plan["input_hashes"],
        "registration exact input membership differs",
    )
    for key, item in seal["inputs"].items():
        require(Path(item["path"]).resolve() == Path(key), "registered input path differs")
        bind(key, item["sha256"], hashes)
    return seal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--register", action="store_true")
    group.add_argument("--execute", action="store_true")
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.execute and (args.output is None or args.expected_plan_sha256 is None):
        parser.error("execute requires --output and --expected-plan-sha256")
    context = load_inputs()
    plan = plan_for(context)
    hashes = context.hashes.copy()
    registration = None
    if args.execute:
        registration = verify_registration(args.plan_root, args.expected_plan_sha256, plan, hashes)
    output = fresh(args.plan_root if args.register else args.output)
    stage = (
        "register_reflection_CPU_CUDA_qualification"
        if args.register
        else "reflection_CPU_CUDA_qualification"
    )
    with provenance(
        stage, output, name="registration" if args.register else "qualification"
    ) as record:
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
        outputs = {}
        snapshots(record, output, context.code, outputs)
        if args.register:
            save(record, output, "plan.json", plan, outputs)
        else:
            require(
                datetime.fromisoformat(registration["finished_at"])
                < datetime.fromisoformat(record.started_at),
                "registration must precede qualification",
            )
            require(torch.cuda.is_available(), "CUDA qualification requires its assigned GPU slot")
            require(
                torch.cuda.get_device_name(0) == context.runtime["cuda_device_name"],
                "GPU differs from fixed runtime",
            )
            previous_threads = torch.get_num_threads()
            torch.set_num_threads(2)
            torch.cuda.reset_peak_memory_stats()
            times, arrays = {}, {}
            try:
                for device in ("cpu", "cuda"):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    values = evaluate_device(context, device)
                    torch.cuda.synchronize()
                    times[device] = time.perf_counter() - started
                    arrays.update({device + "__" + key: value for key, value in values.items()})
                cpu = {
                    k.removeprefix("cpu__"): v for k, v in arrays.items() if k.startswith("cpu__")
                }
                cuda = {
                    k.removeprefix("cuda__"): v for k, v in arrays.items() if k.startswith("cuda__")
                }
                passed, errors = compare_outputs(cpu, cuda)
            finally:
                torch.set_num_threads(previous_threads)
            arrays.update({"context__" + k: v for k, v in context.saved.items()})
            arrays.update({"pixels__" + k: v for k, v in context.pixels.items()})
            arrays["initial_pose_xyz"] = context.initial.copy()
            for v, view in enumerate(VIEWS):
                arrays[view + "__points"] = context.points[view].copy()
                arrays[view + "__ray_origins"] = context.rays.origins[v].copy()
                arrays[view + "__ray_directions"] = context.rays.directions[v].copy()
            save(record, output, "arrays.npz", arrays, outputs)
            summary = dict(
                schema="reflection_CPU_CUDA_qualification_v1",
                passed=passed,
                frame_idx=FRAME,
                batch=3,
                start_arms=list(ARMS),
                tolerances=TOLERANCES,
                errors=errors,
                runtime=context.runtime,
                code_hashes=context.code,
                input_hashes=context.hashes,
                tracker_config=dataclasses.asdict(context.cfg),
                direct_mode="unbounded",
                fitted_mask_arm="fresnel",
                opt_steps=14,
                times_seconds=times,
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                gradient_formula=plan["gradient_formula"],
                no_temporal_tracking=True,
                no_new_image_decode=True,
                GT=False,
                cuda_device_verified=True,
                limit="Fixed one-frame numerical parity, not tracking quality or all poses/devices",
            )
            save(record, output, "summary.json", summary, outputs)
            record.param("passed", passed)
        record.param("plan", plan)
        unchanged({**hashes, **outputs})
        record.param("all_input_source_snapshot_output_sha_unchanged", True)
    print(
        json.dumps(
            dict(
                status="registered" if args.register else "complete",
                output=str(output),
                plan_sha256=sha256(output / "plan.json") if args.register else None,
            )
        )
    )


if __name__ == "__main__":
    main()
