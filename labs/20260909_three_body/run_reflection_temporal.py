"""Registered 200-frame reflection comparison; seed extraction precedes inference.

Registration alone may decompress the previous minute's full pose_xyz NPZ field
to extract its one optimistic frame-79236 seed. Execution never deserializes that
candidate: it reads the separately sealed seed and independent native inputs.
Both trajectories are saved before any fit-excluded pixel evaluation begins.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import scipy
import torch

LAB = Path(__file__).resolve().parent
REPO = LAB.parents[1]
sys.path[:0] = [str(LAB), str(REPO / "src")]

import identity_inputs  # noqa: E402
import run_experiment as base  # noqa: E402
import run_persistent_node_recovery as persistent_runner  # noqa: E402
from body_pixel_evaluation import sample_body_evaluation, summarize_pixel_error  # noqa: E402
from motion import FlowConfig  # noqa: E402
from node_geometry import RayObservations, supported_nodes  # noqa: E402
from node_recovery import NodeRecoveryConfig  # noqa: E402
from observations import FrameObservation, SequenceReader, sample_pixels  # noqa: E402
from persistent_node_tracker import (  # noqa: E402
    PersistentNodeConfig,
    PersistentNodeThreeBodyTracker,
)
from prepare_sequence import array_sha256, decode_frame_masks, sha256  # noqa: E402
from reflection_renderer import ReflectionFishRenderer  # noqa: E402
from reflection_tracker import ReflectionPersistentNodeTracker  # noqa: E402
from renderer import RendererConfig  # noqa: E402
from resumable_observations import iter_from  # noqa: E402
from source_sequence import SourceSequenceReader  # noqa: E402
from tracker import TrackerConfig  # noqa: E402

from medaka3d.calib import load_rig  # noqa: E402
from medaka3d.provenance import provenance  # noqa: E402
from medaka3d.types import VIEWS  # noqa: E402

START, STOP = 79236, 79436
FRAMES = np.arange(START, STOP, dtype=np.int64)
ARMS = ("direct", "fresnel")
ARRAYS = persistent_runner.ORIGINAL_ARRAYS | {"node_missing", "direct_pectoral_age"}
PROTOCOL = REPO / "work/three_body/REFLECTION_TEMPORAL_PLAN.md"
PROTOCOL_SHA = "212acf26cf1c8300588a7dcad26b0a4f85b50fb9d8509dd843eccaca28911d1d"
TRACKER_SHA = "a6512910ea9fee3395fc33e30e0d1c1d7dd24565f40bfd37444c70751709d8df"
LOCAL_AUDIT = REPO / "work/three_body/reflection_local_probe_independent_audit_v1"
LOCAL_AUDIT_SHA = "1615d69f149a2a19eb79f18f9542692f728e658013ed355016b2d2f54c3b69ad"
QUALIFICATION_SOURCE = REPO / "work/three_body/qualify_reflection_gpu.py"
QUALIFICATION_SHA = "9698aa9811d9a8e50032cf98aea1bc0af3eefc100534c286f9870b4e36e9d6b6"
QUALIFICATION_TOLERANCES = dict(
    response=1e-9,
    gradient=1e-9,
    optical_weight=1e-12,
    fit_pose_mm=1e-8,
    fit_score=1e-9,
    fit_parts=1e-9,
    flags="exact",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def equal(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(
        b, sort_keys=True, allow_nan=False
    )


def workspace(value):
    path = Path(value)
    path = (path if path.is_absolute() else REPO / path).resolve(strict=True)
    require(path.is_relative_to(REPO), "evidence must remain in workspace")
    return path


def bind(path, digest, bindings):
    path = workspace(path)
    actual = sha256(path)
    require(digest is None or actual == digest, f"SHA differs: {path}")
    require(bindings.get(str(path), actual) == actual, "conflicting evidence hash")
    bindings[str(path)] = actual
    return path


def verify_files(bindings):
    for path, digest in bindings.items():
        require(sha256(path) == digest, f"input/code/output changed: {path}")


def verify_outputs(root, filename, stage, bindings):
    root = workspace(root)
    seal = read_json(bind(root / filename, None, bindings))
    require(seal["status"] == "ok" and seal["stage"] == stage, "successful matching seal required")
    for entry in seal["outputs"].values():
        path = workspace(entry["path"])
        require(path.is_relative_to(root), "foreign sealed output")
        bind(path, entry["sha256"], bindings)
    return seal


def output_path(seal, role, expected):
    require(
        role in seal["outputs"] and workspace(seal["outputs"][role]["path"]) == expected,
        "required output role missing or mismatched",
    )
    return expected


def code_hashes():
    paths = {Path(__file__).resolve(), PROTOCOL}
    for module in tuple(sys.modules.values()):
        value = getattr(module, "__file__", None)
        if value:
            path = Path(value).resolve()
            if path.suffix == ".py" and (
                path.is_relative_to(LAB) or path.is_relative_to(REPO / "src")
            ):
                paths.add(path)
    result = {str(path): sha256(path) for path in sorted(paths)}
    require(result[str(PROTOCOL)] == PROTOCOL_SHA, "fixed temporal plan differs")
    require(result[str(LAB / "reflection_tracker.py")] == TRACKER_SHA, "qualified tracker differs")
    return result


def software_runtime():
    """No CUDA context/device initialization, including during CPU preflight."""
    return dict(
        python=sys.version.split()[0],
        numpy=np.__version__,
        scipy=scipy.__version__,
        opencv=cv2.__version__,
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        torch_cpu_threads=2,
    )


def configs():
    return dict(
        tracker=asdict(TrackerConfig(device="cuda", flow_proposals=True)),
        recovery=asdict(NodeRecoveryConfig()),
        persistence=asdict(PersistentNodeConfig()),
        flow=asdict(FlowConfig()),
    )


def validate_qualification_errors(errors):
    """Check the complete producer diagnostics, without re-running any fit."""
    limits = {
        name: 1e-9
        for name in (
            "direct_alpha",
            "inner_alpha",
            "outer_alpha",
            "direct_union",
            "unit_union",
            "fresnel_union",
            "initial_lengths_mm",
            "fit_score",
            "fit_parts",
            *("gradient_" + arm for arm in ("direct", "unit", "fresnel")),
            *("loss_" + arm for arm in ("direct", "unit", "fresnel")),
        )
    }
    limits.update(inner_weight=1e-12, outer_weight=1e-12, fit_pose_xyz=1e-8)
    flags = {"ray_valid", "direct_valid", "inner_valid", "outer_valid", "fit_direct_mask_support"}
    require(set(errors) == set(limits) | flags, "qualification diagnostic fields differ")
    for name, tolerance in limits.items():
        item = errors[name]
        maximum = item["maximum_absolute_difference"]
        require(
            item["passed"] is True
            and item["tolerance"] == tolerance
            and np.isfinite(maximum)
            and 0 <= maximum <= tolerance,
            "qualification numerical comparison failed",
        )
    for name in flags:
        require(
            errors[name]["exact"] is True and errors[name]["mismatched_elements"] == 0,
            "qualification flag comparison failed",
        )


def verify_qualification(root, codes, bindings):
    root = workspace(root)
    seal = verify_outputs(
        root, "qualification.provenance.json", "reflection_CPU_CUDA_qualification", bindings
    )
    summary_path = output_path(seal, "summary.json", root / "summary.json")
    output_path(seal, "arrays.npz", root / "arrays.npz")
    summary = read_json(summary_path)
    require(
        summary["schema"] == "reflection_CPU_CUDA_qualification_v1"
        and summary["passed"] is True
        and summary["frame_idx"] == 79381
        and summary["batch"] == 3
        and summary["start_arms"] == ["direct", "unit", "fresnel"]
        and equal(summary["tolerances"], QUALIFICATION_TOLERANCES)
        and summary["direct_mode"] == "unbounded"
        and summary["fitted_mask_arm"] == "fresnel"
        and summary["opt_steps"] == 14
        and summary["GT"] is False
        and summary["no_temporal_tracking"] is True
        and summary["cuda_device_verified"] is True,
        "registered qualification is incomplete or failed",
    )
    validate_qualification_errors(summary["errors"])
    normalized = {str(workspace(path)): digest for path, digest in summary["code_hashes"].items()}
    require(
        normalized.get(str(QUALIFICATION_SOURCE)) == QUALIFICATION_SHA,
        "qualification producer source differs",
    )
    require(normalized.get(str(PROTOCOL)) == PROTOCOL_SHA, "qualification protocol differs")
    require(
        equal(summary["tracker_config"], {**configs()["tracker"], "device": "cpu"}),
        "qualification algorithm configuration differs",
    )
    registered = seal["params"]["plan"]
    for key in (
        "frame_idx",
        "batch",
        "start_arms",
        "tolerances",
        "code_hashes",
        "input_hashes",
        "runtime",
        "tracker_config",
        "direct_mode",
        "fitted_mask_arm",
        "opt_steps",
        "GT",
        "no_temporal_tracking",
    ):
        require(equal(registered[key], summary[key]), "qualification differs from its registration")
    require(
        seal["params"].get("all_input_source_snapshot_output_sha_unchanged") is True,
        "qualification final guard missing",
    )
    # Qualification uses its own earlier fixed-frame fixture; no coordinates are
    # deserialized here. Every recorded dependency and copied source is bound.
    for source_path, digest in summary["input_hashes"].items():
        require(
            source_path in seal["inputs"]
            and seal["inputs"][source_path]["sha256"] == digest
            and workspace(seal["inputs"][source_path]["path"]) == workspace(source_path),
            "qualification input binding differs",
        )
        bind(source_path, digest, bindings)
    for source_path, digest in normalized.items():
        require(
            summary["input_hashes"].get(source_path) == digest,
            "qualification source is not an input",
        )
        target = root / "code_snapshot" / Path(source_path).relative_to(REPO)
        matching = [
            entry for entry in seal["outputs"].values() if workspace(entry["path"]) == target
        ]
        require(
            len(matching) == 1 and matching[0]["sha256"] == digest,
            "qualification source snapshot unbound",
        )
    for name in (
        "reflection_renderer.py",
        "reflection_paths.py",
        "reflection_tracker.py",
        "reflection_local_fit.py",
        "renderer.py",
        "graph_fit.py",
        "tracker.py",
        "node_geometry.py",
        "persistent_node_tracker.py",
    ):
        source_path = str(LAB / name)
        require(
            normalized.get(source_path) == codes[source_path],
            f"qualification source mismatch: {name}",
        )
    for source_path, digest in normalized.items():
        bind(source_path, digest, bindings)
    local = software_runtime()
    require(
        all(summary["runtime"].get(key) == value for key, value in local.items()),
        "qualification software/runtime differs",
    )
    require(
        isinstance(summary["runtime"].get("cuda_device_name"), str), "qualification device required"
    )
    return dict(
        root=str(root),
        summary_sha256=sha256(summary_path),
        runtime=summary["runtime"],
        tolerances=summary["tolerances"],
        schema=summary["schema"],
    )


def prepare_registration(geometry_run, qualification_root, codes, *, preflight=False):
    """Only this registration path may deserialize the historical pose field."""
    verified = identity_inputs._verify_run(geometry_run, appearance=False)
    require(
        np.array_equal(verified.frames, np.arange(76620, 86580)), "complete parent minute required"
    )
    cfg = configs()
    for name, key in (
        ("tracker", "tracker_config"),
        ("recovery", "node_recovery_config"),
        ("persistence", "persistence_config"),
        ("flow", "flow_config"),
    ):
        require(equal(cfg[name], verified.params[key]), "inherited algorithm configuration differs")
    bindings = dict(verified.hashes)
    for relative, digest in verified.code.items():
        path = str(REPO / relative)
        if path in codes:
            require(codes[path] == digest, "live inherited source differs from parent")
    require(sha256(LOCAL_AUDIT / "audit.provenance.json") == LOCAL_AUDIT_SHA, "local audit changed")
    audit = verify_outputs(
        LOCAL_AUDIT, "audit.provenance.json", "independent_reflection_local_output_audit", bindings
    )
    local_summary = read_json(output_path(audit, "summary.json", LOCAL_AUDIT / "summary.json"))
    require(local_summary["screening"]["fresnel"]["gate_pass"] is True, "local Fresnel gate failed")
    qualification = None
    if qualification_root is not None:
        qualification = verify_qualification(qualification_root, codes, bindings)
    require(preflight or qualification is not None, "successful GPU qualification required")
    # Loading one NPZ field decompresses the full array. No other trajectory field
    # is accessed; only this single row survives the registration function.
    with np.load(verified.root / "candidate.npz", allow_pickle=False) as saved:
        all_poses = saved["pose_xyz"]
        require(
            all_poses.shape == (len(verified.frames), 3, 3, 3), "parent pose field shape differs"
        )
        seed = all_poses[START - int(verified.frames[0])].copy()
    del all_poses
    require(
        seed.dtype == np.float64
        and np.isfinite(seed).all()
        and np.all(seed >= [0, 0, -6])
        and np.all(seed <= [192, 192, 186]),
        "seed geometry invalid",
    )
    require(np.all(np.linalg.norm(np.diff(seed, axis=1), axis=-1) > 0.05), "degenerate seed body")
    manifest = verified.manifest
    inference = dict(verified.inference)
    for filename in (
        "calibration_manifest.json",
        *(f"cam{i}_bundle_refractive.json" for i in range(3)),
    ):
        path = verified.root / "calibration_snapshot" / filename
        inference[str(path)] = bindings[str(path)]
    plan = dict(
        schema="reflection_temporal_registration_v1",
        frame_range=[START, STOP],
        source_frame_range=[int(verified.frames[0]), int(verified.frames[-1]) + 1],
        frames=FRAMES.tolist(),
        fps=166,
        arms=list(ARMS),
        config=cfg,
        sequence=str(verified.sequence),
        input_root=str(verified.input_root),
        calibration_root=str(verified.root / "calibration_snapshot"),
        experiment=manifest["experiment"],
        source_manifest_sha256=sha256(verified.sequence / "manifest.json"),
        code_hashes=codes,
        evidence_hashes=bindings,
        inference_hashes=inference,
        software_runtime=software_runtime(),
        qualification=qualification,
        protocol_sha256=PROTOCOL_SHA,
        raw_storage=manifest.get("raw_storage", "lossless_clip"),
        seed=dict(
            frame_idx=START,
            source_candidate=str(verified.root / "candidate.npz"),
            source_candidate_sha256=bindings[str(verified.root / "candidate.npz")],
            array="pose_xyz",
            retained_row=START - int(verified.frames[0]),
            actual_decompression="full pose_xyz field, registration only",
            future_influence_possible=True,
            biological_ID_truth=False,
            old_review_carried=False,
            initial_review=[False] * 3,
            initial_velocity=np.zeros((3, 3)).tolist(),
            initial_ages="all zero",
        ),
        direct_support_renderer="original unbounded direct only",
        evaluation_after_both_trajectories_saved=True,
        preflight_only=preflight,
        no_global_identity_lineage=True,
    )
    return plan, seed


def snapshot(code, output, record):
    for path, digest in code.items():
        target = output / "code_snapshot" / Path(path).relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        require(sha256(target) == digest, "snapshot differs")
        record.output(target.relative_to(output).as_posix(), target, sha256=digest)


def save_json(path, value, record, role=None):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    record.output(role or path.name, path, sha256=sha256(path))


def guard(bindings, code, record):
    verify_files(bindings)
    verify_files(code)
    verify_files({entry["path"]: entry["sha256"] for entry in record.outputs.values()})


def register(args, code):
    plan, seed = prepare_registration(
        args.geometry_run, args.qualification, code, preflight=args.preflight
    )
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    stage = (
        "reflection_temporal_preflight" if args.preflight else "reflection_temporal_registration"
    )
    name = "preflight" if args.preflight else "registration"
    with provenance(stage, output, name=name) as record:
        for path, digest in {**plan["evidence_hashes"], **code}.items():
            record.input(path, path, sha256=digest)
        snapshot(code, output, record)
        seed_path = output / "initial_pose_xyz.npy"
        np.save(seed_path, seed, allow_pickle=False)
        plan["seed_array_sha256"] = sha256(seed_path)
        record.output("initial_pose", seed_path, sha256=plan["seed_array_sha256"])
        save_json(output / "seed_registration.json", plan["seed"], record, "seed")
        save_json(output / "plan.json", plan, record, "plan")
        guard(plan["evidence_hashes"], code, record)
    return plan


def load_registration(root, code):
    """No historical NPZ/coordinate loading in this execution-side verifier."""
    root = workspace(root)
    bindings = {}
    seal = verify_outputs(
        root, "registration.provenance.json", "reflection_temporal_registration", bindings
    )
    plan = read_json(output_path(seal, "plan", root / "plan.json"))
    require(
        plan["preflight_only"] is False
        and plan["frames"] == FRAMES.tolist()
        and plan["arms"] == list(ARMS)
        and plan["protocol_sha256"] == PROTOCOL_SHA
        and equal(plan["config"], configs())
        and equal(plan["code_hashes"], code)
        and equal(plan["software_runtime"], software_runtime()),
        "registered conditions differ",
    )
    expected = {**plan["evidence_hashes"], **code}
    require(
        {k: v["sha256"] for k, v in seal["inputs"].items()} == expected,
        "registration input set differs",
    )
    for path, digest in expected.items():
        require(
            workspace(seal["inputs"][path]["path"]) == workspace(path), "input key/path differs"
        )
        # Only hash the parent candidate; never np.load it in this branch.
        bind(path, digest, bindings)
    qualification = verify_qualification(plan["qualification"]["root"], code, bindings)
    require(equal(qualification, plan["qualification"]), "qualification changed since registration")
    seed_path = output_path(seal, "initial_pose", root / "initial_pose_xyz.npy")
    require(sha256(seed_path) == plan["seed_array_sha256"], "seed hash differs")
    seed = np.load(seed_path, allow_pickle=False)
    require(
        seed.shape == (3, 3, 3) and seed.dtype == np.float64 and np.isfinite(seed).all(),
        "seed schema differs",
    )
    return plan, seed.copy(), bindings


def bounded_frames(reader, start=START, stop=STOP):
    """Retain the reader's full manifest/cache; seek exactly once, never read stop."""
    require(
        type(start) is int
        and type(stop) is int
        and start < stop
        and start >= int(reader.frames[0])
        and stop <= int(reader.frames[-1]) + 1,
        "invalid native subwindow",
    )
    source = iter_from(reader, start)
    try:
        for frame in range(start, stop):
            try:
                observation = next(source)
            except StopIteration as exc:
                raise ValueError("source ended before the declared window") from exc
            require(observation.frame_idx == frame, "non-native/misordered frame")
            yield observation
    finally:
        source.close()


def validate_arrays(result, reader):
    require(set(result) == ARRAYS, "exact 21-array persistent schema required")
    base.validate_result(result, FRAMES, reader.nodes)
    persistent_runner.validate_arm_arrays(result, FRAMES)
    require(np.isfinite(result["pose_xyz"]).all(), "nonfinite latent trajectory")
    expected = np.where(result["observed"][:, :, 1, None], result["pose_xyz"][:, :, 1], np.nan)
    require(
        np.array_equal(result["xyz"], expected, equal_nan=True), "raw chest availability differs"
    )
    expected = np.where(result["identity_review"][:, :, None], np.nan, expected)
    require(
        np.array_equal(result["accepted_xyz"], expected, equal_nan=True),
        "accepted review mask differs",
    )


def execute_arms(plan, seed, reader, rig, output, record):
    """No evaluation/old geometry access; save direct then Fresnel before returning."""
    require(
        torch.cuda.is_available(), "registered CUDA device is required; no automatic CPU fallback"
    )
    device_name = torch.cuda.get_device_name(0)
    require(
        device_name == plan["qualification"]["runtime"]["cuda_device_name"], "qualified GPU differs"
    )
    results, timings = {}, {}
    config = TrackerConfig(**plan["config"]["tracker"])
    for arm in ARMS:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        tracker_cls = (
            PersistentNodeThreeBodyTracker if arm == "direct" else ReflectionPersistentNodeTracker
        )
        tracker = tracker_cls(
            rig,
            seed.copy(),
            config=config,
            recovery_config=NodeRecoveryConfig(**plan["config"]["recovery"]),
            persistence_config=PersistentNodeConfig(**plan["config"]["persistence"]),
        )
        observations = bounded_frames(reader)
        count, started = 0, time.perf_counter()
        progress_path = output / f"{arm}_progress.jsonl"
        with progress_path.open("x", encoding="utf-8", buffering=1) as stream:

            def inputs(observations=observations, arm=arm, started=started):
                nonlocal count
                for observation in observations:
                    yield observation
                    count += 1
                    if count % 25 == 0 or count == len(FRAMES):
                        row = dict(
                            arm=arm,
                            processed=count,
                            frame_idx=observation.frame_idx,
                            elapsed_seconds=time.perf_counter() - started,
                        )
                        stream.write(json.dumps(row) + "\n")
                        print(json.dumps(row), flush=True)

            try:
                result = tracker.track(inputs())
                torch.cuda.synchronize()
            finally:
                observations.close()
        elapsed = time.perf_counter() - started
        require(
            count == len(FRAMES) and tracker.initialize_count == 1,
            "incomplete or reinitialized tracking",
        )
        validate_arrays(result, reader)
        path = output / f"{arm}_candidate.npz"
        np.savez_compressed(path, **result)
        record.output(arm + "_candidate", path, sha256=sha256(path))
        record.output(arm + "_progress", progress_path, sha256=sha256(progress_path))
        timings[arm] = dict(
            tracking_seconds=elapsed,
            fps=len(FRAMES) / elapsed,
            initialization_count=tracker.initialize_count,
            reset_count=0,
            track_invocation_count=1,
            native_frames_read=count,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            proposal_diagnostics=tracker.node_recovery_diagnostics,
        )
        results[arm] = result
        del tracker
    return results, timings


def runs_of_true(flags, frames):
    padded = np.r_[False, np.asarray(flags, bool), False].astype(np.int8)
    starts, stops = np.flatnonzero(np.diff(padded) == 1), np.flatnonzero(np.diff(padded) == -1)
    return [[int(frames[a]), int(frames[b - 1]) + 1] for a, b in zip(starts, stops, strict=True)]


def metrics(result):
    observed, review = result["observed"], result["identity_review"]
    accepted = np.isfinite(result["accepted_xyz"]).all(-1)
    gaps = [runs_of_true(~observed[:, f, 1], FRAMES) for f in range(3)]
    return dict(
        direct_nodes_per_slot=observed.sum(0).tolist(),
        direct_nodes_total=observed.sum((0, 1)).tolist(),
        direct_chest_per_slot=observed[:, :, 1].sum(0).tolist(),
        direct_chest_total=int(observed[:, :, 1].sum()),
        accepted_chest_per_slot=accepted.sum(0).tolist(),
        accepted_chest_total=int(accepted.sum()),
        longest_chest_gap_per_slot=[max((b - a for a, b in row), default=0) for row in gaps],
        chest_gap_ranges=gaps,
        review_ranges=[runs_of_true(review[:, f], FRAMES) for f in range(3)],
    )


def promotion_gate(matrix, statistics):
    matrix = np.asarray(matrix, float)
    require(matrix.shape == (2, 2) and np.isfinite(matrix).all(), "finite 2x2 matrix required")
    old, new = statistics["direct"], statistics["fresnel"]
    checks = dict(
        image_vs_direct=bool(matrix[1, 1] < matrix[0, 0] - 1e-10),
        image_same_model=bool(matrix[1, 1] < matrix[0, 1] - 1e-10),
        direct_chest_total=new["direct_chest_total"] >= old["direct_chest_total"],
        accepted_chest_total=new["accepted_chest_total"] >= old["accepted_chest_total"],
        direct_chest_every_slot=bool(
            np.all(np.asarray(new["direct_chest_per_slot"]) >= old["direct_chest_per_slot"])
        ),
        accepted_chest_every_slot=bool(
            np.all(np.asarray(new["accepted_chest_per_slot"]) >= old["accepted_chest_per_slot"])
        ),
        longest_gap_every_slot=bool(
            np.all(
                np.asarray(new["longest_chest_gap_per_slot"]) <= old["longest_chest_gap_per_slot"]
            )
        ),
    )
    return dict(
        checks=checks,
        passed=all(checks.values()),
        tolerance=1e-10,
        meaning="Permits consideration of further development only; no identity certification",
    )


def evaluate_completed(results, reader, rig, config, output, record):
    """Only called after BOTH candidate files are saved and bound as outputs."""
    for arm in ARMS:
        path = output / f"{arm}_candidate.npz"
        require(
            arm + "_candidate" in record.outputs
            and sha256(path) == record.outputs[arm + "_candidate"]["sha256"],
            "both trajectories must be saved unchanged before evaluation",
        )
    all_parts = []
    sample_dir = output / "evaluation"
    sample_dir.mkdir(exist_ok=False)
    for t, frame in enumerate(FRAMES):
        index = int(frame - reader.frames[0])
        points = {v: reader.nodes.at(v, int(frame)) for v in VIEWS}
        rays = RayObservations.from_points(rig, points)
        for arm in ARMS:
            result = results[arm]
            sources, observed, _ = supported_nodes(
                result["pose_xyz"][t], result["sources"][t], rays, rig
            )
            require(
                np.array_equal(sources, result["sources"][t])
                and np.array_equal(observed, result["observed"][t]),
                "strict final support differs",
            )
        masks = {}
        for view in VIEWS:
            data = reader.masks[view]
            decoded = decode_frame_masks(data, index)
            require(
                array_sha256(decoded) == data["frame_mask_sha256"][index].decode(),
                "decoded mask SHA differs",
            )
            masks[view] = decoded if data["view_available"][index] else None
        dummy = np.zeros((1200, 1200, 3), np.uint8)
        observation = FrameObservation(int(frame), points, dict.fromkeys(VIEWS, dummy), masks, {})
        fitting = sample_pixels(
            observation,
            "detectron2",
            config.positive_samples,
            config.near_samples,
            config.global_samples,
        )
        targets = sample_body_evaluation(observation, fitting.pixels)
        saved = dict(
            frame_idx=np.array(frame),
            fit_pixels=np.stack([fitting.pixels[v] for v in VIEWS]),
            fit_target=fitting.target,
            fit_weights=fitting.weights,
            target=targets.target,
            weights=targets.weights,
            foreground=targets.foreground,
            strata=targets.strata,
            sampled_counts=targets.sampled_counts,
            pool_counts=targets.pool_counts,
            excluded_fit_unique_count=targets.excluded_fit_unique_count,
            view_available=targets.view_available,
            foreground_nonempty=targets.foreground_nonempty,
            **{v + "_pixels": targets.pixels[v] for v in VIEWS},
        )
        poses = np.stack([results[a]["pose_xyz"][t] for a in ARMS])
        if targets.target.shape[1] == 0:
            valid = np.empty((3, 0), bool)
            unions = {arm: np.empty((2, 3, 0)) for arm in ARMS}
        else:
            renderer = ReflectionFishRenderer(
                rig,
                targets.pixels,
                RendererConfig(width_ratio=config.width_ratio),
                direct_mode="unbounded",
                device="cpu",
                dtype=torch.float64,
            )
            with torch.no_grad():
                rendering = renderer.render(poses)
            valid = rendering.ray_valid.numpy()
            unions = {arm: getattr(rendering, arm + "_union").numpy() for arm in ARMS}
        parts = []
        for scorer in ARMS:
            summary = summarize_pixel_error(unions[scorer], valid, targets)
            saved[scorer + "_union"] = unions[scorer]
            for key, value in summary.items():
                saved[scorer + "_" + key] = value
            parts.append(summary["loss_by_stratum"])
        all_parts.append(np.stack(parts, axis=1))
        saved["calibrated_valid"] = valid
        path = sample_dir / f"f{frame}.npz"
        np.savez_compressed(path, **saved)
        record.output(path.relative_to(output).as_posix(), path, sha256=sha256(path))
    parts = np.stack(all_parts)  # T,fitted arm,scorer,view,stratum
    means = parts.sum(-1).mean(-1)
    statistics = {arm: metrics(results[arm]) for arm in ARMS}
    matrix = means.mean(0)
    summary = dict(
        matrix=matrix.tolist(),
        axes=["fitted_arm", "scoring_model"],
        arms=list(ARMS),
        statistics=statistics,
        gate=promotion_gate(matrix, statistics),
        per_frame_mean_errors=means.tolist(),
        per_view_stratum_means=parts.mean(0).tolist(),
        same_model_worse_frames=[
            int(FRAMES[t]) for t in np.flatnonzero(means[:, 1, 1] > means[:, 0, 1])
        ],
        common_direct_worse_frames=[
            int(FRAMES[t]) for t in np.flatnonzero(means[:, 1, 0] > means[:, 0, 0])
        ],
        evaluation_device="cpu",
        evaluation_labels="same independent D2 masks",
        dummy_images="native shape only; D2 sampling never accesses pixel content",
        strict_support_recomputed_frames_per_arm=200,
        gt_read=False,
        biological_ID_score=False,
    )
    path = output / "evaluation_metrics.npz"
    np.savez_compressed(
        path, frame_idx=FRAMES, per_view_stratum_errors=parts, mean_view_errors=means
    )
    record.output("evaluation_metrics", path, sha256=sha256(path))
    save_json(output / "comparison.json", summary, record, "comparison")
    return summary


def execute(args, code):
    plan, seed, bindings = load_registration(args.registration, code)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        with provenance("reflection_temporal_comparison", output, name="experiment") as record:
            for path, digest in {**bindings, **code}.items():
                record.input(path, path, sha256=digest)
            snapshot(code, output, record)
            save_json(output / "plan.json", plan, record, "plan")
            np.save(output / "initial_pose_xyz.npy", seed, allow_pickle=False)
            require(
                sha256(output / "initial_pose_xyz.npy") == plan["seed_array_sha256"],
                "copied seed differs",
            )
            record.output(
                "initial_pose", output / "initial_pose_xyz.npy", sha256=plan["seed_array_sha256"]
            )
            manifest = read_json(Path(plan["sequence"]) / "manifest.json")
            reader_cls = (
                SourceSequenceReader
                if manifest.get("raw_storage") == "original_source_video"
                else SequenceReader
            )
            reader = reader_cls(plan["sequence"], plan["input_root"])
            require(
                np.array_equal(reader.frames, np.arange(*plan["source_frame_range"])),
                "full source range differs",
            )
            require(
                sha256(Path(plan["sequence"]) / "manifest.json") == plan["source_manifest_sha256"],
                "source manifest differs",
            )
            rig = load_rig(plan["calibration_root"])
            require(tuple(rig.views) == VIEWS, "registered view order differs")
            results, timings = execute_arms(plan, seed, reader, rig, output, record)
            # This boundary occurs only after both tracks and NPZ writes complete.
            comparison = evaluate_completed(
                results, reader, rig, TrackerConfig(**plan["config"]["tracker"]), output, record
            )
            summary = dict(
                schema="reflection_temporal_comparison_v1",
                frame_range=[START, STOP],
                native_fps=166,
                temporal_subsampling=False,
                arms=list(ARMS),
                timings=timings,
                technical_completion=True,
                qualification=plan["qualification"],
                seed=plan["seed"],
                initialized_once_each=True,
                external_resets_each=0,
                inference_old_candidate_deserialized=False,
                gt_read=False,
                evaluation_after_both_saved=True,
                direct_support_renderer="original direct only",
                code_unchanged=True,
                inputs_unchanged=True,
                comparison=comparison,
                config=plan["config"],
                timing_scope="Tracking includes native decode/mask decode/flow/proposals/fits; "
                "excludes staging/hashes/evaluation",
                no_global_identity_lineage=True,
                biological_ID_validation=False,
            )
            save_json(output / "summary.json", summary, record, "summary")
            guard(bindings, code, record)
    finally:
        torch.set_num_threads(old_threads)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--register-only", action="store_true")
    modes.add_argument(
        "--preflight",
        action="store_true",
        help="CPU input/seed checks only; cannot authorize inference",
    )
    modes.add_argument("--registration", type=Path)
    parser.add_argument(
        "--geometry-run",
        type=Path,
        default=REPO / "work/three_body/persistent_mouth_continuous_one_minute_v1",
    )
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    require(
        args.output.is_relative_to(REPO) and not args.output.exists(),
        "new workspace output required",
    )
    code = code_hashes()
    if args.registration:
        execute(args, code)
    else:
        register(args, code)
    print(json.dumps(dict(status="ok", output=str(args.output), preflight_only=args.preflight)))


if __name__ == "__main__":
    main()
