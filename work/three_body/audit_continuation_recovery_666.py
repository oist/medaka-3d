"""CPU-only independent audit of the fixed four-phase 666-frame continuation experiment."""

import argparse
import dataclasses
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))

import identity_inputs as verified  # noqa: E402
from observations import SequenceReader  # noqa: E402
from prepare_sequence import decode_frame_masks  # noqa: E402
from tracker_checkpoint import read_checkpoint  # noqa: E402

from medaka3d.calib import load_rig  # noqa: E402
from medaka3d.provenance import provenance  # noqa: E402

BASE = verified.base
VIEWS = BASE.VIEWS
PHASES = ("single", "prefix271", "resumed380", "resumed666")
TOTALS = (666, 271, 380, 666)
PREVIOUS = (0, 0, 271, 380)
RESTORES = (0, 0, 1, 2)
PLAN_ROOT = REPO / "work/three_body/continuation_recovery_666_plan_v1"
PLAN_SHA256 = "684fcb2d3b9c5f94ebfa761d4ef770172f89dd7891491a74f846d56dbc48a5f7"
STATE_FIELDS = (
    "pose",
    "velocity",
    "missing",
    "review",
    "cost",
    "node_missing",
    "direct_pectoral_age",
)
HISTORY_FIELDS = (
    "frame_idx",
    "pose_xyz",
    "observed",
    "sources",
    "identity_review",
    "mask_supported",
    "body_supported",
    "missing",
    "costs",
    "path_costs",
    "loss_parts",
    "score_margin",
    "parent",
    "proposal_kind",
    "hypotheses",
    "proposal_flow_nodes",
    "proposal_flow_parent",
    "node_missing",
    "direct_pectoral_age",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest_array(array):
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def same_array(actual, expected, label):
    if (
        actual.dtype != expected.dtype
        or actual.shape != expected.shape
        or actual.tobytes(order="C") != expected.tobytes(order="C")
    ):
        raise ValueError("dtype/shape/bytes differ: " + label)
    return digest_array(actual)


def normalize(value):
    if isinstance(value, np.ndarray):
        return digest_array(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [normalize(item) for item in value]
    return value


def bind(path, expected, hashes):
    path = Path(path).resolve(strict=True)
    actual = BASE.sha256(path)
    if actual != expected or hashes.get(str(path), actual) != actual:
        raise ValueError("SHA mismatch: " + str(path))
    hashes[str(path)] = actual


def unchanged(hashes):
    for path, expected in hashes.items():
        if BASE.sha256(path) != expected:
            raise RuntimeError("audit evidence changed: " + path)


def code_paths():
    paths = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == ".py" and (
                path.is_relative_to(LAB) or path.is_relative_to(REPO / "src/medaka3d")
            ):
                paths.add(path)
    return sorted(paths)


def verify_plan(reference, phases, hashes):
    seal_path, plan_path = PLAN_ROOT / "plan.provenance.json", PLAN_ROOT / "plan.json"
    seal = read_json(seal_path)
    if seal.get("stage") != "three_body_continuation_recovery_plan" or seal.get("status") != "ok":
        raise ValueError("recovery plan has no successful seal")
    bind(seal_path, BASE.sha256(seal_path), hashes)
    output = seal["outputs"]["plan"]
    if Path(output["path"]).resolve() != plan_path or output["sha256"] != PLAN_SHA256:
        raise ValueError("registered recovery plan output binding differs")
    bind(plan_path, PLAN_SHA256, hashes)
    plan = read_json(plan_path)
    expected = {
        "schema": "continuation_recovery_development_plan_v1",
        "source_frame_range": [76620, 77286],
        "native_frames": 666,
        "fps": 166,
        "uninterrupted_process_frames": 666,
        "split_completed_counts": [271, 380, 666],
        "resume_next_native_frames": [76891, 77000],
        "prior_outputs_opened_for_design_only": True,
        "tracker_inference_prior_candidate_opened_only_after_full_output": True,
    }
    if (
        any(canonical(plan.get(key)) != canonical(value) for key, value in expected.items())
        or Path(plan["reference_run"]).resolve() != reference.root
        or set(plan["input_source_sha256"]) != set(seal["inputs"])
    ):
        raise ValueError("recovery plan content/input set differs")
    for path, digest in plan["input_source_sha256"].items():
        item = seal["inputs"][path]
        if Path(item["path"]).resolve() != Path(path) or item["sha256"] != digest:
            raise ValueError("recovery plan input role/digest differs")
        bind(path, digest, hashes)
    finished = datetime.fromisoformat(seal["finished_at"])
    if any(finished >= datetime.fromisoformat(phase["seal"]["started_at"]) for phase in phases):
        raise ValueError("recovery plan was not sealed before all new tracking phases")
    return plan


def verify_phase(
    root, total, prior, restore_count, reference, hashes, expected_bindings=None, parent=None
):
    seal_path = root / "continuation.provenance.json"
    seal = read_json(seal_path)
    if seal.get("status") != "ok" or seal.get("stage") != "three_body_persistent_continuation":
        raise ValueError("phase is not successfully sealed: " + str(root))
    bind(seal_path, BASE.sha256(seal_path), hashes)
    params, outputs = seal["params"], seal["outputs"]
    bindings = params["bindings"]
    if expected_bindings is not None and canonical(bindings) != canonical(expected_bindings):
        raise ValueError("phase input/code/runtime bindings differ")
    if (
        params["stop_after"] != total
        or params.get("all_input_source_snapshot_output_sha_unchanged") is not True
        or bindings["source_range"] != [76620, 77286]
        or Path(bindings["reference_run"]).resolve() != reference.root
        or bindings["input_sha256"] != reference.hashes
    ):
        raise ValueError("phase interval/reference/guard differs")
    if params["resume_phase"] != (None if parent is None else str(parent)):
        raise ValueError("phase resume parent differs")
    expected_inputs = {**reference.hashes, **bindings["code_sha256"]}
    if parent is not None:
        for filename in (
            "continuation.provenance.json",
            "summary.json",
            "checkpoint/manifest.json",
            "checkpoint/arrays.npz",
            "checkpoint/metadata.json",
        ):
            path = parent / filename
            expected_inputs[str(path)] = BASE.sha256(path)
    if set(seal["inputs"]) != set(expected_inputs):
        raise ValueError("phase input seal membership differs")
    for key, expected in expected_inputs.items():
        item = seal["inputs"][key]
        if Path(item["path"]).resolve() != Path(key) or item.get("sha256") != expected:
            raise ValueError("phase input path/digest binding differs")
        bind(key, expected, hashes)
    for source, expected in bindings["code_sha256"].items():
        bind(source, expected, hashes)
        snapshot = root / "code_snapshot" / Path(source).relative_to(REPO)
        item = outputs.get(str(snapshot))
        if item is None or Path(item["path"]).resolve() != snapshot:
            raise ValueError("phase source snapshot not sealed")
        bind(snapshot, expected, hashes)
    for key, item in outputs.items():
        path = Path(item["path"]).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("phase output outside its root: " + key)
        bind(path, item["sha256"], hashes)
        if item.get("size") != path.stat().st_size:
            raise ValueError("phase output size differs")
    for name, filename in (
        ("summary", "summary.json"),
        ("progress", "progress.jsonl"),
        ("checkpoint_manifest", "checkpoint/manifest.json"),
    ):
        if Path(outputs[name]["path"]).resolve() != root / filename:
            raise ValueError("named output binding differs: " + name)
    for filename in ("checkpoint/arrays.npz", "checkpoint/metadata.json"):
        if str(root / filename) not in outputs:
            raise ValueError("checkpoint payload output seal is absent")
    arrays, metadata, manifest = read_checkpoint(root / "checkpoint", expected_bindings=bindings)
    summary = read_json(root / "summary.json")
    expected = {
        "schema": "persistent_continuation_phase_v1",
        "technical_completion": True,
        "full_sequence_complete": total == 666,
        "frames": [76620, 77286],
        "processed_total": total,
        "phase_frames": [76620 + prior, 76620 + total],
        "phase_processed": total - prior,
        "logical_initialization_count": 1,
        "checkpoint_restore_count": restore_count,
        "external_pose_reset_count": 0,
        "temporal_subsampling": False,
        "original_candidate_21_arrays_byte_exact": total == 666,
        "original_candidate_loaded_only_after_full_inference": total == 666,
        "human_annotation_input": False,
    }
    if any(canonical(summary.get(key)) != canonical(value) for key, value in expected.items()):
        raise ValueError("phase summary lifecycle differs")
    if (
        metadata["processed_frames"] != total
        or metadata["previous_frame"] != 76619 + total
        or metadata["restore_count"] != restore_count
        or metadata["initialize_count"] != 1
        or summary["node_recovery_diagnostics"] != metadata["node_recovery_diagnostics"]
        or summary["checkpoint_payload_bytes"] != manifest["payload"]["bytes"]
    ):
        raise ValueError("checkpoint lifecycle/summary differs")
    progress = [json.loads(line) for line in (root / "progress.jsonl").read_text().splitlines()]
    processed = [item["processed"] for item in progress]
    if (
        not processed
        or processed != sorted(set(processed))
        or processed[-1] != total
        or any(
            item["source_frame"] != 76619 + item["processed"]
            or item["phase_processed"] != item["processed"] - prior
            for item in progress
        )
    ):
        raise ValueError("phase progress frame alignment differs")
    if (root / "candidate.npz").exists() != (total == 666):
        raise ValueError("partial phase must not contain a final candidate")
    if total == 666 and Path(outputs["candidate"]["path"]).resolve() != root / "candidate.npz":
        raise ValueError("final candidate output seal differs")
    return {
        "root": root,
        "arrays": arrays,
        "metadata": metadata,
        "summary": summary,
        "bindings": bindings,
        "seal": seal,
        "progress": progress,
    }


def audit_history(phase, initial):
    arrays, metadata = phase["arrays"], phase["metadata"]
    offsets = arrays["history_offsets"]
    total = metadata["processed_frames"]
    if (
        offsets.dtype != np.int64
        or offsets.shape != (total + 1,)
        or offsets[0] != 0
        or np.any(np.diff(offsets) < 1)
        or np.any(np.diff(offsets) > 6)
        or metadata["record_fields"] != list(HISTORY_FIELDS)
        or metadata["state_fields"] != list(STATE_FIELDS)
    ):
        raise ValueError("history schema/offsets differ")
    previous = [
        {
            "pose": initial,
            "velocity": np.zeros((3, 3)),
            "missing": np.zeros(3, int),
            "review": np.zeros(3, bool),
            "cost": 0.0,
            "node_missing": np.zeros((3, 3), int),
            "direct_pectoral_age": np.zeros(3, int),
        }
    ]
    parent_nonzero, eligible_calls = 0, 0
    for time in range(total):
        eligible_calls += bool(
            np.any(
                (previous[0]["node_missing"][:, 0] >= 4) & (previous[0]["direct_pectoral_age"] >= 4)
            )
        )
        states = []
        for row in range(int(offsets[time]), int(offsets[time + 1])):
            record = {key: arrays["history_" + key][row] for key in HISTORY_FIELDS}
            index = int(record["parent"])
            if not 0 <= index < len(previous) or record["frame_idx"] != 76620 + time:
                raise ValueError("history parent/native frame differs")
            parent_nonzero += index != 0
            parent = previous[index]
            sources, observed = record["sources"], record["observed"]
            if not np.array_equal(observed, (sources >= 0).sum(1) >= 2):
                raise ValueError("direct support does not match retained sources")
            for view in range(3):
                for node in range(3):
                    used = sources[:, view, node]
                    used = used[used >= 0]
                    if len(used) != len(np.unique(used)):
                        raise ValueError("history source is reused")
            body = observed[:, 1] | (
                record["mask_supported"] & (sources[:, :, :2] >= 0).any(axis=(1, 2))
            )
            missing = np.where(body, 0, parent["missing"] + 1)
            node_missing = np.where(observed, 0, parent["node_missing"] + 1)
            direct_age = np.where(observed[:, 1], parent["direct_pectoral_age"] + 1, 0)
            for name, value in (
                ("body_supported", body),
                ("missing", missing),
                ("node_missing", node_missing),
                ("direct_pectoral_age", direct_age),
            ):
                same_array(record[name], value, name)
            if np.any(parent["review"] & ~record["identity_review"]) or np.any(
                (missing >= 4) & ~record["identity_review"]
            ):
                raise ValueError("sticky review lost")
            path_cost = parent["cost"] + record["costs"]
            same_array(np.asarray(record["path_costs"]), np.asarray(path_cost), "cumulative cost")
            delta = record["pose_xyz"][:, 1] - parent["pose"][:, 1]
            velocity = 0.5 * parent["velocity"] + 0.5 * delta
            speed = np.linalg.norm(velocity, axis=-1, keepdims=True)
            velocity *= np.minimum(1.0, 2.0 / np.maximum(speed, 1e-12))
            velocity = np.where(body[:, None], velocity, 0.8 * parent["velocity"])
            states.append(
                {
                    "pose": record["pose_xyz"],
                    "velocity": velocity,
                    "missing": missing,
                    "review": record["identity_review"],
                    "cost": float(path_cost),
                    "node_missing": node_missing,
                    "direct_pectoral_age": direct_age,
                }
            )
        previous = states
    for key in STATE_FIELDS:
        same_array(arrays["state_" + key], np.asarray([state[key] for state in previous]), key)
    if phase["summary"]["retained_history_rows"] != int(offsets[-1]) or phase["summary"][
        "active_beam_count"
    ] != len(previous):
        raise ValueError("summary history counts differ")
    if eligible_calls != metadata["node_recovery_diagnostics"]["calls_with_age_eligible_fish"]:
        raise ValueError("age-eligible calls disagree with first-parent age history")
    return {
        "completed_frames": total,
        "retained_history_rows": int(offsets[-1]),
        "active_beam_count": len(previous),
        "nonzero_parent_history_rows": parent_nonzero,
        "all_parent_state_recurrences_byte_exact": True,
        "age_eligible_calls_independently_counted": eligible_calls,
        "checkpoint_logical_array_bytes": sum(value.nbytes for value in arrays.values()),
    }


def audit_previous(phase, reader):
    arrays, metadata = phase["arrays"], phase["metadata"]
    native = metadata["previous_frame"]
    time = native - int(reader.frames[0])
    records = {}
    for view in VIEWS:
        capture = cv2.VideoCapture(str(reader.videos[view]))
        try:
            if not capture.isOpened() or not capture.set(cv2.CAP_PROP_POS_FRAMES, time):
                raise ValueError("cannot seek original previous raw frame")
            ok, image = capture.read()
            if not ok or capture.get(cv2.CAP_PROP_POS_FRAMES) != time + 1:
                raise ValueError("raw previous frame alignment failed")
        finally:
            capture.release()
        if hashlib.sha256(image.tobytes()).hexdigest() != reader.pixel_hashes[view][time]:
            raise ValueError("previous raw pixel SHA differs from staging")
        mask = (
            decode_frame_masks(reader.masks[view], time)
            if reader.masks[view]["view_available"][time]
            else None
        )
        expected = {
            "images_bgr": image,
            "detections": reader.nodes.at(view, native),
            "masks": mask,
            "backgrounds_bgr": reader.backgrounds[view],
        }
        records[view] = {}
        for field, value in expected.items():
            key = "previous_" + field + "_" + view
            if metadata["previous_observation"][key] != (value is not None):
                raise ValueError("previous None/available evidence differs")
            if value is not None:
                records[view][field] = same_array(arrays[key], value, key)
            else:
                if key in arrays:
                    raise ValueError("unavailable previous mask has invented array")
                records[view][field] = None
    return {
        "native_frame": native,
        "all_four_fields_three_views_byte_exact": True,
        "raw_pixel_sha_matches_original_manifest": True,
        "views": records,
    }


def traced_candidate(arrays):
    offsets = arrays["history_offsets"]
    selected, parent = [], 0
    for time in range(len(offsets) - 2, -1, -1):
        row = int(offsets[time]) + parent
        selected.append(row)
        parent = int(arrays["history_parent"][row])
    selected = selected[::-1]
    result = {
        key: np.asarray([arrays["history_" + key][row] for row in selected])
        for key in HISTORY_FIELDS
        if key != "parent"
    }
    result["xyz"] = np.where(result["observed"][:, :, 1, None], result["pose_xyz"][:, :, 1], np.nan)
    result["accepted_xyz"] = np.where(result["identity_review"][:, :, None], np.nan, result["xyz"])
    result["body_xyz"] = np.where(
        result["body_supported"][:, :, None], result["pose_xyz"][:, :, 1], np.nan
    )
    return result


def recovery_boundaries(phases, original, plan):
    final = phases[-1]
    counters = final["metadata"]["node_recovery_diagnostics"]
    if (
        counters["calls_with_age_eligible_fish"] != 60
        or counters["calls_with_additional_proposals"] != 2
    ):
        raise ValueError("the registered development recovery count differs")
    traced = traced_candidate(final["arrays"])
    selected_times = traced["frame_idx"][traced["proposal_kind"] == "mouth_recovery"].tolist()
    if selected_times != plan["resume_next_native_frames"]:
        raise ValueError("selected mouth recoveries differ from the registered two cases")
    rows = []
    for index, native in enumerate(selected_times):
        before, after = phases[index + 1], phases[index + 2]
        arrays = before["arrays"]
        if before["metadata"]["previous_frame"] != native - 1:
            raise ValueError("checkpoint did not stop immediately before recovery")
        eligible = (arrays["state_node_missing"][0, :, 0] >= 4) & (
            arrays["state_direct_pectoral_age"][0] >= 4
        )
        if not eligible.any():
            raise ValueError("first resumed frame lacked retained eligible parent ages")
        time = native - 76620
        offsets = final["arrays"]["history_offsets"]
        left, right = (int(value) for value in offsets[time : time + 2])
        fields = {
            key: normalize(traced[key][time])
            for key in ("observed", "node_missing", "direct_pectoral_age", "identity_review")
        }
        before_count = before["metadata"]["node_recovery_diagnostics"][
            "calls_with_additional_proposals"
        ]
        after_count = after["metadata"]["node_recovery_diagnostics"][
            "calls_with_additional_proposals"
        ]
        if before_count != index or after_count != index + 1:
            raise ValueError("additional-call count was reset or gained an unregistered call")
        same_array(
            np.asarray(traced["proposal_kind"][time]),
            np.asarray(original["proposal_kind"][time]),
            "recovery selection",
        )
        rows.append(
            {
                "resume_native_frame": native,
                "saved_previous_native_frame": native - 1,
                "first_parent_node_missing": arrays["state_node_missing"][0].tolist(),
                "first_parent_direct_pectoral_age": arrays["state_direct_pectoral_age"][0].tolist(),
                "eligible_fish_slots": np.flatnonzero(eligible).tolist(),
                "selected_proposal_kind": str(traced["proposal_kind"][time]),
                "selected_nodes_observed": traced["observed"][time].tolist(),
                "selected_identity_review": traced["identity_review"][time].tolist(),
                "selected_node_missing": traced["node_missing"][time].tolist(),
                "selected_direct_pectoral_age": traced["direct_pectoral_age"][time].tolist(),
                "selected_array_hashes": fields,
                "frame_hypotheses": int(traced["hypotheses"][time]),
                "retained_recovery_beam_records": int(
                    np.count_nonzero(
                        final["arrays"]["history_proposal_kind"][left:right] == "mouth_recovery"
                    )
                ),
                "additional_calls_before_checkpoint": before_count,
                "additional_calls_at_next_checkpoint": after_count,
            }
        )
    hypothesis_counts = traced["hypotheses"]
    if np.max(final["arrays"]["history_hypotheses"]) > 18:
        raise ValueError("the registered no-over-18 budget observation differs")
    return {
        "boundaries": rows,
        "selected_mouth_recovery_frames": selected_times,
        "maximum_hypotheses": int(hypothesis_counts.max()),
        "frames_over_18_hypotheses": int(np.count_nonzero(hypothesis_counts > 18)),
        "budget24_exercised": bool(np.any(hypothesis_counts == 24)),
        "meaning": "Actual two additions and selected mouths survived process restore; "
        "anatomical or biological correctness was not assessed.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for phase in PHASES:
        parser.add_argument("--" + phase, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    roots = [getattr(args, phase).resolve(strict=True) for phase in PHASES]
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(REPO):
        raise ValueError("audit output must be a new workspace directory")
    sources = code_paths()
    hashes = {str(path): BASE.sha256(path) for path in sources}
    # All four completion seals are mandatory before opening any phase arrays.
    for root in roots:
        path = root / "continuation.provenance.json"
        seal = read_json(path)
        if seal.get("status") != "ok" or seal.get("stage") != "three_body_persistent_continuation":
            raise ValueError("all four phases must be successfully complete")
        bind(path, BASE.sha256(path), hashes)
    binding = read_json(roots[0] / "continuation.provenance.json")["params"]["bindings"]
    reference = verified._verify_run(Path(binding["reference_run"]), appearance=False)
    if reference.frames.tolist() != list(range(76620, 77286)):
        raise ValueError("only the fixed 666-frame continuation experiment is audited")
    hashes.update(reference.hashes)
    phases = []
    for index, root in enumerate(roots):
        phases.append(
            verify_phase(
                root,
                TOTALS[index],
                PREVIOUS[index],
                RESTORES[index],
                reference,
                hashes,
                binding,
                parent=roots[index - 1] if index >= 2 else None,
            )
        )
    plan = verify_plan(reference, phases, hashes)
    output.mkdir(parents=True, exist_ok=False)
    with provenance(
        "three_body_continuation_independent_comparison", output, name="audit"
    ) as record:
        record.param("geometry_refit", False)
        record.param("gpu_used", False)
        record.param("new_tracking_run", False)
        record.param("human_or_GT_inputs", False)
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
        snapshots = {}
        for path in sources:
            target = output / "code_snapshot" / path.relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            snapshots[str(target)] = hashes[str(path)]
        reader = SequenceReader(reference.sequence, reference.input_root)
        rig = load_rig(reference.root / "calibration_snapshot")
        context = phases[0]["metadata"]["context"]
        if (
            context["tracker_config"] != reference.params["tracker_config"]
            or context["recovery_config"] != reference.params["node_recovery_config"]
            or context["persistence_config"] != reference.params["persistence_config"]
            or context["initial_pose"] != digest_array(reference.initial)
            or context["cameras"] != normalize([dataclasses.asdict(cam) for cam in rig.cameras])
        ):
            raise ValueError("checkpoint actual seed/config/calibration context differs")
        checks = {}
        for name, phase in zip(PHASES, phases, strict=True):
            if phase["metadata"]["context"] != context:
                raise ValueError("phase tracker contexts differ")
            checks[name] = {
                "history": audit_history(phase, reference.initial),
                "previous_observation": audit_previous(phase, reader),
            }
        single = phases[0]["arrays"]
        for name, phase in zip(PHASES[1:], phases[1:], strict=True):
            count = phase["metadata"]["processed_frames"]
            rows = int(single["history_offsets"][count])
            same_array(
                phase["arrays"]["history_offsets"],
                single["history_offsets"][: count + 1],
                name + " offsets",
            )
            for key in HISTORY_FIELDS:
                first, second = phase["arrays"]["history_" + key], single["history_" + key][:rows]
                if key == "proposal_kind":
                    if not np.array_equal(first, second):
                        raise ValueError("prefix proposal strings differ")
                else:
                    same_array(first, second, name + " history " + key)
            checks[name]["entire_history_matches_single_prefix"] = True
        resumed = phases[-1]["arrays"]
        if set(single) != set(resumed):
            raise ValueError("final checkpoint array keys differ")
        checkpoint_comparison = {key: same_array(resumed[key], single[key], key) for key in single}
        comparisons = {}
        with np.load(reference.root / "candidate.npz", allow_pickle=False) as archive:
            old = {key: archive[key] for key in archive.files}
        if len(old) != 21:
            raise ValueError("original must have the frozen 21-array schema")
        for name, phase in (("single", phases[0]), ("resumed666", phases[-1])):
            with np.load(phase["root"] / "candidate.npz", allow_pickle=False) as archive:
                candidate = {key: archive[key] for key in archive.files}
            trace = traced_candidate(phase["arrays"])
            if set(candidate) != set(old) or set(trace) != set(old):
                raise ValueError("candidate/backtrace schema differs")
            comparisons[name] = {}
            for key in old:
                comparisons[name][key] = same_array(candidate[key], old[key], name + " " + key)
                same_array(trace[key], candidate[key], name + " backtrace " + key)
        old_diagnostics = read_json(reference.root / "summary.json")["node_recovery_diagnostics"]
        if any(
            phase["summary"]["node_recovery_diagnostics"] != old_diagnostics
            for phase in (phases[0], phases[-1])
        ):
            raise ValueError("complete diagnostics differ from original")
        recovery = recovery_boundaries(phases, old, plan)
        times = [
            {
                "phase": name,
                **{
                    key: phase["summary"][key]
                    for key in (
                        "phase_frames",
                        "phase_processed",
                        "processed_total",
                        "checkpoint_restore_count",
                        "preparation_seconds",
                        "tracking_seconds",
                        "checkpoint_seconds",
                        "checkpoint_payload_bytes",
                        "retained_history_rows",
                        "active_beam_count",
                    )
                },
            }
            for name, phase in zip(PHASES, phases, strict=True)
        ]
        chain_seconds = {
            key: sum(item[key] for item in times[1:])
            for key in ("preparation_seconds", "tracking_seconds", "checkpoint_seconds")
        }
        summary = {
            "schema": "continuation_666_independent_comparison_v1",
            "status": "ok",
            "frames": [76620, 77286],
            "native_frames": 666,
            "native_fps": 166,
            "input_code_runtime_bindings_identical": True,
            "original_single_resumed_21_arrays_byte_exact": True,
            "full_final_checkpoint_arrays_byte_exact": checkpoint_comparison,
            "candidate_comparisons": comparisons,
            "phase_checks": checks,
            "phases": times,
            "chain_phase_time_sums": chain_seconds,
            "logical_initializations": 1,
            "chain_restores": 2,
            "external_pose_resets": 0,
            "complete_recovery_diagnostics_exact": old_diagnostics,
            "registered_plan_sha256": PLAN_SHA256,
            "recovery_boundaries": recovery,
            "new_data_or_GT_opened": False,
            "new_tracking_or_GPU_used": False,
            "limits": [
                "Only the fixed native 666-frame development interval was measured.",
                "No new three-animal identity truth or zero-swap conclusion follows.",
                "Two real recovery additions are exercised, not a 24-hypothesis batch.",
                "Full history and previous images are copied; costs grow with duration.",
                "Phase timers exclude startup, input hashes, staging and final comparison.",
                "Separate run timings are descriptive, not a controlled speed comparison.",
            ],
        }
        lines = [
            "# 666フレーム途中保存・復元の独立監査",
            "",
            "単一実行と271→380→666フレームの3プロセス実行は、元の完了結果を含む21配列すべてが"
            "dtype・shape・C-order bytesで完全一致しました。"
            "全入力・実行ソース・snapshot・出力SHAと"
            "成功sealも確認しました。新しい追跡・再fit・GPU計算は行っていません。",
            "",
            "| phase | native range (終端を含まない) | 今回frames | tracking秒 | 保存秒 |"
            " NPZ bytes |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for item in times:
            lines.append(
                f"| {item['phase']} | {item['phase_frames'][0]}:{item['phase_frames'][1]} | "
                f"{item['phase_processed']} | {item['tracking_seconds']:.3f} | "
                f"{item['checkpoint_seconds']:.3f} | {item['checkpoint_payload_bytes']} |"
            )
        lines += [
            "",
            "論理的な初期化は1回、checkpoint復元は2回、"
            "外部の姿勢による再初期化は0回です。"
            "各checkpointの全beam/historyを単一実行の対応prefixと照合し、親インデックス、"
            "累積cost、velocity、欠測age、直接胸部age、支持・review保持を独立再計算しました。"
            "最終checkpointの全配列も完全一致しています。",
            "",
            "直前フレームの3視点BGR・全instance masks・SLEAP detections・背景は、元の封印済み"
            "raw/packed mask/SL入力と全bytes一致しました。次processは次のnative frameから"
            "処理し、同じ直前画像をflowへ渡しています。",
            "",
            "事前に封印した計画どおり、76891・77000の直前で保存して別processから再開しました。"
            "元の年齢条件判定60回・口候補追加2回のカウンタと、両時刻のmouth_recovery選択が"
            "保持されました。checkpointに保存した親のnode欠測age・胸部直接支持ageから、"
            "初再開フレームの年齢条件も独立に確認しています。最大候補数は18以下で、"
            "24候補のbatchを実データで試した結果ではありません。"
            "人手ID真値は使っておらず、生物学的IDの正しさ・30分間の完走・必要な人手量も"
            "この監査からは確定しません。",
            "",
            "履歴全体を保存する形式なので、保存・復元コストは長さとともに増えます。"
            "表の時間は入力ステージング、hash検査、process起動、最終比較を含みません。"
            "別時刻に実行した値であり、速度改善率として解釈しません。",
            "",
        ]
        outputs = {}
        for filename, data in (
            ("summary.json", json.dumps(summary, indent=2, allow_nan=False)),
            ("RESULT.md", "\n".join(lines)),
        ):
            path = output / filename
            path.write_text(data, encoding="utf-8")
            outputs[str(path)] = BASE.sha256(path)
        for path, digest in {**snapshots, **outputs}.items():
            record.output(path, path, sha256=digest)
        unchanged({**hashes, **snapshots, **outputs})
        record.param("all_input_source_snapshot_output_sha_unchanged", True)
    print(json.dumps({"output": str(output), "status": "ok", "arrays_byte_exact": 21}))


if __name__ == "__main__":
    main()
