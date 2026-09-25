"""Separate fixed-window diagnostic replay of the original complete beam state.

Preflight reads small metadata/source files and file sizes only. It does NOT
validate payload hashes, decode video, restore a tracker, or query CUDA. Execute
performs the full immutable input checks once, then waits for the ROOT-owned
read-only GPU scheduling window. No production output or binding is rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORK, LAB = REPO / "work/three_body", REPO / "labs/20260909_three_body"
sys.path[:0] = [str(WORK), str(LAB), str(REPO / "src")]

import cv2  # noqa: E402
import long_phase_audit as phase  # noqa: E402
import numpy as np  # noqa: E402
import scipy  # noqa: E402
import torch  # noqa: E402
from chest_candidate_diagnostics import CandidateDiagnostics  # noqa: E402
from continuous_tracker import (  # noqa: E402
    RECORD_FIELDS,
    STATE_FIELDS,
    AcceleratedContinuousPersistentNodeTracker,
)
from mask_validation_receipts import ReceiptStore  # noqa: E402
from motion import FlowConfig  # noqa: E402
from node_recovery import NodeRecoveryConfig  # noqa: E402
from observations import CachedNodes  # noqa: E402
from persistent_node_tracker import PersistentNodeConfig  # noqa: E402
from prepare_sequence import decode_frame_masks  # noqa: E402
from receipted_long_phase_audit import verified_inputs  # noqa: E402
from replay_gpu_window import ReplayGPUWindow  # noqa: E402
from resumable_observations import iter_from  # noqa: E402
from run_long_remaining_supervisor import (  # noqa: E402
    sha256,
    snapshot,
    sources,
    unchanged,
    within,
)
from source_sequence import SourceSequenceReader  # noqa: E402
from tracker import TrackerConfig  # noqa: E402
from tracker_checkpoint import read_checkpoint, write_checkpoint  # noqa: E402

from medaka3d.provenance import provenance  # noqa: E402
from medaka3d.types import VIEWS  # noqa: E402

FIRST, START, STOP, REFERENCE_STOP, CHUNK = 49800, 66400, 66706, 76360, 1660
DIAGNOSTIC_RANGE = (66656, 66706)
PARENT = WORK / "long_continuation_through66400_v1"
REFERENCE = WORK / "long_remaining_supervisor_v1/phases/f066400-076360/tracking"
PLAN_SOURCE = WORK / "CHEST_RECOVERY_NEXT_EXPERIMENT.md"
STAGE = "chest_deficit_original_beam_replay_v1"
require = phase.require


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def byte_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def code_hashes():
    # Outer experimental provenance only; never use this set as original binding.
    return {**sources(), str(Path(__file__).resolve()): sha256(__file__),
            str(PLAN_SOURCE): sha256(PLAN_SOURCE)}


def static_preflight(parent=PARENT, reference=REFERENCE, evidence=None):
    """Small metadata/source/size checks; explicitly not a full evidence verifier."""
    parent, reference = within(parent), within(reference)
    require(parent == PARENT and reference == REFERENCE, "fixed replay parent/reference required")
    rows = []
    for root, stop in ((parent, START), (reference, REFERENCE_STOP)):
        if evidence is not None:
            for name in ("continuation.provenance.json", "checkpoint/metadata.json",
                         "checkpoint/manifest.json"):
                evidence.add(root / name)
        seal = read(root / "continuation.provenance.json")
        require(seal.get("status") == "ok" and seal.get("stage") == "three_body_long_continuation"
                and seal["params"].get("all_input_source_snapshot_output_sha_unchanged") is True
                and seal["params"]["through_frame"] == stop, "successful fixed phase required")
        for item in seal["outputs"].values():
            path = within(item["path"])
            require(path.is_relative_to(root) and path.stat().st_size == item["size"],
                    "output scope/size differs")
        metadata = read(root / "checkpoint/metadata.json")
        require(metadata["processed_frames"] == stop-FIRST
                and metadata["previous_frame"] == stop-1 and metadata["initialize_count"] == 1,
                "checkpoint native lifecycle differs")
        manifest = read(root / "checkpoint/manifest.json")
        require(len(manifest["arrays"]) == 39
                and manifest["arrays"]["state_pose"]["shape"] == [6, 3, 3, 3]
                and tuple(metadata["record_fields"]) == RECORD_FIELDS
                and tuple(metadata["state_fields"]) == STATE_FIELDS,
                "complete ordered beam/checkpoint schema required")
        rows.append((seal, metadata))
    binding = rows[0][0]["params"]["bindings"]
    phase.equal(binding, rows[1][0]["params"]["bindings"], "original phase bindings differ")
    phase.equal(rows[0][1]["context"], rows[1][1]["context"], "original contexts differ")
    require(Path(rows[1][0]["params"]["resume_phase"]).resolve() == parent
            and rows[1][1]["restore_count"] == rows[0][1]["restore_count"]+1,
            "reference is not the next restore of this parent")
    for path, digest in binding["code_sha256"].items():
        require(sha256(within(path)) == digest, "frozen original source changed")
    protocol_path = LAB / "long_tracking_protocol.json"
    if evidence is not None:
        evidence.add(protocol_path, binding["tracking_protocol_sha256"])
    require(sha256(protocol_path) == binding["tracking_protocol_sha256"], "protocol differs")
    protocol = read(protocol_path)
    phase.equal(asdict(FlowConfig()), protocol["flow_config"], "flow settings differ")
    return dict(
        schema="chest_deficit_replay_preflight_v1", status="metadata_preflight_only",
        parent=str(parent), reference=str(reference), native_interval=[START, STOP],
        diagnostic_interval=list(DIAGNOSTIC_RANGE), original_binding=binding,
        original_context=rows[0][1]["context"], record_fields=list(RECORD_FIELDS),
        state_fields=list(STATE_FIELDS), all_ordered_beams=6,
        expected_restore_count=rows[0][1]["restore_count"]+1,
        runtime_python=sys.version, cuda_queried=False, tracker_constructed=False,
        payload_hashes_verified=False, mask_semantics_verified=False, raw_decoded_frames=0,
        checkpoint_payload_bytes=[(p / "checkpoint/arrays.npz").stat().st_size
                                  for p in (parent, reference)],
        full_validation_required_before_replay=True,
        reference_future_history_used_only_for_validation=True,
        C34_or_human_truth_input=False, production_outputs_changed=False)


def verify_previous_observation(parent, inputs):
    """Match checkpoint buffers to bound native SHA, packed D2, background and SL.

    The raw buffer is checked against its producer's native BGR SHA, without
    another MP4 decode. This does not introduce a replacement previous frame.
    """
    native = START-1
    root = inputs.chunk_roots[(native-FIRST)//CHUNK]
    manifest = read(root / "manifest.json")
    index = native-manifest["frames"][0]
    nodes = CachedNodes(inputs.input_root, inputs.protocol["experiment"], [native])
    result = {}
    for view in VIEWS:
        image = parent["arrays"]["previous_images_bgr_"+view]
        require(image.dtype == np.uint8 and image.shape == (1200, 1200, 3),
                "previous raw shape/dtype differs")
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        require(digest == manifest["raw"][view]["frame_bgr_sha256"][index],
                "previous observation native BGR differs")
        with np.load(root / manifest["masks"][view]["path"], allow_pickle=False) as z:
            packed = {key: z[key] for key in z.files}
        mask = decode_frame_masks(packed, index) if packed["view_available"][index] else None
        values = dict(detections=nodes.at(view, native), masks=mask,
                      backgrounds_bgr=cv2.imread(str(root / manifest["backgrounds"][view]["path"])))
        for field, value in values.items():
            key = "previous_"+field+"_"+view
            require(parent["metadata"]["previous_observation"][key] == (value is not None),
                    "previous buffer availability differs")
            require((value is None and key not in parent["arrays"])
                    or (value is not None and byte_equal(parent["arrays"][key], value)),
                    "previous buffer content differs: "+key)
        result[view] = dict(native_frame=native, raw_bgr_sha256=digest,
                            masks_SL_background_exact=True, raw_video_redecoded=False)
    return result


def prepare(evidence, store):
    """Original full validation, with separately evidenced semantic receipts."""
    parent_root, parent_seal = phase.sealed(PARENT, evidence)
    reference_root, reference_seal = phase.sealed(REFERENCE, evidence)
    binding = parent_seal["params"]["bindings"]
    phase.equal(binding, reference_seal["params"]["bindings"], "reference binding differs")
    inputs, protocol = verified_inputs(binding, REFERENCE_STOP, evidence, receipt_store=store)
    parent = phase.load_phase(parent_root, parent_seal, inputs, evidence, protocol)
    reference = phase.load_phase(reference_root, reference_seal, inputs, evidence, protocol)
    recurrence = phase.suffix_history(parent, reference)
    previous = verify_previous_observation(parent, inputs)
    store.final_guard()
    for path, digest in store.evidence_hashes.items():
        evidence.add(path, digest)
    root = inputs.chunk_roots[(START-FIRST)//CHUNK]
    reader = SourceSequenceReader(root, inputs.input_root)
    require(reader.frames[0] == START and reader.frames[-1]+1 == START+CHUNK,
            "replay requires unchanged original next chunk")
    return dict(inputs=inputs, protocol=protocol, binding=binding, parent=parent,
                reference=reference, reader=reader, original_suffix_audit=recurrence,
                previous_observation=previous)


def runtime(binding, *, gpu):
    result = dict(numpy=np.__version__, scipy=scipy.__version__, opencv=cv2.__version__,
                  torch=torch.__version__, cuda=torch.version.cuda)
    for key, value in result.items():
        require(value == binding["package_versions"][key], "runtime differs: "+key)
    require(cv2.getNumThreads() == binding["opencv_threads"]
            and torch.get_num_threads() == binding["torch_cpu_threads"]
            and torch.are_deterministic_algorithms_enabled()
            == binding["torch_deterministic_algorithms"]
            and torch.backends.cuda.matmul.allow_tf32 == binding["cuda_matmul_tf32"]
            and torch.backends.cudnn.allow_tf32 == binding["cudnn_tf32"]
            and torch.backends.cudnn.benchmark == binding["cudnn_benchmark"],
            "original runtime execution flags differ")
    if gpu:
        require(torch.cuda.is_available(), "CUDA runtime unavailable")
        result["cuda_device_name"] = torch.cuda.get_device_name(0)
        phase.equal(result, binding["package_versions"], "CUDA device/runtime differs")
    return dict(package_versions=result, cuda_queried=gpu,
                torch_cpu_threads=torch.get_num_threads(), opencv_threads=cv2.getNumThreads())


def compare_records(records, expected, native):
    """All beams/fields/order; Unicode scalar storage width is not a label value."""
    offset = native-FIRST
    lo, hi = expected["history_offsets"][offset:offset+2]
    errors = []
    if len(records) != hi-lo:
        errors.append("beam_count")
    else:
        for field in RECORD_FIELDS:
            saved = expected["history_"+field][lo:hi]
            actual = np.asarray([r[field] for r in records])
            if saved.dtype.kind == actual.dtype.kind == "U":
                matches = actual.shape == saved.shape and np.array_equal(actual, saved)
            else:
                matches = byte_equal(actual, saved)
            if not matches:
                errors.append(field)
    return dict(frame_idx=native, reference_rows=[int(lo), int(hi)],
                all_beams=len(records), compared_fields=list(RECORD_FIELDS),
                exact=not errors, differing_fields=errors)


def bounded_frames(reader, *, first=START, stop=STOP):
    """Keep full reader scope; never request stop or decode a preceding prefix."""
    require(first < stop and reader.frames[0] <= first <= stop <= reader.frames[-1]+1,
            "bounded native interval outside original reader")
    iterator = iter_from(reader, first)
    try:
        for native in range(first, stop):
            frame = next(iterator)
            require(frame.frame_idx == native, "replay reader skipped/reordered native frame")
            yield frame
    finally:
        iterator.close()


def save_diagnostics(output, records):
    for row in records:
        root = output / "diagnostics" / f"frame_{row['frame_idx']:06d}"
        if root.exists():
            continue
        root.mkdir(parents=True)
        np.savez_compressed(root / "arrays.npz", **row["arrays"])
        save_json(root / "metadata.json", row["metadata"])


def run_replay(context, output, window, evidence, code, store):
    binding, protocol, parent = context["binding"], context["protocol"], context["parent"]
    torch.set_num_threads(binding["torch_cpu_threads"])
    before = runtime(binding, gpu=False)
    iterator = bounded_frames(context["reader"])
    tracker = diagnostic = None
    matches = []
    started = time.perf_counter()
    try:
        first_frame = next(iterator)  # 3 native BGRs verified before waiting; no CUDA fit.
        # Construction queries availability but allocates no CUDA fit tensors.
        # Full-history restore is CPU work and belongs before the short GPU window.
        actual_runtime = runtime(binding, gpu=True)
        tracker = AcceleratedContinuousPersistentNodeTracker(
            context["inputs"].rig, context["inputs"].initial_pose_xyz.copy(),
            TrackerConfig(**protocol["tracker_config"]),
            recovery_config=NodeRecoveryConfig(**protocol["node_recovery_config"]),
            persistence_config=PersistentNodeConfig(**protocol["persistence_config"]))
        tracker.restore_state(parent["arrays"], parent["metadata"])
        phase.equal(tracker._context(), parent["metadata"]["context"], "restored context differs")
        require(tracker.previous_frame == START-1 and tracker.processed_frames == START-FIRST
                and len(tracker._states) == 6, "restored full state boundary differs")
        save_json(output / "ready.json", dict(native_frame=first_frame.frame_idx,
                  prepared_seconds=time.perf_counter()-started, runtime=actual_runtime,
                  CPU_runtime_before_CUDA_query=before, cuda_fit_calls=0,
                  tracker_restored=True, reader_full_native_interval=[START, START+CHUNK],
                  raw_decoded_frames=1))
        evidence.final_guard()
        unchanged(code)
        store.final_guard()
        print(json.dumps({"event": "cpu_prepared_waiting_for_gpu_window",
                          "output": str(output)}), flush=True)
        window.wait_until_open()
        # Only small code hashes after the wait. Full raw hashes are checked again
        # in the final guard; every subsequently decoded frame has its own SHA.
        unchanged(code)
        runtime(binding, gpu=True)
        window.assert_open()
        fit_started = time.perf_counter()
        with CandidateDiagnostics(tracker, frame_range=DIAGNOSTIC_RANGE) as diagnostic:
            frame = first_frame
            for native in range(START, STOP):
                if native != START:
                    frame = next(iterator)
                window.assert_open()
                with diagnostic.frame(native):
                    tracker.advance(frame)
                torch.cuda.synchronize()
                window.assert_open()
                comparison = compare_records(
                    tracker._history[-1], context["reference"]["arrays"], native)
                matches.append(comparison)
                if not comparison["exact"]:
                    save_json(output / "first_difference.json", comparison)
                    raise ValueError("original all-beam replay diverged at native "+str(native))
                if (native-START+1) % 25 == 0 or native == STOP-1:
                    print(json.dumps({"event": "replay_progress", "frame": native,
                                      "exact_frames": len(matches)}), flush=True)
            window.assert_open()
        fit_seconds = time.perf_counter()-fit_started
        save_json(output / "fit_completed.json", dict(last_native=tracker.previous_frame,
                  tracking_seconds=fit_seconds, all_frames_exact=True, runtime=actual_runtime))
        require([r["frame_idx"] for r in diagnostic.records] == list(range(*DIAGNOSTIC_RANGE)),
                "diagnostic target coverage differs")
        return dict(exact_frames=len(matches), diagnostic_frames=len(diagnostic.records),
                    tracking_seconds=fit_seconds, final_previous_frame=tracker.previous_frame,
                    logical_initialization_count=tracker.initialize_count,
                    checkpoint_restore_count=tracker.restore_count, external_pose_reset_count=0)
    finally:
        iterator.close()
        if diagnostic is not None:
            save_diagnostics(output, diagnostic.records)
        save_json(output / "history_comparison.json", dict(
            rows=matches, reference=str(REFERENCE), fields=list(RECORD_FIELDS),
            first_difference=next((r for r in matches if not r["exact"]), None),
            final_winner_prefix_used_as_replay_criterion=False,
            numeric_bytes_exact=True, Unicode_labels_exact_ignoring_array_capacity=True))
        if tracker is not None and not tracker._failed:
            arrays, metadata = tracker.export_state()
            prefix = START-FIRST
            for key in ("history_offsets", *("history_"+k for k in RECORD_FIELDS)):
                end = (prefix+1 if key == "history_offsets"
                       else parent["arrays"]["history_offsets"][-1])
                value, prior = arrays[key][:end], parent["arrays"][key]
                exact = (
                    value.shape == prior.shape and np.array_equal(value, prior)
                    if value.dtype.kind == prior.dtype.kind == "U"
                    else byte_equal(value, prior))
                require(exact,
                        "parent retained history changed: "+key)
            checkpoint_binding = dict(
                schema="chest_deficit_diagnostic_checkpoint_binding_v1",
                original_fixed_binding=binding,
                experiment_plan_sha256=sha256(output / "registration/plan.json"))
            write_checkpoint(output / "checkpoint", arrays, metadata, bindings=checkpoint_binding)
            # The diagnostic checkpoint is deliberately NOT a production continuation.
            saved, saved_meta, _ = read_checkpoint(
                output / "checkpoint", expected_bindings=checkpoint_binding)
            require(all(byte_equal(saved[k], arrays[k]) for k in arrays) and saved_meta == metadata,
                    "diagnostic checkpoint roundtrip differs")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--receipt-root", type=Path)
    parser.add_argument("--supervisor-root", type=Path)
    parser.add_argument("--child-pid", type=int)
    parser.add_argument("--timeout-seconds", type=float, default=7200)
    args = parser.parse_args(argv)
    require(not args.execute or (args.receipt_root is not None and args.supervisor_root is not None
                                and args.child_pid is not None),
            "execute requires receipt cache and ROOT-owned child/supervisor window")
    output = within(args.output, exists=False)
    require(not output.exists(), "new diagnostic output required")
    code = code_hashes()
    output.mkdir(parents=True)
    evidence = phase.Digests()
    window = store = None
    summary = dict(schema=STAGE, status="incomplete", replay_complete=False,
                   native_interval=[START, STOP], diagnostic_interval=list(DIAGNOSTIC_RANGE),
                   C34_or_human_truth_input=False, production_outputs_changed=False,
                   full_sequence_complete=False, independent_ID_truth=False)
    with provenance(STAGE, output, name="replay") as record:
        try:
            for path, digest in code.items():
                evidence.add(path, digest)
            snapshot(code, output, record)
            if args.execute:
                window = ReplayGPUWindow(args.supervisor_root, args.child_pid, output,
                                         timeout_seconds=args.timeout_seconds)
                window.attach()  # Retain currently LIVE owned child before expensive CPU I/O.
            preflight = static_preflight(evidence=evidence)
            save_json(output / "preflight.json", preflight)
            if args.preflight_only:
                summary.update(status="metadata_preflight_only", cuda_queried=False,
                               payload_hashes_verified=False, raw_decoded_frames=0)
            else:
                store = ReceiptStore(within(args.receipt_root, exists=False))
                start = time.perf_counter()
                context = prepare(evidence, store)
                registration = output / "registration"
                registration.mkdir()
                plan = dict(preflight, schema="chest_deficit_replay_registration_v1",
                            payload_hashes_verified=True, code_hashes=code,
                            input_hashes=dict(evidence.paths), receipt_uses=store.uses,
                            expected_original_record_fields=list(RECORD_FIELDS),
                            previous_observation=context["previous_observation"],
                            original_reference_suffix_audit=context["original_suffix_audit"],
                            preparation_seconds=time.perf_counter()-start,
                            no_reinitialization_from_reference_geometry=True,
                            fail_on_first_all_beam_difference=True)
                with provenance("register_chest_deficit_replay_v1", registration,
                                name="registration") as reg:
                    save_json(registration / "plan.json", plan)
                    for path, digest in evidence.paths.items():
                        reg.input(path, path, sha256=digest)
                    reg.output("plan", registration / "plan.json",
                               sha256=sha256(registration / "plan.json"))
                    unchanged(code)
                    reg.param("inputs_verified_by_immutable_loader_and_phase_audit", True)
                evidence.add(registration / "registration.provenance.json")
                evidence.add(registration / "plan.json")
                summary.update(run_replay(context, output, window, evidence, code, store))
                summary.update(status="original_all_beam_replay_exact", replay_complete=True,
                               preparation_seconds=plan["preparation_seconds"],
                               payload_hashes_verified=True, raw_decoded_frames=STOP-START)
        finally:
            if window is not None:
                window.close()
                summary["gpu_window"] = window.report
            save_json(output / "summary.json", summary)
            if store is not None:
                save_json(output / "mask_validation_receipts.json", store.uses)
            for path, digest in evidence.paths.items():
                record.input(path, path, sha256=digest)
            outputs = {str(p): sha256(p) for p in output.rglob("*") if p.is_file()
                       and p != output / "replay.provenance.json"}
            for path, digest in outputs.items():
                record.output(str(Path(path).relative_to(output)), path, sha256=digest)
            unchanged(code)
            evidence.final_guard()
            if store is not None:
                store.final_guard()
            unchanged(outputs)
            record.param("all_input_source_snapshot_output_sha_unchanged", True)
            record.param("original_production_binding_unchanged", True)
            record.param("experimental_sources_not_added_to_production_binding", True)
    print(json.dumps(summary), flush=True)
    return summary


if __name__ == "__main__":
    main()
