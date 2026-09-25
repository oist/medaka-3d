"""CPU audit of the sealed four-chunk continuation to native frame59760.

No tracker, detector, GPU, GT, or new uninterrupted reference run. The immutable
support auditor, checkpoint codec, and exact backtrace utilities are reused.
History/lifecycle adapters remove old audit-window constants without monkeypatches.
Raw BGR is independently decoded at 15 fixed native frames only; support is
recomputed at ALL9960 frames. The producer/loader's full checks are separate.
"""

import argparse
import dataclasses
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import audit_long_continuation_boundary as prior
import numpy as np

from medaka3d.provenance import provenance

COMMON, BASE, REPO, LAB, VIEWS = prior.COMMON, prior.BASE, prior.REPO, prior.LAB, prior.VIEWS
FIRST, SPLIT, STOP, FULL_STOP = 49800, 53120, 59760, 348600
BOUNDARIES = (53120, 54780, 56440, 58100)
SAMPLED_FRAMES = tuple(sorted({f for b in BOUNDARIES for f in (b-1, b, b+1)}
                              | {STOP-3, STOP-2, STOP-1}))
PARENT = REPO / "work/three_body/long_continuation_through53120_v1"
RUN = REPO / "work/three_body/long_continuation_through59760_v1"
PLAN = REPO / "work/three_body/long_multichunk_minute_plan_v1"
PLAN_SHA = "1635bc068709df8ef42c4268fde262327a20bfe726e40b8299e13b03d6c362f7"
PINNED_HELPERS = {
    "audit_long_continuation_boundary.py":
        "e11dfbbe73be6673de88dfb8752a474601a7a2467e94e37e063b1ec197899120",
    "audit_continuation_comparison.py":
        "c7355c194cc9daba79492bd9bbf00a338c24ee34e5a4c3c50d3439e2260e73f4",
    "audit_long_continuation_support.py":
        "cd79f2582919504ef1e625b5968997ef03a98f0e7f87d7f6e5957695d368b41c",
    "audit_persistent_minute.py":
        "31d5d9796c5ba10e7f211381c8b37a5fe2dbb3367ff2e829b18b3ebd4f94d0bc",
}
require, equal = prior.require, prior.json_equal


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def completed_seals(evidence):
    """Check BOTH success seals before any candidate/checkpoint array is opened."""
    result = []
    for root in (PARENT, RUN):
        path = root / "continuation.provenance.json"
        evidence.add(path)
        seal = COMMON.read_json(path)
        require(seal.get("status") == "ok"
                and seal.get("stage") == "three_body_long_continuation"
                and seal["params"]["all_input_source_snapshot_output_sha_unchanged"] is True,
                "both tracking phases must already be successfully sealed")
        result.append(seal)
    return result


def verify_plan(seals, evidence):
    evidence.add(PLAN / "plan.json", PLAN_SHA)
    evidence.add(PLAN / "registration.provenance.json")
    plan = COMMON.read_json(PLAN / "plan.json")
    registration = COMMON.read_json(PLAN / "registration.provenance.json")
    require(registration["status"] == "ok"
            and registration["stage"] == "register_long_multichunk_execution",
            "multichunk preregistration is incomplete")
    for item in (*registration["inputs"].values(), *registration["outputs"].values()):
        evidence.add(item["path"], item["sha256"])
    require(Path(registration["outputs"]["plan"]["path"]).resolve() == PLAN / "plan.json"
            and registration["outputs"]["plan"]["sha256"] == PLAN_SHA,
            "registration plan output binding differs")
    require(datetime.fromisoformat(registration["finished_at"])
            <= datetime.fromisoformat(seals[1]["started_at"]), "plan saved after tracking began")
    for key, expected in {
        "schema": "three_body_long_multichunk_execution_plan_v1",
        "full_target_interval": [FIRST, FULL_STOP], "parent_processed_interval": [FIRST, SPLIT],
        "next_native_interval": [SPLIT, STOP], "output_interval": [FIRST, STOP],
        "new_input_intervals": [[f, f+1660] for f in BOUNDARIES],
        "logical_initializations_expected": 1, "checkpoint_restores_expected": 2,
        "external_pose_resets_expected": 0, "parent_history_rows_expected": 19913,
    }.items():
        equal(plan[key], expected, "registered value differs: " + key)
    require(Path(plan["parent_phase"]).resolve() == PARENT, "registered parent differs")
    binding = seals[1]["params"]["bindings"]
    equal(binding, seals[0]["params"]["bindings"], "parent/current binding changed")
    equal(binding, plan["frozen_input_and_code_binding"], "registered binding changed")
    require(Path(seals[1]["params"]["resume_phase"]).resolve() == PARENT
            and seals[0]["params"]["through_frame"] == SPLIT
            and seals[1]["params"]["through_frame"] == STOP, "phase endpoint differs")
    return binding


def load_phase(root, seal, inputs, evidence, *, stop, phase_start, restores, chunks):
    """Exact memberships and checkpoint/backtrace checks; no tracker is constructed."""
    binding = seal["params"]["bindings"]
    later_roots = inputs.chunk_roots[chunks:]
    expected = {p: d for p, d in inputs.verified_hashes.items()
                if not any(Path(p).is_relative_to(r) for r in later_roots)}
    expected.update(binding["code_sha256"])
    parent_path = Path(seal["params"]["resume_phase"]).resolve(strict=True)
    expected.update({str(parent_path / f): BASE.sha256(parent_path / f)
                     for f in prior.PARENT_FILES})
    require(set(expected) == set(seal["inputs"]), "phase input membership differs")
    for name, digest in expected.items():
        item = seal["inputs"][name]
        require(Path(item["path"]).resolve() == Path(name) and item["sha256"] == digest,
                "phase input role/path/SHA differs")
        evidence.add(name, digest)
    for item in seal["outputs"].values():
        path = Path(item["path"]).resolve(strict=True)
        require(path.is_relative_to(root) and path.stat().st_size == item["size"],
                "output escaped run or changed size")
        evidence.add(path, item["sha256"])
    for role, filename in {"candidate": "candidate.npz", "summary": "summary.json",
                           "progress": "progress.jsonl", "consumed_chunks": "consumed_chunks.json",
                           "checkpoint_manifest": "checkpoint/manifest.json"}.items():
        require(Path(seal["outputs"][role]["path"]).resolve() == root / filename,
                "named output binding differs")
    for filename in ("checkpoint/arrays.npz", "checkpoint/metadata.json"):
        require(str(root / filename) in seal["outputs"], "unsealed checkpoint payload")
    for source, digest in binding["code_sha256"].items():
        path = root / "code_snapshot" / Path(source).relative_to(REPO)
        require(str(path) in seal["outputs"], "unsealed producer source snapshot")
        evidence.add(path, digest)
    arrays, metadata, manifest = prior.read_checkpoint(
        root / "checkpoint", expected_bindings=binding)
    summary = COMMON.read_json(root / "summary.json")
    for key, value in dict(
        schema="long_continuation_phase_v1", technical_completion=True,
        full_sequence_complete=False, prefix_backtrace_provisional=True,
        registered_interval=[FIRST, FULL_STOP], processed_interval=[FIRST, stop],
        processed_total=stop-FIRST, phase_frames=[phase_start, stop],
        phase_processed=stop-phase_start, consumed_chunk_count=chunks,
        logical_initialization_count=1, checkpoint_restore_count=restores,
        external_pose_reset_count=0, temporal_subsampling=False, human_annotation_input=False,
        post_initialization_baseline_coordinates_input=False,
    ).items():
        equal(summary.get(key), value, "phase summary differs: " + key)
    require(len(arrays) == 39 and metadata["processed_frames"] == stop-FIRST
            and metadata["previous_frame"] == stop-1 and metadata["initialize_count"] == 1
            and metadata["restore_count"] == restores
            and metadata["node_recovery_diagnostics"] == summary["node_recovery_diagnostics"]
            and manifest["payload"]["bytes"] == summary["checkpoint_payload_bytes"],
            "checkpoint lifecycle or array count differs")
    consumed = COMMON.read_json(root / "consumed_chunks.json")
    equal(consumed, inputs.chunk_records[:chunks], "consumed chunk evidence changed")
    progress = [json.loads(line) for line in (root / "progress.jsonl").read_text().splitlines()]
    counts = [r["processed"] for r in progress]
    require(counts and counts == sorted(set(counts)) and counts[-1] == stop-FIRST
            and all(r["source_frame"] == FIRST+r["processed"]-1
                    and r["phase_processed"] == r["processed"]-(phase_start-FIRST)
                    and r["phase_processed"] > 0 for r in progress), "native progress differs")
    require({c["stop"]-FIRST for c in consumed if c["stop"] > phase_start}.issubset(counts),
            "progress lacks completed chunk endpoints")
    candidate = read_npz(root / "candidate.npz")
    traced = COMMON.traced_candidate(arrays)
    require(set(candidate) == set(traced) and len(candidate) == 21, "candidate schema differs")
    comparisons = {k: COMMON.same_array(candidate[k], traced[k], k) for k in candidate}
    COMMON.same_array(candidate["frame_idx"], np.arange(FIRST, stop, dtype=np.int64),
                      "native frames")
    return dict(arrays=arrays, metadata=metadata, summary=summary, consumed=consumed,
                candidate=candidate, comparisons=comparisons)


def history_replay(parent, current, initial):
    """The previous independent recurrence, generalized only by native range.

    Replay from the registered initial state, retain the actual ordered beam at
    each reader boundary, and compare the parent checkpoint at its exact split.
    """
    a, metadata = current["arrays"], current["metadata"]
    offsets = a["history_offsets"]
    require(offsets.dtype == np.int64 and offsets.shape == (STOP-FIRST+1,) and offsets[0] == 0
            and np.all((np.diff(offsets) >= 1) & (np.diff(offsets) <= 6))
            and metadata["record_fields"] == list(COMMON.HISTORY_FIELDS)
            and metadata["state_fields"] == list(COMMON.STATE_FIELDS), "history schema differs")
    old = parent["arrays"]
    COMMON.same_array(offsets[:SPLIT-FIRST+1], old["history_offsets"], "parent offsets")
    require(old["history_offsets"][-1] == 19913 and len(old["state_pose"]) == 6,
            "registered parent history or beam differs")
    prefix = {}
    for key in COMMON.HISTORY_FIELDS:
        require(len(a["history_"+key]) == offsets[-1], "unaccounted history rows")
        prefix[key] = COMMON.same_array(a["history_"+key][:19913], old["history_"+key], key)
    expected_keys = ({"history_offsets"} | {"history_"+k for k in COMMON.HISTORY_FIELDS}
                     | {"state_"+k for k in COMMON.STATE_FIELDS}
                     | {k for k, present in metadata["previous_observation"].items() if present})
    require(set(a) == expected_keys, "unexpected checkpoint array fields")
    previous = [dict(pose=initial.copy(), velocity=np.zeros((3, 3)), missing=np.zeros(3, int),
                     review=np.zeros(3, bool), cost=0., node_missing=np.zeros((3, 3), int),
                     direct_pectoral_age=np.zeros(3, int))]
    states_at_boundary, eligible, nonzero = {}, 0, 0
    for t in range(STOP-FIRST):
        native = FIRST+t
        if native in BOUNDARIES:
            require(len(previous) == 6, "reader boundary lost an active parent")
            states_at_boundary[native] = {k: np.asarray([p[k] for p in previous]).copy()
                                           for k in COMMON.STATE_FIELDS}
            if native == SPLIT:
                for k in COMMON.STATE_FIELDS:
                    COMMON.same_array(states_at_boundary[native][k], old["state_"+k], k)
                require(eligible == parent["metadata"]["node_recovery_diagnostics"]
                        ["calls_with_age_eligible_fish"], "parent eligible count differs")
        eligible += bool(np.any((previous[0]["node_missing"][:, 0] >= 4)
                                & (previous[0]["direct_pectoral_age"] >= 4)))
        following = []
        for row in range(int(offsets[t]), int(offsets[t+1])):
            r = {k: a["history_"+k][row] for k in COMMON.HISTORY_FIELDS}
            pi = int(r["parent"])
            require(0 <= pi < len(previous) and r["frame_idx"] == native, "parent/frame differs")
            nonzero += pi != 0
            p, pose, sources, held = previous[pi], r["pose_xyz"], r["sources"], r["observed"]
            require(np.isfinite(pose).all() and np.all(pose >= [0, 0, -6])
                    and np.all(pose <= [192, 192, 186]) and np.all(sources >= -1),
                    "history engineering pose/source contract differs")
            COMMON.same_array(held, (sources >= 0).sum(1) >= 2, "observed counts")
            for view in range(3):
                for node in range(3):
                    used = sources[:, view, node]
                    used = used[used >= 0]
                    require(len(used) == len(np.unique(used)), "source exclusivity violation")
            body = held[:, 1] | (r["mask_supported"] & (sources[:, :, :2] >= 0).any((1, 2)))
            missing = np.where(body, 0, p["missing"]+1)
            age = np.where(held, 0, p["node_missing"]+1)
            direct_age = np.where(held[:, 1], p["direct_pectoral_age"]+1, 0)
            for key, value in (("body_supported", body), ("missing", missing),
                               ("node_missing", age), ("direct_pectoral_age", direct_age)):
                COMMON.same_array(r[key], value, "parent recurrence " + key)
            require(not np.any(p["review"] & ~r["identity_review"])
                    and not np.any((missing >= 4) & ~r["identity_review"]), "sticky review erased")
            cost = p["cost"]+r["costs"]
            COMMON.same_array(np.asarray(cost), np.asarray(r["path_costs"]), "path cost")
            velocity = .5*p["velocity"] + .5*(pose[:, 1]-p["pose"][:, 1])
            norm = np.linalg.norm(velocity, axis=-1, keepdims=True)
            velocity *= np.minimum(1., 2./np.maximum(norm, 1e-12))
            velocity = np.where(body[:, None], velocity, .8*p["velocity"])
            following.append(dict(pose=pose, velocity=velocity, missing=missing,
                                  review=r["identity_review"], cost=float(cost), node_missing=age,
                                  direct_pectoral_age=direct_age))
        previous = following
    for k in COMMON.STATE_FIELDS:
        COMMON.same_array(a["state_"+k], np.asarray([p[k] for p in previous]), "final " + k)
    require(eligible == metadata["node_recovery_diagnostics"]["calls_with_age_eligible_fish"]
            and current["summary"]["retained_history_rows"] == offsets[-1]
            and current["summary"]["active_beam_count"] == len(previous), "history counters differ")
    return dict(old_ordered_records_retained=19913, old_history_fields_exact=prefix,
                all_parent_recurrences_exact=True, active_beam_count=len(previous),
                retained_history_rows=int(offsets[-1]), nonzero_parent_rows=nonzero,
                independent_age_eligible_calls=eligible), states_at_boundary


def sample_native(inputs, parent, current, states, protocol, binding):
    """Decode only registered samples; original MP4 is opened once per view.

    The immutable loader has verified every chunk/mask schema and decoded mask
    SHA already. Use its manifests here without reinitializing a SourceReader
    (which would repeat large full-video hashes for every sampled chunk).
    """
    cv2 = prior.cv2
    manifests = [COMMON.read_json(root / "manifest.json") for root in inputs.chunk_roots]
    packed = {}
    for ci in {int((f-FIRST)//1660) for f in SAMPLED_FRAMES}:
        packed[ci] = {v: read_npz(inputs.chunk_roots[ci] / manifests[ci]["masks"][v]["path"])
                      for v in VIEWS}
    nodes = prior.CachedNodes(inputs.input_root, inputs.protocol["experiment"], inputs.frame_idx)
    backgrounds = {
        v: cv2.imread(str(inputs.chunk_roots[0] / manifests[0]["backgrounds"][v]["path"]))
        for v in VIEWS}
    require(all(x is not None and x.shape == (1200, 1200, 3) and x.dtype == np.uint8
                for x in backgrounds.values()), "background decode shape differs")
    observations = {f: prior.FrameObservation(f, {v: nodes.at(v, f) for v in VIEWS}, {}, {},
                                              backgrounds) for f in SAMPLED_FRAMES}
    output = dict(frame_idx=np.array(SAMPLED_FRAMES, np.int64),
                  raw_bgr_sha256=np.empty((len(SAMPLED_FRAMES), 3), "S64"),
                  mask_decoded_sha256=np.empty((len(SAMPLED_FRAMES), 3), "S64"),
                  sleap_points_sha256=np.empty((len(SAMPLED_FRAMES), 3), "S64"))
    for vi, view in enumerate(VIEWS):
        root, manifest = inputs.chunk_roots[0], manifests[0]
        cap = cv2.VideoCapture(str(root / manifest["raw"][view]["path"]))
        try:
            require(cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) == FULL_STOP
                    and np.isclose(cap.get(cv2.CAP_PROP_FPS), 166, atol=1e-6, rtol=0)
                    and cap.get(cv2.CAP_PROP_FRAME_HEIGHT) == 1200
                    and cap.get(cv2.CAP_PROP_FRAME_WIDTH) == 1200, "source metadata differs")
            for i, f in enumerate(SAMPLED_FRAMES):
                ci, t = divmod(f-FIRST, 1660)
                require(cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                        and cap.get(cv2.CAP_PROP_POS_FRAMES) == f, "native seek differs")
                ok, image = cap.read()
                require(ok and image is not None and image.shape == (1200, 1200, 3)
                        and image.dtype == np.uint8 and cap.get(cv2.CAP_PROP_POS_FRAMES) == f+1,
                        "native image decode/index differs")
                image_hash = COMMON.digest_array(image)["sha256"]
                require(image_hash == manifests[ci]["raw"][view]["frame_bgr_sha256"][t],
                        "sample native BGR SHA differs")
                data = packed[ci][view]
                decoded = prior.decode_frame_masks(data, t)
                mask_hash = COMMON.digest_array(decoded)["sha256"]
                require(mask_hash == bytes(data["frame_mask_sha256"][t]).decode(),
                        "sample decoded mask SHA differs")
                available = bool(data["view_available"][t])
                require(available or len(decoded) == 0, "unavailable view contains masks")
                observations[f].images_bgr[view] = image
                observations[f].masks[view] = decoded if available else None
                output["raw_bgr_sha256"][i, vi] = image_hash
                output["mask_decoded_sha256"][i, vi] = mask_hash
                output["sleap_points_sha256"][i, vi] = COMMON.digest_array(
                    observations[f].detections[view])["sha256"]
        finally:
            cap.release()
    payload_checks = {}
    for phase in (parent, current):
        frame = observations[phase["metadata"]["previous_frame"]]
        for category in ("detections", "images_bgr", "masks", "backgrounds_bgr"):
            for view in VIEWS:
                key, value = "previous_"+category+"_"+view, getattr(frame, category)[view]
                require(phase["metadata"]["previous_observation"][key] == (value is not None),
                        "previous observation availability differs")
                if value is not None:
                    payload_checks[f"{frame.frame_idx}:{key}"] = COMMON.same_array(
                        phase["arrays"][key], value, key)
                else:
                    require(key not in phase["arrays"], "invented previous observation")
    config = prior.FlowConfig()
    equal(dataclasses.asdict(config), protocol["flow_config"], "flow config differs")
    flow_info, old_threads = [], cv2.getNumThreads()
    cv2.setNumThreads(binding["opencv_threads"])
    try:
        for f in BOUNDARIES:
            flow = prior.FlowObservations.from_frames(observations[f-1], observations[f],
                                                      inputs.rig, states[f]["pose"][0], config)
            a = current["arrays"]
            left, right = a["history_offsets"][f-FIRST:f-FIRST+2]
            for r in range(int(left), int(right)):
                COMMON.same_array(a["history_proposal_flow_nodes"][r], flow.node_valid,
                                  "boundary flow nodes")
                require(a["history_proposal_flow_parent"][r] == 0, "flow parent reset")
            for field in dataclasses.fields(flow):
                output[f"f{f}_flow_{field.name}"] = np.asarray(getattr(flow, field.name))
            for key, value in states[f].items():
                output[f"f{f}_previous_state_{key}"] = value
            flow_info.append(dict(previous_frame=f-1, current_frame=f, parents=6,
                                  first_parent_flow_flags_exact=True,
                                  valid_nodes=int(flow.node_valid.sum())))
    finally:
        cv2.setNumThreads(old_threads)
    return dict(sampled_frames=list(SAMPLED_FRAMES), decoded_view_images=3*len(SAMPLED_FRAMES),
                sample_BGR_and_mask_SHA_exact=True, previous_checkpoint_payload=payload_checks,
                boundary_flow=flow_info, all_new_raw_images_independently_decoded=False,
                scope="15 fixed native samples (45 images); full mask validation is reused from "
                      "the immutable loader. Flow coordinates are recomputed diagnostics; only "
                      "flow valid flags were saved by tracking."), output, nodes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    require(not output.exists() and output.is_relative_to(REPO), "new workspace output required")
    evidence = prior.support_helpers.Digests()
    code = sorted({*COMMON.code_paths(), Path(__file__).resolve(), Path(prior.__file__).resolve(),
                   Path(prior.first.__file__).resolve(), Path(prior.support.__file__).resolve(),
                   Path(prior.support_helpers.__file__).resolve()})
    for path in code:
        evidence.add(path, PINNED_HELPERS.get(path.name))
    seals = completed_seals(evidence)
    binding = verify_plan(seals, evidence)
    inputs = prior.first.verified.load_long_inputs(
        Path(binding["inputs"]["seed"]["registration_path"]).parent,
        Path(binding["inputs"]["input_stage_root"]), Path(binding["inputs"]["seed"]["input_root"]),
        through_frame=STOP)
    equal(inputs.fixed_bindings, binding["inputs"], "verified input binding differs")
    require(np.array_equal(inputs.frame_idx, np.arange(FIRST, STOP))
            and len(inputs.chunk_roots) == 6 and inputs.full_frame_range == (FIRST, FULL_STOP),
            "six-chunk native range differs")
    for chunk in inputs.chunk_records:
        for link in chunk["raw_hardlinks"].values():
            require(os.path.samefile(link["path"], link["source_path"]), "raw hardlink differs")
            evidence.add(link["path"], link["sha256"])
    parent = load_phase(PARENT, seals[0], inputs, evidence,
                        stop=SPLIT, phase_start=51460, restores=1, chunks=2)
    current = load_phase(RUN, seals[1], inputs, evidence,
                         stop=STOP, phase_start=SPLIT, restores=2, chunks=6)
    prior.first.verified.verify_prefix_records(parent["consumed"], current["consumed"])
    context = current["metadata"]["context"]
    equal(context, parent["metadata"]["context"], "restored context differs")
    equal(context["initial_pose"], COMMON.digest_array(inputs.initial_pose_xyz),
          "initial pose differs")
    equal(context["cameras"], COMMON.normalize([dataclasses.asdict(c) for c in inputs.rig.cameras]),
          "calibration context differs")
    evidence.add(prior.first.PROTOCOL, prior.first.PROTOCOL_SHA)
    protocol = COMMON.read_json(prior.first.PROTOCOL)
    for a, b in (("tracker_config", "tracker_config"), ("recovery_config", "node_recovery_config"),
                 ("persistence_config", "persistence_config")):
        equal(context[a], protocol[b], "saved config differs")
    output.mkdir(parents=True, exist_ok=False)
    with provenance("long_multichunk_minute_independent_audit", output, name="audit") as record:
        started = time.perf_counter()
        for path in code:
            target = output / "code_snapshot" / path.relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            evidence.add(target, evidence.paths[str(path)])
        history, states = history_replay(parent, current, inputs.initial_pose_xyz)
        native, native_arrays, nodes = sample_native(
            inputs, parent, current, states, protocol, binding)
        geometry, support_arrays = prior.support.verify_all_support(current["candidate"], nodes,
                                                                    inputs.rig)
        counts = prior.support.direct_counts(current["candidate"])
        for key, summary_key in (
            ("direct_pectoral_fish_frames", "direct_pectoral_count"),
            ("accepted_pectoral_fish_frames", "accepted_pectoral_count"),
            ("review_fish_frames", "identity_review_count"),
            ("direct_pectoral_per_slot", "per_slot_direct_pectoral"),
            ("accepted_pectoral_per_slot", "per_slot_accepted_pectoral"),
        ):
            equal(counts[key], current["summary"][summary_key], "summary support counts differ")
        changes = {}
        for key, old in parent["candidate"].items():
            new = current["candidate"][key][:SPLIT-FIRST]
            changed = [i for i in range(len(old)) if old[i].tobytes() != new[i].tobytes()]
            changes[key] = dict(changed_frames=len(changed), first_changed_native_frame=
                                None if not changed else FIRST+changed[0])
        result = dict(schema="long_multichunk_minute_independent_audit_v1", status="ok",
                      parent=str(PARENT), run=str(RUN),
                      candidate_sha256=evidence.paths[str(RUN / "candidate.npz")],
                      frame_range=[FIRST, STOP], native_frames=STOP-FIRST,
                      new_interval=[SPLIT, STOP], new_native_frames=STOP-SPLIT,
                      logical_initialization_count=1, checkpoint_restore_count=2,
                      external_pose_reset_count=0, full_sequence_complete=False,
                      prefix_backtrace_provisional=True, all_39_checkpoint_arrays_verified=True,
                      all_21_candidate_arrays_exact_backtrace=current["comparisons"],
                      history=history, native_sample=native, all_source_geometry=geometry,
                      counts=counts, provisional_prefix_changes=changes,
                      phase_timing={k: current["summary"][k] for k in
                                    ("preparation_seconds", "reader_setup_seconds",
                                     "tracking_seconds", "checkpoint_seconds",
                                     "checkpoint_payload_bytes")},
                      independent_ID_truth=False, uninterrupted_counterfactual_run=False,
                      audit_compute_seconds=time.perf_counter()-started)
        np.savez_compressed(output / "native_evidence.npz", **native_arrays)
        np.savez_compressed(output / "native_support.npz", **support_arrays)
        (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False),
                                             encoding="utf-8")
        (output / "RESULT.md").write_text(
            "# 4チャンク接続・1分prefixの独立監査\n\n"
            "[49800,59760) の9960 native frames。親[49800,53120)から追加6640framesを同じ"
            "状態系列へ接続し、親の全19,913履歴行・全6状態、各reader境界の再帰、全39checkpoint"
            "配列と最終21配列のbacktraceを照合しました。初期化1、累積復元2、外部リセット0です。\n\n"
            "全9960frameの宣言sourceを2mm ray/12px投影で全数再計算しました。"
            f"胸部direct {counts['direct_pectoral_fish_frames']}/29880、"
            f"条件付き受理 {counts['accepted_pectoral_fish_frames']}/29880、"
            f"review {counts['review_fish_frames']}。"
            "数値は幾何支持・可用性でありID精度ではありません。\n\n"
            "独立raw再decodeは4境界の前後±1と末尾3frame、計15時点・3視点=45画像です。"
            "各BGR/mask SHA、親と最終checkpointのprevious観測、4境界のCPU flow valid flagsを"
            "照合しました。今回の追加19,920画像すべてを独立再decodeしたとは主張しません。"
            "全入力mask検証は不変loader、全raw処理は封印producerの保証と区別します。\n\n"
            "これは30分目標の暫定1分prefixです。新しい9960frame無中断対照は実行しておらず、"
            "将来情報で過去の最終選択が変わり得ます。初期対応は同じ楽観的登録seed、"
            "全3個体の独立人手真値は未入手で、ゼロswapを証明しません。\n", encoding="utf-8")
        for path, digest in evidence.paths.items():
            if not Path(path).is_relative_to(output):
                record.input(path, path, sha256=digest)
        outputs = {}
        for path in output.rglob("*"):
            if path.is_file():
                outputs[str(path)] = BASE.sha256(path)
                record.output(path.relative_to(output).as_posix(), path, sha256=outputs[str(path)])
        evidence.final_guard()
        COMMON.unchanged(outputs)
        record.param("all_input_source_snapshot_output_sha_unchanged", True)
        record.param("GPU_new_tracking_or_GT", False)
    print(json.dumps(dict(status="ok", output=str(output), counts=counts)))


if __name__ == "__main__":
    main()
