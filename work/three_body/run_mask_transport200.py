"""Fixed200 Original/LK/CT-first ownership tracking; new finite experiment.

Original coefficient0 must finish and match the old 21 candidate and 19 retained
fields byte-for-byte before either new loss arm starts. Reference numeric arrays
are opened only by that completed-arm gate. Every new arm starts from the same
registered seed, runs once without restore/reset, and constructs its own causal
t0 anchors. Frozen prepared pixels and saved point memberships are checked at
consumption. No new point model, detector, proposal generator or query exists.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import time
from pathlib import Path

import cv2
import mask_transport_inputs as inputs
import mask_transport_tracker as tracking
import numpy as np
import object_mask_transport as transport
import scipy
import temporal_point_core as point_core
import torch
import track_mask_transport200 as points
from continuous_tracker import RECORD_FIELDS
from node_recovery import NodeRecoveryConfig
from persistent_node_tracker import PersistentNodeConfig
from tracker import TrackerConfig

base, prepared = inputs.base, inputs.prepared
WORK, REPO, VIEWS = inputs.WORK, inputs.REPO, tuple(inputs.VIEWS)
FIRST, STOP, COUNT = 333519, 333719, 200
ARMS = ("original", "lk", "ct_first")
STAGE = "mask_transport200_three_arm_tracking_v1"
POINTS = WORK / "mask_transport200_point_tracking_v1"
POINTS_SHA = "05a6612f4870547a50f2866b39436c0fcf9a884dee58c041c529e6af584275b1"
SAMPLES = WORK / "mask_transport200_fixed_samples_v1"
SAMPLES_SHA = "dc87f87c31d2e92186e1d0cc69eba6bead66b729125ce188697ef9a4cb473c60"
REFERENCE = WORK / "continuation_transient_200_concurrent_v1"
REFERENCE_SHA = "163135a1f70c25bcd8251b5f5077c4b6eecc5b707e6df09d44466d4b8b5c05ce"
REFERENCE_CANDIDATE_SHA = "c36e9279317749224ec73b254e72d4282f65d1d818195f52e1374a6637a11447"
REFERENCE_MANIFEST_SHA = "9fe3c5f67ed9c5b6378caa4d38182c0c9dfaf8d800cd4701268ca9aa1466f657"
REFERENCE_ARRAYS_SHA = "56d7be65316a56bb09a625e6250da696cb103a66ec9770ae4b4037dda97f1d8d"
REFERENCE_METADATA_SHA = "7c089f5b3da8239bde679cb159c71866260f6fe8e290e7035bdfd7c2b154e7d1"
SOURCE_PINS = {
    "work/three_body/object_mask_transport.py": (
        "1fafd6a45793a1602e5160f1d0049d72cdc73bfe4a53a6a79bb6019f306920df"
    ),
    "work/three_body/track_mask_transport200.py": (
        "9a4c1ed3326309e2a8b05c56ce3f652cea3fdf9d1032449d149d66feba36080d"
    ),
    "work/three_body/mask_transport_inputs.py": (
        "eb579dfda7db4f1ab9cc175b72050f332663c3206ab5f77a9cec56cedb58a1fc"
    ),
    "work/three_body/mask_transport_tracker.py": (
        "3452e05e393d9995525a508d1d510fdb4b1faed6c74e5026a4c0c379ab936634"
    ),
    "work/three_body/mask_transport_fitter.py": (
        "51554ea7da2e1d35ed1ca3ade32b76a92894bb4acfe552f47493853f8c7fc232"
    ),
    "labs/20260909_three_body/accelerated_tracker.py": (
        "107974f9cb05dc0b21ddc555e8b858dbbb25a44192ea2621e408d045c545d310"
    ),
    "labs/20260909_three_body/graph_fit.py": (
        "807151e664ed8658f291c4d49eaff69ae71feff0f9a1f9cc9fca00292845edc0"
    ),
    "labs/20260909_three_body/tracker.py": (
        "f481e2d4105a0223cf24719482e1d057ba4e68094ff9df74af0a826a88d31c20"
    ),
    "labs/20260909_three_body/persistent_node_tracker.py": (
        "841ff2e3a67e174a7ea3205da17368730c47432eff7e8a79cf13000302ea148e"
    ),
    "labs/20260909_three_body/node_recovery.py": (
        "814f91f0e2638ec63222f976f87dcdd9fba6f27808d3c77d7a5e3dda457bbbc8"
    ),
    "labs/20260909_three_body/node_geometry.py": (
        "c7bea7333d5a20704e67c2283e8dc925a70e2f05c4520353d74b18c64b5aff69"
    ),
    "labs/20260909_three_body/motion.py": (
        "3ed3c4fbfa39a7c1df5804453423a76dd9d9ff064e66ab45a5beda1696852f85"
    ),
    **inputs.SOURCE_PINS,
}
EXECUTION_READY = True
require = base.require


def same(a, b, label):
    require(transport.signature(a) == transport.signature(b), label + " bytes differ")


def read_arrays(root, members, relative):
    path = prepared.bound_path(root, members, relative)
    return prepared.arrays_from_bytes(prepared.checked_bytes(path, members))


def member_index(record, root):
    """Old named+absolute aliases may coexist only with the same size and SHA."""
    result = {}
    for row in record["outputs"].values():
        path = Path(row["path"]).resolve()
        require(path.is_relative_to(root), "reference output escaped root")
        normalized = dict(
            path=str(path), sha256=row["sha256"], size_bytes=row.get("size_bytes", row.get("size"))
        )
        require(type(normalized["size_bytes"]) is int, "reference output size absent")
        require(
            str(path) not in result or result[str(path)] == normalized, "reference alias conflict"
        )
        result[str(path)] = normalized
    return result


def reference_header(evidence, original_context, shared):
    """Only small reference metadata before tracking; no candidate/state arrays."""
    seal_path = REFERENCE / "continuation.provenance.json"
    evidence.add(seal_path, REFERENCE_SHA)
    seal = base.read_json(seal_path)
    require(
        seal["status"] == "ok" and seal["stage"] == "three_body_persistent_continuation",
        "completed original200 proof required",
    )
    members = member_index(seal, REFERENCE)

    def small(relative, digest):
        path = prepared.bound_path(REFERENCE, members, relative, digest)
        evidence.add(path, digest)
        return json.loads(prepared.checked_bytes(path, members))

    manifest = small("checkpoint/manifest.json", REFERENCE_MANIFEST_SHA)
    metadata = small("checkpoint/metadata.json", REFERENCE_METADATA_SHA)
    require(
        metadata["context"] == original_context
        and metadata["initialize_count"] == 1
        and metadata["restore_count"] == 0
        and metadata["processed_frames"] == COUNT
        and metadata["previous_frame"] == STOP - 1,
        "original runtime/seed context differs",
    )
    require(
        manifest["status"] == "ok"
        and manifest["payload"]["sha256"] == REFERENCE_ARRAYS_SHA
        and manifest["metadata"]["sha256"] == REFERENCE_METADATA_SHA,
        "old checkpoint links differ",
    )
    binding = manifest["bindings"]
    require(
        binding["source_range"] == [FIRST, STOP]
        and binding["tracker_config"] == original_context["tracker_config"],
        "old fixed interval/config",
    )
    for path, digest in shared.items():
        require(
            binding["input_sha256"].get(path) == digest, "reference SL/calibration binding differs"
        )
    for path, digest in SOURCE_PINS.items():
        if path.startswith("labs/") or path.startswith("src/"):
            require(
                binding["code_sha256"].get(str(REPO / path)) == digest, "old objective code differs"
            )
    prepared.bound_path(REFERENCE, members, "candidate.npz", REFERENCE_CANDIDATE_SHA)
    prepared.bound_path(REFERENCE, members, "checkpoint/arrays.npz", REFERENCE_ARRAYS_SHA)
    return dict(seal=seal, members=members, metadata=metadata, binding=binding)


def validate_point_contract(plan, summary):
    require(
        plan["schema"] == summary["schema"] == points.STAGE and summary["status"] == "ok",
        "successful fixed anonymous point stage required",
    )
    require(
        plan["native_interval"] == summary["native_interval"] == [FIRST, STOP]
        and plan["views"] == list(VIEWS),
        "fixed point native/views",
    )
    require(
        summary["query_count"] == 334
        and summary["initial_mask_groups"] == 10
        and summary["model_forward_calls"] == 72
        and summary["model_initializations"] == 3
        and summary["saved_png_decodes"] == 600
        and summary["original64_prefix_all_fields_byte_exact"] is True,
        "point full scope/prefix qualification differs",
    )
    require(
        plan["preparation_sha256"] == inputs.PREPARATION_SHA
        and plan["anonymous_initial_mask_groups"] is True
        and plan["first_emission_primary"] is True
        and summary["no_reseed"] is True
        and summary["no_query_replacement"] is True,
        "point input/group/release contract differs",
    )
    require(summary["environment"]["opencv_threads"] == 32, "point LK thread context differs")


def horizon(native):
    t = native - FIRST
    require(0 <= t < COUNT, "native outside fixed200")
    return FIRST + (15 if t < 16 else (t // 8 + 1) * 8 - 1)


class SavedPointEvidence:
    """One-native immutable handoff; no pose or candidate supplied to provider."""

    def __init__(self, seal, members):
        self.members, self.cache, self.query_rows = (
            members,
            {},
            base.bound_json(POINTS, members, "query_rows.json"),
        )
        require([r["view"] for r in self.query_rows] == list(VIEWS), "point query row order")
        self.queries, self.coords, self.flags = {}, {}, {}
        for row in self.query_rows:
            view = row["view"]
            query = read_arrays(POINTS, members, f"{view}/queries.npz")
            points.validate_queries(query, row, view)
            require(
                seal["inputs"].get(str(points.OLD_PREPARATION / row["path"])) == row["sha256"],
                "original anonymous query input not bound",
            )
            self.queries[view] = query
            lk = read_arrays(POINTS, members, f"{view}/lk.npz")
            ct = read_arrays(POINTS, members, f"{view}/emissions.npz")
            for method, xy, flags in (
                ("lk", lk["xy"], lk["valid"]),
                ("ct_first", ct["first_xy"], ct["first_visibility"]),
            ):
                require(
                    xy.dtype == np.float32
                    and xy.shape == (COUNT, len(query["xy"]), 2)
                    and flags.dtype == bool
                    and flags.shape == xy.shape[:2],
                    "point arrays scope",
                )
                self.coords[method, view] = transport._freeze(xy)
                self.flags[method, view] = transport._freeze(flags)
            same(lk["xy"][0], query["xy"], "LK initial queries")
            same(ct["frame_idx"], np.arange(FIRST, STOP, dtype=np.int64), "point native vector")
            same(
                ct["first_release_native"],
                np.array([horizon(n) for n in range(FIRST, STOP)], np.int64),
                "first-release horizon",
            )
        self.query_sha256 = tuple(r["sha256"] for r in self.query_rows)
        self.group_counts = tuple(points.GROUP_COUNTS[v] for v in VIEWS)

    def prepare(self, frame):
        """Verify each saved membership field against the same native all-N mask."""
        native, t = int(frame.frame_idx), int(frame.frame_idx) - FIRST
        self.cache = {"lk": [], "ct_first": []}
        for vi, view in enumerate(VIEWS):
            saved = read_arrays(POINTS, self.members, f"{view}/memberships/native_{native}.npz")
            same(saved["frame_idx"], np.asarray(native, np.int64), "membership native")
            same(
                saved["release_native"], np.asarray(horizon(native), np.int64), "membership release"
            )
            for method in ("lk", "ct_first"):
                actual = point_core.mask_memberships(
                    self.coords[method, view][t], frame.masks[view], (1200, 1200)
                )
                saved_keys = {k[len(method) + 1 :] for k in saved if k.startswith(method + "_")}
                require(saved_keys == set(actual), "all point membership fields required")
                for key, value in actual.items():
                    same(np.asarray(value), saved[method + "_" + key], "native membership " + key)
                vote = transport.transport_votes(
                    self.queries[view]["mask_index"],
                    self.flags[method, view][t],
                    actual["membership"],
                    group_count=self.group_counts[vi],
                    mask_available=frame.masks[view] is not None,
                    native=native,
                    release_horizon=horizon(native),
                    query_sha256=self.query_sha256[vi],
                )
                self.cache[method].append(vote)
        self.cache = {k: tuple(v) for k, v in self.cache.items()}

    def provider(self, method):
        require(method in ARMS, "unknown arm")

        def provide(native):
            require(method != "original", "zero arm must not request temporal targets")
            values = self.cache[method]
            require(
                len(values) == 3 and all(v.native == native for v in values), "vote cursor mismatch"
            )
            return values

        return provide


def validate_samples(arrays, native):
    same(arrays["frame_idx"], np.asarray(native, np.int64), "fixed sample native")
    require(
        arrays["fit_pixels"].shape == (3, 1024, 2)
        and arrays["fit_target"].shape == arrays["fit_weights"].shape == (3, 1024)
        and arrays["fit_foreground"].shape == (3, 1024)
        and arrays["unused_pixels"].shape == (3, 256, 2)
        and not arrays["unused_fit_overlap"].any(),
        "fixed sample budget/overlap differs",
    )


def fit_sample_match(capture, samples):
    for key in ("fit_pixels", "fit_target", "fit_weights", "fit_foreground"):
        same(capture[key], samples[key], "actual fitter " + key)


def check_lifecycle(tracker, completed):
    require(
        tracker.initialize_count == 1
        and tracker.restore_count == 0
        and tracker.processed_frames == completed
        and tracker.previous_frame == (None if completed == 0 else FIRST + completed - 1),
        "initialization/restore/native lifecycle drift",
    )
    require(1 <= len(tracker._states) <= 6, "retained beam count outside budget")


def context_match(tracker, original):
    now = copy.deepcopy(tracker._context())
    policy = now.pop("mask_transport")
    now["implementation"] = original["implementation"]
    require(now == original, "new tracker changes original science context")
    require(
        policy["maximum_frames"] == COUNT
        and policy["proposal_permutation_relabels_target"] is False,
        "new target lineage policy differs",
    )


def initial_match(records, anchors, saved_records, saved_anchors):
    require(
        len(records) == len(saved_records) == len(anchors) == len(saved_anchors), "t0 beam differs"
    )
    for current, old in zip(records, saved_records, strict=True):
        require(set(current) == set(old) == set(RECORD_FIELDS), "t0 all19 required")
        for key in RECORD_FIELDS:
            if np.asarray(current[key]).dtype.kind == "U":
                require(str(current[key]) == str(old[key]), "t0 string value differs")
            else:
                same(np.asarray(current[key]), np.asarray(old[key]), "t0 " + key)
    for now, old in zip(anchors, saved_anchors, strict=True):
        require(
            now.anchor_id == old.anchor_id and now.initial_native == old.initial_native,
            "t0 anchor identity differs",
        )
        for a, b in zip(now.links, old.links, strict=True):
            same(a, b, "t0 causal initial links")


def run_frames(tracker, frames, *, prepare_frame, before_frame, after_frame, on_frame, synchronize):
    """Finite serial loop; lifecycle/sample/persistence hooks are fixture-injectable."""
    check_lifecycle(tracker, 0)
    consumed = 0
    for frame in frames:
        require(
            type(frame.frame_idx) is int
            and frame.frame_idx == FIRST + consumed
            and consumed < COUNT,
            "fixed200 frame sequence differs",
        )
        check_lifecycle(tracker, consumed)
        prepare_frame(frame)
        before_frame(frame.frame_idx)
        start = time.perf_counter()
        tracker.advance(frame)
        synchronize()
        elapsed = time.perf_counter() - start
        check_lifecycle(tracker, consumed + 1)
        on_frame(frame, elapsed)
        after_frame(frame.frame_idx)
        consumed += 1
    require(consumed == COUNT, "fixed200 input ended early")
    check_lifecycle(tracker, COUNT)


def compare_array_maps(actual, expected, label):
    require(set(actual) == set(expected), label + " array membership differs")
    rows = []
    for key in sorted(actual):
        a, b = transport.signature(actual[key]), transport.signature(expected[key])
        rows.append(
            dict(
                field=key,
                actual=a.__dict__
                if hasattr(a, "__dict__")
                else dict(dtype=a.dtype, shape=a.shape, sha256=a.sha256),
                expected=dict(dtype=b.dtype, shape=b.shape, sha256=b.sha256),
                byte_exact=a == b,
            )
        )
    return dict(
        label=label,
        array_count=len(rows),
        byte_exact=all(r["byte_exact"] for r in rows),
        arrays=rows,
    )


def original_gate(candidate, core, reference, evidence, *, finished, output):
    require(
        finished is True and len(candidate["frame_idx"]) == COUNT,
        "old arrays require completed Original",
    )
    members = reference["members"]
    old_candidate = read_arrays(REFERENCE, members, "candidate.npz")
    old_core = read_arrays(REFERENCE, members, "checkpoint/arrays.npz")
    evidence.add(REFERENCE / "candidate.npz", REFERENCE_CANDIDATE_SHA)
    evidence.add(REFERENCE / "checkpoint/arrays.npz", REFERENCE_ARRAYS_SHA)
    require(len(candidate) == len(old_candidate) == 21, "all21 original candidate required")
    require(len(core) == len(old_core) == 39, "all39 original state required")
    candidate_check = compare_array_maps(candidate, old_candidate, "original21")
    history_keys = {"history_offsets"} | {"history_" + k for k in RECORD_FIELDS}
    history_check = compare_array_maps(
        {k: core[k] for k in history_keys},
        {k: old_core[k] for k in history_keys},
        "all19 retained histories",
    )
    state_check = compare_array_maps(core, old_core, "all39 original numeric state")
    result = dict(
        schema="mask_transport_original_control_v1",
        native_interval=[FIRST, STOP],
        reference_loaded_after_original=True,
        candidate=candidate_check,
        history=history_check,
        state=state_check,
        passed=all(r["byte_exact"] for r in (candidate_check, history_check, state_check)),
    )
    output.write("original_control.json", result)
    require(result["passed"], "Original replay differs; new ownership arms must not start")
    return result


def save_tree(output, name, value):
    arrays = {}
    tree = tracking._pack(value, arrays)
    output.write(name + ".npz", arrays)
    output.write(
        name + ".json",
        dict(tree=tree, array_signatures={k: tracking.signature(v) for k, v in arrays.items()}),
    )


def arm_summary(name, candidate, core, tracker, times):
    observed = candidate["observed"]
    return dict(
        arm=name,
        frames=COUNT,
        first_native=FIRST,
        last_native=STOP - 1,
        initialize_count=tracker.initialize_count,
        restore_count=tracker.restore_count,
        coefficient=0.0 if name == "original" else 3.0,
        retained_rows=int(core["history_offsets"][-1]),
        initial_anchors=len(tracker._anchors),
        node_support_by_slot=observed.sum(axis=0).tolist(),
        node_support_total=observed.sum(axis=(0, 1)).tolist(),
        identity_review_by_slot=candidate["identity_review"].sum(axis=0).tolist(),
        proposal_counts=[len(x["parent"]) for x in tracker._fit_history],
        active_owner_frames=sum(bool(x["owner"]["active"]) for x in tracker._fit_history),
        advance_and_synchronize_seconds=float(sum(times)),
        no_external_pose_reset=True,
        independent_ID_truth=False,
    )


def validate_execution(args):
    require(
        args.execute is True and args.device == "cuda", "explicit reviewed CUDA execution required"
    )
    require(
        EXECUTION_READY and all(base.valid_sha(x) for x in SOURCE_PINS.values()),
        "source pins/qualification still in draft",
    )
    require(args.samples_seal_sha256 == SAMPLES_SHA, "fixed successful samples seal SHA required")


def runtime(binding):
    require(
        binding["opencv_threads"] == 32 and binding["torch_cpu_threads"] == 2,
        "original thread policy differs",
    )
    cv2.setNumThreads(32)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    require(
        torch.backends.cuda.matmul.allow_tf32 == binding["cuda_matmul_tf32"]
        and torch.backends.cudnn.allow_tf32 == binding["cudnn_tf32"]
        and torch.backends.cudnn.benchmark == binding["cudnn_benchmark"]
        and torch.are_deterministic_algorithms_enabled()
        == binding["torch_deterministic_algorithms"],
        "original numerical runtime flags differ",
    )
    require(torch.cuda.is_available(), "CUDA unavailable")
    versions = dict(
        numpy=np.__version__,
        scipy=scipy.__version__,
        opencv=cv2.__version__,
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        cuda_device_name=torch.cuda.get_device_name(0),
    )
    validate_runtime_versions(binding["package_versions"], versions)
    torch.cuda.set_per_process_memory_fraction(0.25, 0)
    torch.cuda.reset_peak_memory_stats(0)
    return dict(
        opencv_threads=cv2.getNumThreads(),
        torch_threads=torch.get_num_threads(),
        torch_interop_threads=torch.get_num_interop_threads(),
        dtype="float64",
        torch=torch.__version__,
        numpy=np.__version__,
        opencv=cv2.__version__,
        gpu_name=torch.cuda.get_device_name(0),
        cuda=torch.version.cuda,
        cuda_matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        allocator_fraction=0.25,
        package_versions=versions,
    )


def validate_runtime_versions(expected, actual):
    keys = {"numpy", "scipy", "opencv", "torch", "cuda", "cuda_device_name"}
    require(set(expected) == set(actual) == keys, "original runtime version schema differs")
    require(expected == actual, "original runtime package/device versions differ")


def gpu_memory(label, *, minimum_gib):
    free, total = torch.cuda.mem_get_info(0)
    require(free >= minimum_gib * 1024**3, "insufficient free GPU memory before " + label)
    return dict(
        label=label,
        free_bytes=free,
        total_bytes=total,
        allocated_bytes=torch.cuda.memory_allocated(0),
        reserved_bytes=torch.cuda.memory_reserved(0),
    )


def load_inputs(evidence, samples_sha):
    for relative, digest in SOURCE_PINS.items():
        require(base.valid_sha(digest), "final source pin missing")
        evidence.add(REPO / relative, digest)
    evidence.add(Path(__file__).resolve())
    common = inputs.load_common(evidence)
    samples_seal, sample_members = base.verified(
        SAMPLES, "samples.provenance.json", inputs.STAGE, samples_sha, evidence
    )
    sample_plan = base.bound_json(SAMPLES, sample_members, "plan.json")
    sample_summary = base.bound_json(SAMPLES, sample_members, "summary.json")
    require(
        sample_plan["native_interval"] == [FIRST, STOP]
        and sample_plan["implementation_plan_sha256"] == inputs.PLAN_SHA
        and sample_plan["preparation_sha256"] == inputs.PREPARATION_SHA
        and sample_plan["before_new_fit"] is True
        and sample_summary["frames"] == COUNT
        and sample_summary["overlap_draws"] == 0,
        "fixed samples scope differs",
    )
    require(
        samples_seal["inputs"].get(str(inputs.PREPARATION / "preparation.provenance.json"))
        == inputs.PREPARATION_SHA,
        "sample/preparation evidence differs",
    )
    point_seal, point_members = base.verified(
        POINTS, "tracking.provenance.json", points.STAGE, POINTS_SHA, evidence
    )
    point_plan, point_summary = (
        base.bound_json(POINTS, point_members, n) for n in ("plan.json", "summary.json")
    )
    validate_point_contract(point_plan, point_summary)
    require(
        point_seal["inputs"].get(str(inputs.PREPARATION / "preparation.provenance.json"))
        == inputs.PREPARATION_SHA,
        "point/preparation evidence differs",
    )
    require(
        point_seal["inputs"].get(str(WORK / "track_mask_transport200.py"))
        == SOURCE_PINS["work/three_body/track_mask_transport200.py"],
        "point implementation binding differs",
    )
    reference = reference_header(
        evidence,
        common["original_context"],
        common["context"]["plan"]["shared_SL_calibration_sha256"],
    )
    for native in range(FIRST, STOP):
        prepared.bound_path(SAMPLES, sample_members, f"samples/native_{native}.npz")
    return common, sample_members, SavedPointEvidence(point_seal, point_members), reference


def run(args):
    validate_execution(args)
    destination = Path(args.output).resolve()
    require(
        destination.is_relative_to(WORK) and not destination.exists(),
        "new workspace output required",
    )
    output, evidence, captured = base.Output(destination), base.Evidence(), {}
    started, start, completed, memory = base.utc(), time.perf_counter(), [], []
    old_cv = cv2.getNumThreads()
    active_arm, last_saved, tracker = None, None, None
    try:
        common, sample_members, saved_points, reference = load_inputs(
            evidence, args.samples_seal_sha256
        )
        environment = runtime(reference["binding"])
        memory.append(gpu_memory("startup", minimum_gib=6))
        prepared.snapshot_sources(evidence, output, captured)
        cfg = common["original_context"]
        require(
            cfg["tracker_config"]["device"] == "cuda"
            and cfg["tracker_config"]["flow_proposals"] is True
            and cfg["tracker_config"]["max_hypotheses"] == 18
            and cfg["tracker_config"]["opt_steps"] == 14,
            "original pool/steps/flow policy differs",
        )
        input_binding = dict(
            schema=STAGE,
            preparation_sha256=inputs.PREPARATION_SHA,
            point_sha256=POINTS_SHA,
            samples_sha256=args.samples_seal_sha256,
            implementation_plan_sha256=inputs.PLAN_SHA,
            source_pins=SOURCE_PINS,
        )
        output.write(
            "plan.json",
            dict(
                schema=STAGE,
                native_interval=[FIRST, STOP],
                arms=list(ARMS),
                input_binding=input_binding,
                original_context=cfg,
                environment=environment,
                target_lineage="causal_t0_retained_parent",
                no_external_pose_reset=True,
                extra_proposals=0,
                maximum_pool=24,
                opt_steps=14,
                owner_coefficient=3.0,
                original_reference_numeric_read="after_original_saved_before_lk",
                first_release_horizons=[horizon(n) for n in range(FIRST, STOP)],
                model_forwards=0,
                no_ID_review_clearing=True,
                independent_ID_truth=False,
                timing_scope=(
                    "advance + CUDA synchronize includes serial flow, target construction, fit, "
                    "t0 geometry and capture; excludes input decode, saved-membership "
                    "verification, saving and guards"
                ),
            ),
        )
        output.write("original_seed.json", common["seed"])
        output.write("query_rows.json", saved_points.query_rows)
        initial_records = initial_anchors = control = None
        for arm in ARMS:
            active_arm, last_saved = arm, None
            require(arm == "original" or control["passed"] is True, "Original control gate absent")
            memory.append(gpu_memory("arm_start_" + arm, minimum_gib=6))
            tracker = tracking.MaskTransportTracker(
                common["rig"],
                common["initial"].copy(),
                config=TrackerConfig(**cfg["tracker_config"]),
                recovery_config=NodeRecoveryConfig(**cfg["recovery_config"]),
                persistence_config=PersistentNodeConfig(**cfg["persistence_config"]),
                first_native=FIRST,
                group_counts=saved_points.group_counts,
                query_sha256=saved_points.query_sha256,
                vote_provider=saved_points.provider(arm),
                input_binding={**input_binding, "arm": arm},
                coefficient=0 if arm == "original" else 3,
            )
            context_match(tracker, cfg)
            tracker.begin()
            samples, times = {}, []

            def prepare_frame(frame, samples=samples):
                samples.clear()
                samples.update(
                    read_arrays(SAMPLES, sample_members, f"samples/native_{frame.frame_idx}.npz")
                )
                validate_samples(samples, frame.frame_idx)
                saved_points.prepare(frame)

            def on_frame(frame, elapsed, tracker=tracker, arm=arm, samples=samples, times=times):
                nonlocal initial_records, initial_anchors, last_saved
                capture = tracker._fit_history[-1]
                # Persist actual values before a post-fit mismatch can reject them.
                save_tree(
                    output,
                    f"{arm}/frames/native_{frame.frame_idx}",
                    dict(
                        records=[
                            {k: np.asarray(r[k]) for k in RECORD_FIELDS}
                            for r in tracker._history[-1]
                        ],
                        fit=capture,
                        anchor_ids=tracker._anchor_ids,
                        release_native=horizon(frame.frame_idx),
                        advance_seconds=elapsed,
                    ),
                )
                last_saved = int(frame.frame_idx)
                fit_sample_match(capture, samples)
                if frame.frame_idx == FIRST:
                    save_tree(
                        output,
                        f"{arm}/initial_anchors",
                        dict(anchors=tracker._anchors, facts=tracker._anchor_facts),
                    )
                    if arm == "original":
                        initial_records, initial_anchors = (
                            copy.deepcopy(tracker._history[0]),
                            tracker._anchors,
                        )
                    else:
                        initial_match(
                            tracker._history[0], tracker._anchors, initial_records, initial_anchors
                        )
                times.append(elapsed)
                if (frame.frame_idx - FIRST + 1) % 20 == 0:
                    print(
                        json.dumps(
                            dict(
                                arm=arm,
                                completed=frame.frame_idx - FIRST + 1,
                                native=frame.frame_idx,
                                release_native=horizon(frame.frame_idx),
                            )
                        ),
                        flush=True,
                    )

            run_frames(
                tracker,
                common["reader"],
                prepare_frame=prepare_frame,
                before_frame=lambda n, arm=arm: memory.append(
                    gpu_memory(f"{arm}_{n}", minimum_gib=2)
                ),
                after_frame=lambda n: None,
                on_frame=on_frame,
                synchronize=lambda: torch.cuda.synchronize(0),
            )
            core, meta, side_arrays, side_meta = tracker.export_transport_state()
            output.write(f"{arm}/checkpoint/arrays.npz", core)
            output.write(f"{arm}/checkpoint/metadata.json", meta)
            output.write(f"{arm}/checkpoint/side_arrays.npz", side_arrays)
            output.write(f"{arm}/checkpoint/side_metadata.json", side_meta)
            candidate = tracker.finish()
            output.write(f"{arm}/candidate.npz", candidate)
            row = arm_summary(arm, candidate, core, tracker, times)
            output.write(f"{arm}/summary.json", row)
            if arm == "original":
                control = original_gate(
                    candidate, core, reference, evidence, finished=True, output=output
                )
            completed.append(row)
            del on_frame, prepare_frame
            tracker = None
            del core, meta, side_arrays, side_meta, candidate
            gc.collect()
            torch.cuda.empty_cache()
            prepared.snapshot_sources(evidence, output, captured)
        output.write("memory_samples.json", memory)
        summary = dict(
            schema=STAGE,
            status="ok",
            native_interval=[FIRST, STOP],
            arms=completed,
            original_control_passed=control["passed"],
            model_forwards=0,
            initialize_calls=3,
            restore_calls=0,
            advance_calls=3 * COUNT,
            maximum_pool=24,
            steps=14,
            environment=environment,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
            source_hashes=captured,
            elapsed_before_guards_seconds=time.perf_counter() - start,
            independent_ID_truth=False,
            new_detector_or_point_model=False,
        )
        output.write("summary.json", summary)
        prepared.snapshot_sources(evidence, output, captured)
        evidence.final_guard()
        output.final_guard()
        seal = dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=base.utc(),
            inputs=evidence.paths,
            outputs=output.records,
            params=dict(all_guards_passed=True),
        )
    except BaseException as error:
        output.write(
            "failure.json",
            dict(
                active_arm=active_arm,
                last_saved_native=last_saved,
                completed_arms=completed,
                error_type=type(error).__name__,
                error=str(error),
                partial_results_are_not_complete=True,
            ),
        )
        seal = dict(
            stage=STAGE,
            status="failed",
            started_at=started,
            finished_at=base.utc(),
            inputs=evidence.paths,
            outputs=output.records,
            error_type=type(error).__name__,
            error=str(error),
        )
        with (destination / "tracking.provenance.json").open("xb") as stream:
            stream.write(base.json_bytes(seal))
        raise
    finally:
        cv2.setNumThreads(old_cv)
    with (destination / "tracking.provenance.json").open("xb") as stream:
        stream.write(base.json_bytes(seal))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument("--device", choices=["cuda"], required=True)
    parser.add_argument("--samples-seal-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args)), flush=True)


if __name__ == "__main__":
    main()
