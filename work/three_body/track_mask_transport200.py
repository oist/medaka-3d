"""Fixed200 anonymous whole-mask points with an exact saved64-prefix control.

Reuse sealed600 native PNGs/all masks and original334 GFTT query bytes. This
stage records observations only; no3D fit, body anchor, ID or review changes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import compare_temporal_known_gap64 as base
import cv2
import mask_transport_stream as streaming
import numpy as np
import temporal_point_core as core
import temporal_prefix_deltas as delta
import track_semantic_node200 as prepared

WORK, REPO = base.WORK, base.REPO
FIRST, STOP, COUNT = 333519, 333719, 200
VIEWS = base.VIEWS
STAGE = "mask_transport200_point_tracking_v1"
PREPARATION = prepared.PREPARATION
PREPARATION_SHA = "153c9f5f4a316f193b609afd2113b875cf1a62380dca4b80a6b27834b53afc15"
OLD_PREPARATION = base.PREPARATION
OLD_COMPARISON = WORK / "temporal_known_gap64_comparison_v1"
OLD_COMPARISON_SHA = "bd0bae4df3da0d8f2f4a31f0e754a93e89460c6b72803d2054327655662e732e"
IMPLEMENTATION_PLAN = WORK / "MASK_TRANSPORT_200_IMPLEMENTATION_PLAN.md"
IMPLEMENTATION_PLAN_SHA = "2fe9c7681cbefd6b1f79c0d49f39e07ff7bc3867c2b5559029b513ea5475b130"
QUERY_COUNTS = dict(zip(VIEWS, (157, 80, 97), strict=True))
GROUP_COUNTS = dict(zip(VIEWS, (3, 4, 3), strict=True))
require = core.require
arrays_from_bytes = prepared.arrays_from_bytes
bound_path = prepared.bound_path
checked_bytes = prepared.checked_bytes
load_model_evidence = prepared.load_model_evidence
snapshot_sources = prepared.snapshot_sources
load_masks = prepared.load_masks
SOURCE_PINS = {
    "track_semantic_node200.py": "219703a8101aac1bc69e60ee061ffbb5875c41027df6ceb337e4a1f8f029bcca",
    "mask_transport_stream.py": "1d18f7b5bdfe5cf9522a6379dde4c17f6e1775622b7f6aef39da487d06eadf16",
    **prepared.SOURCE_PINS,
}


def validate_queries(query, row, view):
    require(set(query) == {"frame_idx", "xy", "mask_index", "corner_index", "available"},
            "five original GFTT fields required")
    xy, group, corner = (query[k] for k in ("xy", "mask_index", "corner_index"))
    require(xy.dtype == np.float32 and xy.shape == (QUERY_COUNTS[view], 2)
            and np.isfinite(xy).all() and ((xy >= 0) & (xy < 1200)).all(),
            "fixed query count/native pixel dtype differs")
    require(group.dtype == corner.dtype == np.int64
            and group.shape == corner.shape == (len(xy),), "group integer schema differs")
    require(query["frame_idx"].dtype == np.int64 and query["frame_idx"].shape == ()
            and int(query["frame_idx"]) == FIRST, "initial native differs")
    require(query["available"].dtype == bool and query["available"].shape == ()
            and bool(query["available"]), "initial masks unavailable")
    require(row["view"] == view and row["query_count"] == len(xy)
            and row["mask_count"] == GROUP_COUNTS[view]
            and len(row["rows"]) == GROUP_COUNTS[view], "fixed view/groups differ")
    require(delta.signatures(query) == row["array_signatures"], "original query bytes differ")
    cursor = 0
    for index, record in enumerate(row["rows"]):
        lo, hi = record["query_start"], record["query_stop"]
        require(type(lo) is int and type(hi) is int and lo == cursor
                and record["mask_index"] == index and 0 <= hi - lo <= 64
                and hi - lo == record["query_count"] and hi <= len(xy)
                and np.all(group[lo:hi] == index)
                and np.array_equal(corner[lo:hi], np.arange(hi-lo, dtype=np.int64)),
                "all initial groups/corner order required")
        cursor = hi
    require(cursor == len(xy), "omitted query group")


def read_old(context, relative):
    path = bound_path(OLD_COMPARISON, context["old_members"], relative)
    return arrays_from_bytes(checked_bytes(path, context["old_members"]))


def match_prefix(current, saved, *, count, name):
    require(set(current) == set(saved), name + " member fields differ")
    for key in current:
        require(current[key].ndim >= 1 and len(current[key]) >= count
                and len(saved[key]) == count, name + " frame count differs")
        base.same(current[key][:count], saved[key], name + ":" + key)


def load_context(evidence, preparation_sha):
    require(preparation_sha == PREPARATION_SHA, "fixed prepared200 seal required")
    evidence.add(IMPLEMENTATION_PLAN, IMPLEMENTATION_PLAN_SHA)
    for name, digest in SOURCE_PINS.items():
        evidence.add(WORK / name, digest)
    evidence.add(Path(__file__).resolve())
    context = prepared.load_context(evidence, preparation_sha)
    old_seal, old_members = base.verified(
        OLD_PREPARATION, "preparation.provenance.json", "temporal_known_gap64_preparation_v1",
        base.PREPARATION_SHA, evidence)
    rows = base.bound_json(OLD_PREPARATION, old_members, "query_rows.json")
    natives = base.bound_json(OLD_PREPARATION, old_members, "native_evidence.json")
    require([r["view"] for r in rows] == list(VIEWS), "ordered old query rows required")
    now = {(r["view"], r["frame_idx"]): r for r in context["natives"]}
    require(len(natives) == 192 and len({(r["view"], r["frame_idx"]) for r in natives}) == 192,
            "saved64 native membership differs")
    for row in natives:
        require(FIRST <= row["frame_idx"] < FIRST + 64, "old64 native extent differs")
        current = now[(row["view"], row["frame_idx"])]
        for key in ("bgr_sha256", "mask_sha256", "mask_count", "mask_available"):
            require(row[key] == current[key], "old64 prepared prefix changed: " + key)
    queries = {}
    for row in rows:
        path = bound_path(OLD_PREPARATION, old_members, row["path"], row["sha256"])
        query = arrays_from_bytes(checked_bytes(path, old_members))
        validate_queries(query, row, row["view"])
        queries[row["view"]] = query
    comparison, members = base.verified(
        OLD_COMPARISON, "comparison.provenance.json", "temporal_known_gap64_comparison_v1",
        OLD_COMPARISON_SHA, evidence)
    require(comparison["inputs"].get(str(OLD_PREPARATION / "preparation.provenance.json"))
            == base.PREPARATION_SHA, "old model comparison not bound to query preparation")
    old_environment = base.bound_json(OLD_COMPARISON, members, "summary.json")["environment"]
    validate_thread_policy(old_seal["environment"], old_environment)
    context.update(queries=queries, query_rows=rows, old_members=members,
                   opencv_threads=old_environment["opencv_threads"],
                   old_environment=old_environment)
    for view in VIEWS:
        old_query = read_old(context, view + "/queries.npz")
        require(delta.signatures(old_query) == delta.signatures(queries[view]),
                "old model consumed different query bytes")
    return context


def validate_thread_policy(old_preparation_environment, old_comparison_environment):
    """Reuse the numerical LK runtime; PNG transport threads are not LK policy."""
    value = old_comparison_environment["opencv_threads"]
    require(type(value) is int and value == 32
            and old_preparation_environment["opencv_threads"] == value,
            "original64 LK thread policy differs")


def run_view(context, view, closure, output, device, memory_rows):
    rows = sorted(
        (r for r in context["natives"] if r["view"] == view), key=lambda r: r["frame_idx"]
    )
    query = context["queries"][view]
    xy, count = query["xy"], len(query["xy"])
    lk = streaming.StreamingLK(xy, first=FIRST)
    anchor, decoded = {}, []

    def images():
        for row in rows:
            path = bound_path(PREPARATION, context["members"], row["png_path"])
            image = cv2.imdecode(
                np.frombuffer(checked_bytes(path, context["members"]), np.uint8), cv2.IMREAD_COLOR
            )
            require(
                image is not None and image.dtype == np.uint8 and image.shape == (1200, 1200, 3),
                "prepared BGR schema differs",
            )
            require(
                hashlib.sha256(image.tobytes()).hexdigest() == row["bgr_sha256"],
                "native pixels differ",
            )
            lk.push(row["frame_idx"], image)
            if not decoded:
                anchor.update(core.sample_patches(image, xy))
            decoded.append(row["frame_idx"])
            yield row["frame_idx"], image

    diagnostics = {
        method: {
            "reported": np.zeros((COUNT, count), bool),
            "mad": np.full((COUNT, count), np.nan, np.float64),
            "zncc": np.full((COUNT, count), np.nan, np.float64),
            "pair_available": np.zeros((COUNT, count), bool),
            "variance_ok": np.zeros((COUNT, count), bool),
            "mask_members": np.zeros((COUNT, count), np.int32),
            "mask_available": np.zeros(COUNT, bool),
            "in_image": np.zeros((COUNT, count), bool),
        }
        for method in ("lk", "ct_first", "stationary")
    }
    releases = []

    def first_emission(native, image, result, horizon):
        t = native - FIRST
        masks = load_masks(context, rows[t])
        members = dict(
            frame_idx=np.asarray(native, np.int64), release_native=np.asarray(horizon, np.int64)
        )
        methods = {
            "lk": (lk.arrays["xy"][t], lk.arrays["valid"][t]),
            "ct_first": (result["xy"], result["reported"]),
            "stationary": (xy, np.ones(count, bool)),
        }
        for method, (coords, flags) in methods.items():
            membership = core.mask_memberships(coords, masks, (1200, 1200))
            members.update(
                {method + "_" + key: np.asarray(value) for key, value in membership.items()}
            )
            sampled = core.sample_patches(image, coords)
            patch = core.patch_comparison(anchor["values"], sampled["values"])
            diag = diagnostics[method]
            diag["reported"][t] = flags
            for key, value in patch.items():
                diag[key][t] = value
            diag["mask_members"][t] = membership["membership"].sum(axis=1)
            diag["mask_available"][t] = membership["available"]
            diag["in_image"][t] = membership["in_image"]
        output.write(f"{view}/memberships/native_{native}.npz", members)
        releases.append((native, horizon))

    persisted_previous, delta_rows = None, []

    def save_call(header, payload):
        nonlocal persisted_previous
        index = header["call_metadata"]["call_index"]
        prefix = f"{view}/calls/{index:03}"
        output.write(prefix + ".npz", payload)
        output.write(prefix + ".json", header)
        saved_header = base.read_json(output.root / (prefix + ".json"))
        saved_payload = arrays_from_bytes((output.root / (prefix + ".npz")).read_bytes())
        persisted_previous = delta.decode_delta(persisted_previous, saved_header, saved_payload)
        if index < 7:
            saved64 = read_old(context, f"{view}/online/call_{index}.npz")
            match_prefix(persisted_previous, saved64, count=16 + 8 * index,
                         name=f"{view}:original64:call{index}")
        delta_rows.append(
            dict(
                call_index=index,
                logical_bytes=header["logical_bytes"],
                delta_array_bytes=header["delta_array_bytes"],
                outside_changed_rows={
                    key: value["outside_changed_rows"] for key, value in header["fields"].items()
                },
            )
        )

    def memory_guard(index):
        if device == "cuda":
            sample = base.gpu_memory()
            base.check_memory(sample)
            memory_rows.append(dict(view=view, call_index=index, **sample))

    backend, memory, start = None, {}, time.perf_counter()
    try:
        backend = base.TorchOnline(device, closure)
        emissions, calls = streaming.run_online200(
            images(),
            xy,
            backend,
            save_call,
            first=FIRST,
            before_call=memory_guard,
            on_first=first_emission,
        )
        require(
            backend.initializations == 1 and backend.forwards == 24,
            "model reset/call count differs",
        )
        load_seconds = backend.load_seconds
    finally:
        if backend is not None:
            memory = backend.close()
            memory_rows.append(dict(view=view, event="backend_closed", **memory))
    require(
        decoded == list(range(FIRST, STOP)) and len(releases) == COUNT,
        "full native delivery differs",
    )
    base.same(emissions["final_xy"], persisted_previous["xy"], "persisted final coordinates")
    base.same(emissions["final_product"], persisted_previous["product"], "persisted final score")
    output.write(f"{view}/queries.npz", query)
    lk_arrays = lk.finish()
    match_prefix(lk_arrays, read_old(context, f"{view}/lk_raw.npz"), count=64,
                 name=f"{view}:original64:LK")
    output.write(f"{view}/lk.npz", lk_arrays)
    output.write(f"{view}/emissions.npz", emissions)
    output.write(f"{view}/calls.json", calls)
    output.write(f"{view}/delta_sizes.json", delta_rows)
    for method, arrays in diagnostics.items():
        output.write(f"{view}/{method}_diagnostics.npz", arrays)
    a, b = diagnostics["lk"], diagnostics["ct_first"]
    common = a["reported"] & b["reported"]
    lk_only, ct_only = a["reported"] & ~b["reported"], b["reported"] & ~a["reported"]
    comparison = dict(common=common, lk_only=lk_only, ct_first_only=ct_only)
    for metric in ("mad", "zncc"):
        comparison[metric + "_ct_minus_lk"] = b[metric] - a[metric]
    output.write(f"{view}/comparison.npz", comparison)
    summary = dict(
        view=view,
        query_count=count,
        all200_query_denominator=COUNT * count,
        tracking199_query_denominator=(COUNT - 1) * count,
        lk_accepted=int(a["reported"][1:].sum()),
        ct_first_reported=int(b["reported"][1:].sum()),
        common=int(common[1:].sum()),
        lk_only=int(lk_only[1:].sum()),
        ct_first_only=int(ct_only[1:].sum()),
        per_query_tracking_counts=[
            dict(
                mask_index=int(query["mask_index"][i]),
                corner_index=int(query["corner_index"][i]),
                lk_accepted=int(a["reported"][1:, i].sum()),
                ct_first_reported=int(b["reported"][1:, i].sum()),
            )
            for i in range(count)
        ],
        common_patch_difference={
            metric: base.finite_distribution(comparison[metric + "_ct_minus_lk"][1:], common[1:])
            for metric in ("mad", "zncc")
        },
        first_release_native=FIRST + 15,
        final_release_native=STOP - 1,
        model_initializations=1,
        model_forward_calls=24,
        no_query_replacement=True,
        load_seconds=load_seconds,
        elapsed_with_diagnostics_seconds=time.perf_counter() - start,
        memory=memory,
        query_signatures=delta.signatures(query),
        old64_all7_calls6_fields_byte_exact=True,
        old64_LK_all_fields_byte_exact=True,
        initial_group_count=GROUP_COUNTS[view],
    )
    output.write(f"{view}/summary.json", summary)
    return summary


def run(args):
    require(args.execute and args.device in ("cpu", "cuda"), "explicit bounded execution required")
    require(base.valid_sha(args.preparation_sha256), "preparation seal SHA required")
    destination = args.output.resolve()
    require(
        destination.is_relative_to(WORK) and not destination.exists(),
        "new workspace output required",
    )
    require(
        not any(k == "cotracker" or k.startswith("cotracker.") for k in sys.modules),
        "fresh model interpreter required",
    )
    output, evidence, captured = base.Output(destination), base.Evidence(), {}
    started, start, views, memory = base.utc(), time.perf_counter(), [], []
    old_cv, torch, old_torch_threads = cv2.getNumThreads(), None, None
    try:
        context = load_context(evidence, args.preparation_sha256)
        closure = load_model_evidence(evidence)
        snapshot_sources(evidence, output, captured)
        cv_threads = context["opencv_threads"]
        require(type(cv_threads) is int and cv_threads > 0, "prepared OpenCV policy absent")
        cv2.setNumThreads(cv_threads)
        require(cv2.getNumThreads() == cv_threads, "prepared OpenCV policy differs")
        if args.device == "cuda":
            sample = base.gpu_memory()
            base.check_memory(sample, startup=True)
            memory.append(dict(event="startup", **sample))
        import torch as torch_library

        torch = torch_library
        old_torch_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        environment = dict(
            python=sys.version,
            packages={name: importlib.metadata.version(name) for name in base.DEPENDENCY_BINDING},
            device=args.device,
            opencv_threads=cv2.getNumThreads(),
            torch_threads=torch.get_num_threads(),
            torch_interop_threads=torch.get_num_interop_threads(),
            cuda_build=torch.version.cuda,
            matmul_precision=torch.get_float32_matmul_precision(),
            allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        )
        require(environment["packages"] == base.DEPENDENCY_BINDING, "qualified runtime differs")
        for key in ("packages", "device", "opencv_threads", "matmul_precision", "allow_tf32",
                    "cudnn_allow_tf32", "cudnn_benchmark", "deterministic_algorithms",
                    "cuda_build"):
            require(environment[key] == context["old_environment"][key],
                    "original64 comparison runtime changed: " + key)
        plan = dict(
            schema=STAGE,
            native_interval=[FIRST, STOP],
            views=list(VIEWS),
            preparation_sha256=args.preparation_sha256,
            original_query_preparation_sha256=base.PREPARATION_SHA,
            original64_comparison_sha256=OLD_COMPARISON_SHA,
            prepared_png_opencv_threads=context["seal"]["environment"]["opencv_threads"],
            source_pins=SOURCE_PINS,
            model_asset_sha256=base.ASSET_SEAL_SHA,
            initial_query_count=334,
            model_initializations=3,
            model_forward_limit=72,
            windows=[[i, i + 16] for i in range(0, 185, 8)],
            no_reseed=True,
            no_query_replacement=True,
            first_emission_primary=True,
            lookahead_frames=dict(initial_maximum=15, thereafter_maximum=7),
            full_prefix_lossless_delta=True,
            native_images_streamed=True,
            official_prefix_state_not_constant_memory=True,
            point_diagnostics=["lk", "ct_first", "stationary"],
            ct_final_coordinates_preserved=True,
            independent_biological_ID_truth=False,
            anonymous_initial_mask_groups=True,
            new_3D=False,
            new_fit=False,
            ID_or_review_changed=False,
            environment=environment,
        )
        output.write("plan.json", plan)
        output.write("query_rows.json", context["query_rows"])
        evidence.final_guard()
        for view in VIEWS:
            views.append(run_view(context, view, closure, output, args.device, memory))
            snapshot_sources(evidence, output, captured)
            print(
                json.dumps(dict(event="view_complete", view=view, calls=24,
                                queries=QUERY_COUNTS[view])), flush=True
            )
        summary = dict(
            schema=STAGE,
            status="ok",
            native_interval=[FIRST, STOP],
            views=views,
            query_count=334,
            all200_query_denominator=66800,
            tracking199_query_denominator=66466,
            model_initializations=3,
            model_forward_calls=72,
            saved_png_decodes=600,
            lk_view_edges=597,
            initial_mask_groups=10,
            original64_prefix_all_fields_byte_exact=True,
            no_reseed=True,
            no_query_replacement=True,
            first_emission_primary=True,
            new_3D=False,
            new_fit=False,
            ID_or_review_changed=False,
            independent_ID_truth=False,
            temporal_holdout=False,
            environment=environment,
            source_hashes=captured,
            elapsed_before_guards_seconds=time.perf_counter() - start,
        )
        output.write("memory_samples.json", memory)
        output.write("summary.json", summary)
        snapshot_sources(evidence, output, captured)
        evidence.final_guard()
        output.final_guard()
        status = dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=base.utc(),
            inputs=evidence.paths,
            outputs=output.records,
            params=dict(all_guards_passed=True, elapsed_seconds=time.perf_counter() - start),
        )
        with (destination / "tracking.provenance.json").open("xb") as file:
            file.write(base.json_bytes(status))
        return summary
    except BaseException as error:
        status = dict(
            stage=STAGE,
            status="failed",
            started_at=started,
            finished_at=base.utc(),
            inputs=evidence.paths,
            outputs=output.records,
            completed_views=[v["view"] for v in views],
            error_type=type(error).__name__,
            error=str(error),
            memory_samples=memory,
        )
        with (destination / "tracking.provenance.json").open("xb") as file:
            file.write(base.json_bytes(status))
        raise
    finally:
        cv2.setNumThreads(old_cv)
        if torch is not None and old_torch_threads is not None:
            torch.set_num_threads(old_torch_threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--preparation-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
