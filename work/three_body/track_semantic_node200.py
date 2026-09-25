"""Fixed200 initial SLEAP-node LK/CoTracker comparison using sealed prepared data.

No new SLEAP/D2 inference, triangulation, fit or identity decisions occur here.
The official backend is frozen; the new runner streams200 frames per view and
records first emissions, revisions, native release times and every query label.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import sys
import time
from pathlib import Path

import compare_temporal_known_gap64 as base
import cv2
import numpy as np
import semantic_node_stream as streaming
import temporal_point_core as core
import temporal_prefix_deltas as delta

WORK, REPO = base.WORK, base.REPO
FIRST, STOP, COUNT = 333519, 333719, 200
VIEWS = base.VIEWS
STAGE = "semantic_node200_tracking_v1"
PREPARATION = WORK / "semantic_node200_preparation_v2"
PREPARATION_STAGE = "semantic_node200_preparation_v2"
GEOMETRY_SHA = "149d159aea1a1f370f912aebfc616c3bc6d31b24732e704bb6d8e557c36b1fb4"
SOURCE_PINS = {
    "compare_temporal_known_gap64.py": (
        "9db089624f62f2d55bd9f883f9ec6ca8408604dd3d61f1a084d38884837272af"
    ),
    "temporal_point_core.py": "3ec308ca7877338f8677c0703d6f180a9d5658c4858b9868451e1152b7d4e25b",
    "temporal_window_stream.py": "e46859d868be4b79d1ffc1698a24bbc6eb693b1e8f191570978bbde663a04048",
    "temporal_prefix_deltas.py": "e164678a6e8900492d5a146a4b1c58f8edb360dd7455d178d0a25a591b6fef36",
    "semantic_node_stream.py": "498d7d57d07beb810540c643ba8a8f5c89a40cc5d837ffe373b99c9af34d035a",
    "semantic_node_queries.py": "ee181e77476161cf6c03f5642caa30a3b1b1e89a1e776205d200ba1e8a947be1",
    "qualify_cotracker3_cpu.py": base.QUALIFIER_SOURCE_SHA,
}
require = core.require


def arrays_from_bytes(raw):
    with np.load(io.BytesIO(raw), allow_pickle=False) as saved:
        require(len(saved.files) == len(set(saved.files)), "duplicate NPZ fields")
        return {name: saved[name].copy() for name in saved.files}


def bound_path(root, members, relative, expected=None):
    path = base.relative(root, relative)
    require(str(path) in members, "path missing from successful preparation")
    row = members[str(path)]
    require(expected is None or row["sha256"] == expected, "manifest/output SHA differs")
    return path


def checked_bytes(path, members):
    raw = path.read_bytes()
    row = members[str(path)]
    require(
        len(raw) == row["size_bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"],
        "prepared input changed at consumption",
    )
    return raw


def validate_queries(query, row, view):
    expected = {
        "frame_idx",
        "xy",
        "body_slot",
        "node_index",
        "source_row",
        "duplicate_pixel",
        "query_index",
    }
    require(set(query) == expected, "seven semantic query fields required")
    q = query["xy"]
    require(
        query["frame_idx"].shape == ()
        and query["frame_idx"].dtype == np.int64
        and int(query["frame_idx"]) == FIRST,
        "initial query native differs",
    )
    require(
        q.dtype == np.float32
        and q.shape == (9, 2)
        and np.isfinite(q).all()
        and ((q >= 0) & (q < 1200)).all(),
        "qualified9 native query pixels required",
    )
    for name in ("body_slot", "node_index", "source_row", "query_index"):
        require(query[name].shape == (9,) and query[name].dtype == np.int32, "query labels differ")
    require(
        np.array_equal(query["body_slot"], np.repeat(np.arange(3, dtype=np.int32), 3))
        and np.array_equal(query["node_index"], np.tile(np.arange(3, dtype=np.int32), 3))
        and np.array_equal(query["query_index"], np.arange(9, dtype=np.int32))
        and np.all(query["source_row"] >= 0),
        "fixed body/node/query order differs",
    )
    require(
        query["duplicate_pixel"].dtype == bool and query["duplicate_pixel"].shape == (9,),
        "duplicate flags differ",
    )
    duplicate = np.array([np.count_nonzero((q == p).all(axis=1)) > 1 for p in q], bool)
    require(np.array_equal(query["duplicate_pixel"], duplicate), "duplicate pixel flags changed")
    for node in range(3):
        require(len(np.unique(query["source_row"][node::3])) == 3, "initial source reuse")
    require(row["view"] == view, "query view differs")


def validate_native_rows(rows):
    require(isinstance(rows, list) and len(rows) == 600, "exact600 native rows required")
    keys = [(row["view"], row["frame_idx"]) for row in rows]
    require(
        len(set(keys)) == 600
        and set(keys) == {(view, native) for view in VIEWS for native in range(FIRST, STOP)},
        "full200 by3 native membership differs",
    )
    for row in rows:
        require(type(row["frame_idx"]) is int, "native frame type differs")
        require(row["native_offset"] == row["frame_idx"] - FIRST, "native offset differs")
        require(row["shape_hw"] == [1200, 1200], "native image shape differs")
        require(base.valid_sha(row["bgr_sha256"]), "native BGR signature absent")


def load_context(evidence, preparation_sha):
    for name, digest in SOURCE_PINS.items():
        evidence.add(WORK / name, digest)
    evidence.add(Path(__file__).resolve())
    seal, members = base.verified(
        PREPARATION,
        "preparation.provenance.json",
        PREPARATION_STAGE,
        preparation_sha,
        evidence,
    )
    geometry = WORK / "temporal_refractive_geometry_diagnosis_v2/diagnosis.provenance.json"
    evidence.add(geometry, GEOMETRY_SHA)
    require(seal["inputs"].get(str(geometry)) == GEOMETRY_SHA, "initial geometry seal not bound")
    plan, summary, natives, query_rows = (
        base.bound_json(PREPARATION, members, name)
        for name in ("plan.json", "summary.json", "native_evidence.json", "query_rows.json")
    )
    validate_native_rows(natives)
    require(
        [row["view"] for row in query_rows] == list(VIEWS),
        "three ordered query rows required",
    )
    queries = {}
    for row in query_rows:
        path = bound_path(PREPARATION, members, row["path"], row["sha256"])
        q = arrays_from_bytes(checked_bytes(path, members))
        validate_queries(q, row, row["view"])
        queries[row["view"]] = q
    for row in natives:
        for field, digest in (
            ("png_path", "png_sha256"),
            ("observation_path", "observation_sha256"),
        ):
            bound_path(PREPARATION, members, row[field], row[digest])
    slots = base.bound_json(PREPARATION, members, "query_slots.json")
    node_sets = base.bound_json(PREPARATION, members, "node_sets.json")
    require(len(slots) == 27 and len(node_sets) == 9, "all initial slot records required")
    return dict(
        seal=seal,
        members=members,
        plan=plan,
        summary=summary,
        natives=natives,
        query_rows=query_rows,
        queries=queries,
        slots=slots,
        node_sets=node_sets,
    )


def load_model_evidence(evidence):
    _, members = base.verified(
        base.ASSETS,
        "acquisition.provenance.json",
        "official_cotracker3_online_asset_acquisition_v2",
        base.ASSET_SEAL_SHA,
        evidence,
    )
    manifest = base.bound_json(base.ASSETS, members, "assets.manifest.json")
    require(
        base.sha(base.ASSETS / "assets.manifest.json") == base.ASSET_MANIFEST_SHA
        and manifest["upstream_weight_sha256"] == base.WEIGHT_SHA
        and manifest["upstream_weight_size_bytes"] == 101695610,
        "official assets differ",
    )
    evidence.add(base.SOURCE_REVIEW, base.SOURCE_REVIEW_SHA)
    closure = base.read_json(base.SOURCE_REVIEW)
    require(
        closure["asset_seal_sha256"] == base.ASSET_SEAL_SHA
        and closure["asset_manifest_sha256"] == base.ASSET_MANIFEST_SHA,
        "closure binding differs",
    )
    for item in closure["closure"].values():
        bound_path(base.ASSETS, members, item["path"], item["sha256"])
    record, qmembers = base.verified(
        base.QUALIFICATION,
        "qualification.provenance.json",
        "cotracker3_cpu_load_qualification_v1",
        base.MODEL_QUALIFICATION_SHA,
        evidence,
    )
    base.validate_model_qualification(base.bound_json(base.QUALIFICATION, qmembers, "summary.json"))
    require(
        record["inputs"].get(str(WORK / "qualify_cotracker3_cpu.py")) == base.QUALIFIER_SOURCE_SHA,
        "qualified loader source differs",
    )
    return closure


def snapshot_sources(evidence, output, captured):
    paths, inventory = base.source_inventory(dict(sys.modules), {Path(__file__).resolve()})
    index = sum(name.startswith("source_inventory/") for name in output.records)
    output.write(f"source_inventory/{index:03}.json", inventory)
    for path in sorted(paths):
        digest = evidence.add(path)
        if str(path) in captured:
            require(captured[str(path)] == digest, "loaded source changed")
        else:
            raw = path.read_bytes()
            require(hashlib.sha256(raw).hexdigest() == digest, "source snapshot changed")
            output.write("code_snapshot/" + path.relative_to(REPO).as_posix(), raw)
            captured[str(path)] = digest


def load_masks(context, row):
    path = bound_path(PREPARATION, context["members"], row["observation_path"])
    arrays = arrays_from_bytes(checked_bytes(path, context["members"]))
    require(
        arrays["frame_idx"].dtype == np.int64
        and arrays["frame_idx"].shape == ()
        and int(arrays["frame_idx"]) == row["frame_idx"],
        "observation native differs",
    )
    masks, available = arrays["masks"], arrays["mask_available"]
    require(available.shape == () and available.dtype == bool, "mask availability scalar differs")
    require(
        masks.dtype == bool
        and masks.shape == (row["mask_count"], 1200, 1200)
        and bool(available) == row["mask_available"],
        "allN mask schema differs",
    )
    return masks if available else None


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
    output.write(f"{view}/lk.npz", lk.finish())
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
                body_slot=int(query["body_slot"][i]),
                node_index=int(query["node_index"][i]),
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
        cv_threads = context["seal"]["environment"]["opencv_threads"]
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
        plan = dict(
            schema=STAGE,
            native_interval=[FIRST, STOP],
            views=list(VIEWS),
            preparation_sha256=args.preparation_sha256,
            initial_geometry_sha256=GEOMETRY_SHA,
            source_pins=SOURCE_PINS,
            model_asset_sha256=base.ASSET_SEAL_SHA,
            initial_query_count=27,
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
            retrospective_initial_body_slots=True,
            new_3D=False,
            new_fit=False,
            ID_or_review_changed=False,
            environment=environment,
        )
        output.write("plan.json", plan)
        output.write("query_slots.json", context["slots"])
        output.write("node_sets.json", context["node_sets"])
        evidence.final_guard()
        for view in VIEWS:
            views.append(run_view(context, view, closure, output, args.device, memory))
            snapshot_sources(evidence, output, captured)
            print(
                json.dumps(dict(event="view_complete", view=view, calls=24, queries=9)), flush=True
            )
        summary = dict(
            schema=STAGE,
            status="ok",
            native_interval=[FIRST, STOP],
            views=views,
            query_count=27,
            all200_query_denominator=5400,
            tracking199_query_denominator=5373,
            model_initializations=3,
            model_forward_calls=72,
            saved_png_decodes=600,
            lk_view_edges=597,
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
