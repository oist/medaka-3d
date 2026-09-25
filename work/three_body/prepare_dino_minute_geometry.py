"""Seal fixed60 original/Tail geometry before any DINO-distance join.

All anonymous masks and latent competitors remain. Only geometry/core conditions
are evaluated; no RGB/crop image array, HSV feature, DINO token/rank, human label,
raw video, optimizer, tracker or GPU operation is used. Existing full-horizon
39-state/21-backtrace proofs are reused by exact candidate SHA; selected60 source
geometry is recomputed. Availability is conditional, never biological identity.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import diagnose_identity_components as identity
import dino_mask_partial_correspondence as partial
import numpy as np
import prepare_dino_minute_crops as crop
from audit_long_final import verify_support_partitioned
from observations import CachedNodes

coh, checks = identity.coh, identity.checks
REPO, WORK = crop.REPO, crop.WORK
FRAMES, VIEWS = crop.FRAMES, tuple(crop.VIEWS)
METHODS = ("original", "tail")
STAGE = "dino_minute_adjacent_geometry_preparation_v1"
CROP_ROOT = WORK / "dino_minute_adjacent_crop_preparation_v1"
CROP_SOURCE_SHA = "47913aba5d44d7cd0516b75693b72bb4bce786f55cfea662f4b8be10bf8cb926"
TAIL_SEAL_SHA = "50ccc8d2a4bbcc7e09aa2b512b70374fb9b37a4d4ece0d0939322c66bfa59416"
PINS = {
    crop.PLAN: crop.PLAN_SHA,
    Path(crop.__file__).resolve(): CROP_SOURCE_SHA,
    Path(
        identity.__file__
    ).resolve(): "eba8c6d64da9f797afd1a08c72bb5886b7c7fd85af93e3c67845101c16d04040",
    Path(
        coh.__file__
    ).resolve(): "39aa2b6bbcd24200b4c1326ec923c06f3e331a20d4624687630bdbbb1dd487fa",
    Path(
        partial.__file__
    ).resolve(): "de487b2095c09c27be14e7fec92980d5a84f8c0ec6bf08017d7342aab145e400",
    WORK
    / "audit_long_final.py": "9140be0d5401228f38fefa04b1d4d4e68ea672ab8a0793130fbd62952c3e61eb",
    WORK / "audit_persistent_minute.py": (
        "31d5d9796c5ba10e7f211381c8b37a5fe2dbb3367ff2e829b18b3ebd4f94d0bc"
    ),
    Path(
        identity.appearance.__file__
    ).resolve(): "17a03a828958bba2402710f59bd9071f1c83649268ca1b3f62863f880c2ff135",
}

# Exact frozen finish schema: all 21 fields have native-time axis zero. There is
# no generic slicing of arbitrary first dimensions or unrecognized static data.
CANDIDATE_FIELDS = {
    "frame_idx": ((), "i"),
    "pose_xyz": ((3, 3, 3), "f"),
    "observed": ((3, 3), "b"),
    "sources": ((3, 3, 3), "i"),
    "identity_review": ((3,), "b"),
    "mask_supported": ((3,), "b"),
    "body_supported": ((3,), "b"),
    "missing": ((3,), "i"),
    "costs": ((), "f"),
    "path_costs": ((), "f"),
    "loss_parts": ((5,), "f"),
    "score_margin": ((), "f"),
    "proposal_kind": ((), "U"),
    "hypotheses": ((), "i"),
    "proposal_flow_nodes": ((3, 3), "b"),
    "proposal_flow_parent": ((), "i"),
    "node_missing": ((3, 3), "i"),
    "direct_pectoral_age": ((3,), "i"),
    "xyz": ((3, 3), "f"),
    "accepted_xyz": ((3, 3), "f"),
    "body_xyz": ((3, 3), "f"),
}
require, signature, read_json = crop.require, crop.signature, crop.read_json


def validate_candidate(arrays, *, first=49800, stop=76360, frames=FRAMES):
    """Check the frozen field schema and preserve each original dtype when slicing."""
    require(set(arrays) == set(CANDIDATE_FIELDS), "exact21 candidate keys required")
    count = stop - first
    for key, (shape, kind) in CANDIDATE_FIELDS.items():
        value = arrays[key]
        require(
            isinstance(value, np.ndarray)
            and value.shape == (count, *shape)
            and value.dtype.kind == kind
            and not value.dtype.hasobject,
            "candidate field shape/dtype differs: " + key,
        )
    require(np.array_equal(arrays["frame_idx"], np.arange(first, stop)), "candidate native range")
    require(
        np.isfinite(arrays["pose_xyz"]).all() and (arrays["sources"] >= -1).all(),
        "finite poses/in-range source sentinel required",
    )
    expected = {
        "xyz": np.where(arrays["observed"][:, :, 1, None], arrays["pose_xyz"][:, :, 1], np.nan),
        "body_xyz": np.where(
            arrays["body_supported"][:, :, None], arrays["pose_xyz"][:, :, 1], np.nan
        ),
    }
    expected["accepted_xyz"] = np.where(
        arrays["identity_review"][:, :, None], np.nan, expected["xyz"]
    )
    for key, value in expected.items():
        require(checks.equal_bytes(arrays[key], value), "derived candidate differs: " + key)
    selected = np.asarray(frames)
    require(
        selected.ndim == 1
        and selected.dtype.kind in "iu"
        and len(selected) > 0
        and np.all(np.diff(selected) > 0)
        and selected[0] >= first
        and selected[-1] < stop,
        "selected native order/range",
    )
    return {key: value[selected - first].copy() for key, value in arrays.items()}


def core_rows(masks, crops, native, view):
    """All N core counts; no RGB or feature extraction needed for the >=12 condition."""
    require(masks.dtype == bool and masks.ndim == 3, "allN bool masks")
    require(len(crops) == len(masks), "crop rows do not preserve allN")
    rows = []
    for index, (mask, saved) in enumerate(zip(masks, crops, strict=True)):
        require(
            saved["frame_idx"] == native
            and saved["view"] == view
            and saved["mask_row"] == index
            and saved["crop_id"] == f"f{native}_{view}_m{index:03d}",
            "crop identity/order",
        )
        core = partial.eroded_core(mask)
        count = int(core.sum())
        require(
            saved["mask_sha256"] == signature(mask)["sha256"]
            and saved["core_sha256"] == signature(core)["sha256"]
            and saved["core_pixels"] == count,
            "core/mask bytes differ from crop preparation",
        )
        rows.append(
            dict(
                mask_row=index,
                crop_id=saved["crop_id"],
                mask_pixels=int(mask.sum()),
                core_pixels=count,
                core_valid=count >= identity.appearance.MIN_CORE_PIXELS,
                mask_sha256=saved["mask_sha256"],
                core_sha256=saved["core_sha256"],
                crop_path=saved["path"],
                crop_sha256=saved["sha256"],
                crop_payload_opened=False,
            )
        )
    return rows


def add_core_conditions(reference, cores, review):
    """Keep the frozen geometry decision untouched, including all latent competitors."""
    result = copy.deepcopy(reference)
    require(
        np.asarray(review).shape == (3,) and np.asarray(review).dtype == bool,
        "three saved review flags",
    )
    eligible = []
    for slot, row in enumerate(result["slots"]):
        index = row["assigned_mask_row"]
        valid = index >= 0 and bool(cores[index]["core_valid"])
        reasons = list(row["reasons"])
        if index >= 0 and not valid:
            reasons.append("assigned_core_invalid")
        row.update(
            core_valid=valid,
            core_pixels=cores[index]["core_pixels"] if index >= 0 else None,
            crop_id=cores[index]["crop_id"] if index >= 0 else None,
            identity_review=bool(review[slot]),
            endpoint_eligible=bool(row["geometry_eligible"] and valid),
            endpoint_reasons=reasons,
        )
        eligible.append(row["endpoint_eligible"])
    result.update(
        all_masks=cores,
        slot_endpoint_eligible=eligible,
        all3_endpoint_eligible=bool(reference["all3_geometry_eligible"] and all(eligible)),
        review_excluded_from_eligibility=False,
        token_validity_evaluated=False,
    )
    return result


def geometry_group(masks, available, scenes, observed, sources, reviews, camera, cores):
    """Two methods x three slots, shared allN distance transforms, exactly 33 3D points."""
    require(
        scenes.shape == (2, 3, 3, 3)
        and observed.shape == (2, 3, 3)
        and sources.shape == (2, 3, 3)
        and reviews.shape == (2, 3),
        "two method geometry shapes",
    )
    bodies = [[coh.project_body(pose, camera) for pose in scene] for scene in scenes]
    xyz = np.asarray([[b["points_xyz"] for b in method] for method in bodies])
    uv = np.asarray([[b["projected_uv"] for b in method] for method in bodies])
    fields = coh.sample_mask_fields(masks, available, uv.reshape(-1, 2))
    references = []
    for mi in range(2):
        projected = np.asarray(camera.project(scenes[mi].copy()))
        require(
            checks.equal_bytes(projected, uv[mi][:, [0, 16, 32]]),
            "saved endpoint/33point projection differs",
        )
        metrics = [
            coh.coherence_metrics(fields, np.arange(33) + (mi * 3 + slot) * 33) for slot in range(3)
        ]
        reference = identity.geometry_reference(
            masks, available, projected, observed[mi], sources[mi], metrics
        )
        references.append(add_core_conditions(reference, cores, reviews[mi]))
    return dict(points_xyz=xyz, projected_uv=uv, **fields), references


def verify_selected_support(arrays, nodes, rig):
    """Original partitioned helper on singleton native intervals, never relabel sparse time."""
    results, blocks = [], []
    for i in range(len(arrays["frame_idx"])):
        part = {
            key: arrays[key][i : i + 1] for key in ("frame_idx", "pose_xyz", "sources", "observed")
        }
        captured = []
        checked = verify_support_partitioned(
            part, nodes, rig, on_block=lambda meta, values, saved=captured: saved.append(values)
        )
        require(len(captured) == 1, "one exact native support block")
        results.append(checked)
        blocks.append(captured[0])
    output = {key: np.concatenate([block[key] for block in blocks]) for key in blocks[0]}
    summary = dict(
        native_frames=len(results),
        temporal_subsampling=True,
        frames=arrays["frame_idx"].tolist(),
        ray_gate_mm=2.0,
        native_pixel_gate=12.0,
        all_sources_and_observed_flags_pass=True,
        source_rows_checked=sum(r["source_rows_checked"] for r in results),
        observed_nodes_checked=sum(r["observed_nodes_checked"] for r in results),
    )
    for key in ("maximum_supported_ray_distance_mm", "maximum_supported_pixel_error"):
        values = [r[key] for r in results if r[key] is not None]
        summary[key] = max(values) if values else None
    return summary, output


def json_output(root, seal, name, hashes):
    path = (Path(root) / name).resolve()
    require(path.is_relative_to(Path(root).resolve()), "output path escaped root")
    crop.member(seal, "outputs", path, hashes)
    return read_json(path)


def crop_input(seal, path, digest):
    require(seal["inputs"].get(str(Path(path).resolve())) == digest, "crop input binding differs")


def validate_comparison(summary):
    require(
        summary.get("schema") == "tail_one_minute_comparison_v1"
        and summary.get("status") == "ok"
        and summary.get("native_interval") == [66400, 76360]
        and summary.get("horizon_stop") == 76360
        and summary.get("native_frames") == 9960
        and all(
            summary.get(k) is True
            for k in (
                "same_parent_and_horizon",
                "one_minute_completed",
                "all39_arrays_semantically_decoded",
                "all21_arrays_exact_backtrace",
            )
        ),
        "completed same76360 comparison required",
    )
    for method, (path, digest) in crop.CANDIDATES.items():
        require(
            Path(summary["candidate_paths"][method]).resolve() == path.resolve()
            and summary["candidate_sha256"][method] == digest,
            "fixed candidate origin changed",
        )


def producer_contract(method):
    require(method in METHODS, "only fixed original/Tail producers")
    return {
        "original": (
            "continuation.provenance.json",
            "three_body_long_continuation",
            crop.ORIGINAL_SHA,
            "checkpoint/metadata.json",
        ),
        "tail": (
            "tracking.provenance.json",
            "tail_one_minute_continuous",
            TAIL_SEAL_SHA,
            "checkpoints/through_076360/metadata.json",
        ),
    }[method]


def bound_metadata(root, method, producer_seal, comparison, hashes):
    name, stage, _, relative = producer_contract(method)
    require(
        producer_seal.get("status") == "ok" and producer_seal.get("stage") == stage,
        "metadata producer stage differs: " + name,
    )
    path = Path(root) / relative
    digest = crop.member(producer_seal, "outputs", path, hashes)
    require(
        crop.member(comparison, "inputs", path, hashes) == digest,
        "producer/comparison metadata SHA differs",
    )
    return read_json(path)


def load_inputs(crop_sha, hashes, code):
    require(
        isinstance(crop_sha, str)
        and len(crop_sha) == 64
        and all(c in "0123456789abcdef" for c in crop_sha),
        "explicit crop success SHA required",
    )
    preparation = crop.header(
        CROP_ROOT, "preparation.provenance.json", crop_sha, crop.STAGE, hashes
    )
    # Only directly reused output bytes are read: no RGB PNG/crop NPZ or feature files.
    plan = json_output(CROP_ROOT, preparation, "plan.json", hashes)
    native_rows = json_output(CROP_ROOT, preparation, "native_evidence.json", hashes)
    crops = json_output(CROP_ROOT, preparation, "crops.json", hashes)
    prepared = json_output(CROP_ROOT, preparation, "summary.json", hashes)
    producer = Path(crop.__file__).resolve()
    crop_input(preparation, producer, CROP_SOURCE_SHA)
    snapshot = CROP_ROOT / "code_snapshot" / producer.relative_to(REPO)
    require(
        crop.member(preparation, "outputs", snapshot, hashes) == CROP_SOURCE_SHA,
        "crop producer snapshot differs",
    )
    require(
        plan["frames"] == list(FRAMES)
        and plan["views"] == list(VIEWS)
        and plan["source_plan_sha256"] == crop.PLAN_SHA
        and prepared["schema"] == crop.STAGE
        and prepared["status"] == "ok"
        and prepared["native_frames"] == 60
        and prepared["model_forward_calls"] == 0
        and prepared["candidate_arrays_read"] is False,
        "fixed crop preparation scope",
    )
    require(
        [(r["frame_idx"], r["view"]) for r in native_rows]
        == [(n, v) for n in FRAMES for v in VIEWS],
        "all180 native observation rows required",
    )
    counts = {(r["frame_idx"], r["view"]): r["mask_count"] for r in native_rows}
    expected_plan = crop.enumerate_plan(counts)
    for key, value in expected_plan.items():
        require(plan[key] == value, "crop predeclared enumeration differs: " + key)
    require(
        [r["crop_id"] for r in crops] == [r["crop_id"] for r in plan["crops"]]
        and prepared["mask_crops"] == len(crops),
        "all crop rows/order required",
    )
    comparison = crop.header(
        crop.COMPARISON,
        "comparison.provenance.json",
        crop.COMPARISON_SHA,
        "tail_one_minute_comparison_v1",
        hashes,
    )
    crop_input(preparation, crop.COMPARISON / "comparison.provenance.json", crop.COMPARISON_SHA)
    summary = json_output(crop.COMPARISON, comparison, "summary.json", hashes)
    crop_input(
        preparation,
        crop.COMPARISON / "summary.json",
        hashes[str((crop.COMPARISON / "summary.json").resolve())],
    )
    validate_comparison(summary)
    shared = summary["shared_SL_calibration_sha256"]
    require(
        shared == plan["shared_SL_calibration_sha256"]
        and len(shared) == 7
        and sum(Path(p).suffix == ".npz" for p in shared) == 3
        and sum(Path(p).suffix == ".json" for p in shared) == 4,
        "sameSL3/calib4",
    )
    for path, digest in shared.items():
        crop_input(preparation, path, digest)
        require(
            crop.member(comparison, "inputs", path, hashes) == digest, "geometry source differs"
        )
    calibration = next(
        Path(p).parent for p in shared if Path(p).name == "calibration_manifest.json"
    )
    manifest = read_json(calibration / "calibration_manifest.json")
    paths = {str((calibration / r["path"]).resolve()) for r in manifest["model_files"]}
    paths.add(str((calibration / "calibration_manifest.json").resolve()))
    require(
        paths == {p for p in shared if Path(p).suffix == ".json"}, "rig loader calibration closure"
    )
    rig = coh.load_rig(calibration)
    slices, candidate_specs = {}, {}
    input_root = None
    for method in METHODS:
        path, digest = crop.CANDIDATES[method]
        crop_input(preparation, path, digest)
        require(
            plan["candidate_references"][method] == dict(path=str(path), sha256=digest)
            and crop.member(comparison, "inputs", path, hashes) == digest,
            "candidate origin differs",
        )
        root = path.parent
        seal_name, stage, seal_sha, _ = producer_contract(method)
        producer_seal = crop.header(root, seal_name, seal_sha, stage, hashes)
        require(
            crop.member(comparison, "inputs", root / seal_name, hashes) == seal_sha
            and crop.member(producer_seal, "outputs", path, hashes) == digest,
            "comparison/candidate producer binding",
        )
        metadata = bound_metadata(root, method, producer_seal, comparison, hashes)
        coh.validate_rig_context(metadata["context"], rig)
        for p, h in shared.items():
            require(
                crop.member(producer_seal, "inputs", p, hashes) == h, "candidate SL/calib input"
            )
        if method == "original":
            input_root = producer_seal["params"]["bindings"]["inputs"]["seed"]["input_root"]
        with np.load(path, allow_pickle=False) as archive:
            require(len(archive.files) == len(set(archive.files)) == 21, "duplicate/missing21 keys")
            arrays = {key: archive[key] for key in archive.files}
        slices[method] = validate_candidate(arrays)
        candidate_specs[method] = {
            key: dict(signature(value), frame_axis=0) for key, value in arrays.items()
        }
        del arrays
        for p, h in code.items():
            if p in producer_seal["inputs"]:
                require(
                    producer_seal["inputs"][p]["sha256"] == h, "historical candidate code changed"
                )
    for p, h in code.items():
        if p in preparation["inputs"]:
            require(preparation["inputs"][p] == h, "historical crop code changed")
        if p in comparison["inputs"]:
            require(comparison["inputs"][p]["sha256"] == h, "historical comparison code changed")
    # Frozen loader requires a contiguous interval; only the fixed60 rows are
    # subsequently queried/reprojected. No sparse index relabelling is used.
    nodes = CachedNodes(input_root, crop.native.EXPERIMENT, np.arange(crop.START, crop.STOP))
    require(
        {str(Path(p).resolve()) for p in nodes.paths.values()}
        == {p for p in shared if Path(p).suffix == ".npz"},
        "SL loader exact3 file closure",
    )
    return dict(
        preparation=preparation,
        plan=plan,
        native_rows=native_rows,
        crops=crops,
        slices=slices,
        candidate_specs=candidate_specs,
        shared=shared,
        rig=rig,
        nodes=nodes,
    )


def observation(row, seal, nodes, hashes):
    path = CROP_ROOT / row["observation_path"]
    require(path.resolve().is_relative_to(CROP_ROOT.resolve()), "observation path scope")
    require(
        crop.member(seal, "outputs", path, hashes) == row["observation_sha256"],
        "observation seal/native evidence differs",
    )
    with np.load(path, allow_pickle=False) as archive:
        require(len(archive.files) == len(set(archive.files)), "duplicate observation keys")
        arrays = {key: archive[key] for key in archive.files}
    require(
        {k: signature(v) for k, v in arrays.items()} == row["observation_array_signatures"],
        "observation logical array signatures",
    )
    masks, available = arrays["masks"], arrays["mask_available"]
    require(
        masks.dtype == bool
        and masks.ndim == 3
        and masks.shape[1:] == (1200, 1200)
        and available.shape == ()
        and available.dtype == bool,
        "native allN mask schema",
    )
    flag = bool(available)
    require(
        (flag or len(masks) == 0)
        and (len(masks) if flag else None) == row["mask_count"]
        and signature(masks)["sha256"] == row["mask_sha256"],
        "missing/zero/allN mask binding",
    )
    require(
        signature(arrays["detections"]) == row["detection_signature"]
        and checks.equal_bytes(arrays["detections"], nodes.at(row["view"], row["frame_idx"])),
        "native SL rows differ",
    )
    return masks, flag


def source_code():
    result = coh.source_code()
    result[str(Path(__file__).resolve())] = checks.sha(__file__)
    for path, digest in PINS.items():
        require(checks.sha(path) == digest, "pinned source/plan changed")
        result[str(path)] = digest
    return result


def run(output, crop_sha):
    output = Path(output).resolve()
    require(output.is_relative_to(WORK) and not output.exists(), "new geometry output required")
    code, hashes, outputs = source_code(), {}, {}
    started = crop.assets.now()
    wall = time.perf_counter()
    output.mkdir(parents=True)

    def record(path):
        outputs[path.relative_to(output).as_posix()] = dict(
            path=str(path), sha256=checks.sha(path), size_bytes=path.stat().st_size
        )

    def write(name, value):
        path = output / name
        crop.assets.write_json(path, coh.encode_nonfinite(value))
        record(path)

    def dump(name, arrays):
        path = output / name
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        record(path)

    try:
        for path, digest in code.items():
            crop.bind(path, digest, hashes)
            dest = output / "code_snapshot" / Path(path).relative_to(REPO)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
            require(checks.sha(dest) == digest, "source snapshot changed")
            record(dest)
        data = load_inputs(crop_sha, hashes, code)
        write(
            "plan.json",
            dict(
                schema=STAGE,
                source_plan_sha256=crop.PLAN_SHA,
                crop_preparation_sha256=crop_sha,
                frames=FRAMES,
                methods=METHODS,
                views=VIEWS,
                horizon_stop=76360,
                method_view_groups=360,
                slot_records=1080,
                body_points=33,
                mask_policy="allN plus union; preserve missing/zero/empty/duplicate/ties",
                core_policy="frozen5x5ellipse; minimum12pixels; no descriptor",
                feature_or_rank_selection=False,
                review_excluded_from_eligibility=False,
                sparse_native_support="60 independent singleton intervals per method, 2mm/12px",
                candidate_field_schema={
                    k: dict(frame_axis=0, trailing_shape=s, dtype_kind=t)
                    for k, (s, t) in CANDIDATE_FIELDS.items()
                },
                geometry_npz=(
                    "180 files: point order method(2),slot(3),bodypoint(33), C-order flatten"
                ),
                nonfinite_JSON="NaN/+inf/-inf strings; NPZ preserves IEEE values",
                full_horizon_proof_reused=crop.COMPARISON_SHA,
            ),
        )
        write("candidate_source_array_signatures.json", data["candidate_specs"])
        support = {}
        for method in METHODS:
            dump(f"candidate_slices_{method}.npz", data["slices"][method])
            support[method], fields = verify_selected_support(
                data["slices"][method], data["nodes"], data["rig"]
            )
            dump(f"strict_support_{method}.npz", fields)
        references = []
        all_crop_rows = {(n, v): [] for n in FRAMES for v in VIEWS}
        for row in data["crops"]:
            all_crop_rows[row["frame_idx"], row["view"]].append(row)
        for row in data["native_rows"]:
            n, view = row["frame_idx"], row["view"]
            fi, vi = FRAMES.index(n), VIEWS.index(view)
            masks, available = observation(row, data["preparation"], data["nodes"], hashes)
            cores = core_rows(masks, all_crop_rows[n, view], n, view)
            traces = [data["slices"][m] for m in METHODS]
            arrays, refs = geometry_group(
                masks,
                available,
                np.stack([t["pose_xyz"][fi] for t in traces]),
                np.stack([t["observed"][fi] for t in traces]),
                np.stack([t["sources"][fi, :, vi] for t in traces]),
                np.stack([t["identity_review"][fi] for t in traces]),
                data["rig"].camera(view),
                cores,
            )
            arrays.update(
                frame_idx=np.asarray(n, np.int64),
                methods=np.asarray(METHODS),
                view=np.asarray(view),
                mask_available=np.asarray(available),
                mask_count=np.asarray(len(masks), np.int64),
            )
            name = f"geometry_{n}_{view}.npz"
            dump(name, arrays)
            for method, ref in zip(METHODS, refs, strict=True):
                ref.update(
                    method=method,
                    frame_idx=n,
                    view=view,
                    mask_available=available,
                    mask_count=len(masks) if available else None,
                    observation_path=row["observation_path"],
                    observation_sha256=row["observation_sha256"],
                    geometry_path=name,
                    geometry_sha256=outputs[name]["sha256"],
                )
                references.append(ref)
            if vi == 2:
                print(json.dumps(dict(event="geometry_native_complete", frame_idx=n)), flush=True)
        require(
            len(references) == 360 and sum(len(r["slots"]) for r in references) == 1080,
            "fixed all1080 denominator",
        )
        write("geometry_references.json", references)
        summary = dict(
            schema=STAGE,
            status="ok",
            frames=list(FRAMES),
            views=list(VIEWS),
            methods=list(METHODS),
            native_frames=60,
            horizon_stop=76360,
            geometry_groups=360,
            slot_records=1080,
            points_per_body=33,
            geometry_precedes_feature_join=True,
            candidate_sha256={m: crop.CANDIDATES[m][1] for m in METHODS},
            comparison_sha256=crop.COMPARISON_SHA,
            crop_preparation_sha256=crop_sha,
            shared_SL_calibration_sha256=data["shared"],
            strict_support=support,
            method_counts={
                m: dict(
                    requested_slots=540,
                    geometry_eligible=sum(
                        s["geometry_eligible"]
                        for r in references
                        if r["method"] == m
                        for s in r["slots"]
                    ),
                    endpoint_eligible=sum(
                        s["endpoint_eligible"]
                        for r in references
                        if r["method"] == m
                        for s in r["slots"]
                    ),
                    all3_endpoint_groups=sum(
                        r["all3_endpoint_eligible"] for r in references if r["method"] == m
                    ),
                    saved_review_slots=sum(
                        s["identity_review"]
                        for r in references
                        if r["method"] == m
                        for s in r["slots"]
                    ),
                    reason_counts=dict(
                        Counter(
                            reason
                            for r in references
                            if r["method"] == m
                            for s in r["slots"]
                            for reason in s["endpoint_reasons"]
                        )
                    ),
                )
                for m in METHODS
            },
            RGB_or_feature_arrays_read=False,
            raw_video_decodes=0,
            model_forward_calls=0,
            fit_calls=0,
            advance_calls=0,
            GPU=False,
            ID_or_review_changed=False,
            independent_ID_truth=False,
            all_thirty_minutes_complete=False,
            reused_full39_and21_audit=True,
            full_checkpoint_arrays_reopened=False,
            verification_scope=(
                "directly consumed sealed metadata/observations/candidates/code, "
                "not all upstream RGB/crop payloads"
            ),
            elapsed_seconds=time.perf_counter() - wall,
        )
        write("summary.json", summary)
        require(source_code() == code, "late workspace imports/source changed")
        crop.prior.final_guard(hashes)
        for item in outputs.values():
            require(checks.sha(item["path"]) == item["sha256"], "output changed before seal")
        crop.assets.write_json(
            output / "geometry.provenance.json",
            dict(
                stage=STAGE,
                status="ok",
                started_at=started,
                finished_at=crop.assets.now(),
                inputs=hashes,
                outputs=outputs,
                model_or_rank_read=False,
                environment=dict(
                    python=coh.sys.version,
                    numpy=np.__version__,
                    scipy=coh.scipy.__version__,
                    opencv=crop.cv2.__version__,
                ),
            ),
        )
        return summary
    except BaseException as error:
        crop.assets.write_json(
            output / "geometry.provenance.json",
            dict(
                stage=STAGE,
                status="failed",
                started_at=started,
                finished_at=crop.assets.now(),
                inputs=hashes,
                outputs=outputs,
                error_type=type(error).__name__,
                error=str(error),
            ),
        )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crop-preparation-sha256", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.output, args.crop_preparation_sha256), indent=2))


if __name__ == "__main__":
    main()
