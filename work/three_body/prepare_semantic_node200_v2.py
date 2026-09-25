"""Prepare fixed semantic-node200 inputs; no model, tracking or original MP4.

The audited legacy FFV1 clips are decoded once and checked against all600
canonical native BGR hashes. Initial labels are conditional retrospective body
slots; only the original registered seed is offered for later initialization.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import prepare_temporal_known_gap64 as old
import semantic_node_queries as selector
from observations import CachedNodes

prior = old.prior
bind, header, member, members = old.bind, old.header, old.member, old.members
read_json, write_json, require = old.read_json, old.write_json, old.require
sha, signature, native, now = old.sha, old.signature, old.native, old.now
REPO, WORK, VIEWS = old.REPO, old.WORK, old.VIEWS
START, STOP = 333519, 333719
FRAMES = tuple(range(START, STOP))
STAGE = "semantic_node200_preparation_v2"
FAILED = WORK / "semantic_node200_preparation_v1"
FAILED_SHA = "58634e75f32393a22716fe4712b516b7eb68d52f793a608ac5831d34f3f788ea"
FAILED_SOURCE = WORK / "prepare_semantic_node200.py"
FAILED_SOURCE_SHA = "a23ffbdec69c7232db9eec37322dbd6713b8412e944604cd1b6032f555da8560"
PLAN = WORK / "SEMANTIC_NODE_200_IMPLEMENTATION_PLAN.md"
PLAN_SHA = "82a196625e8fa80fd3f9867ed65e4428a55301e33c39d7d6620648dca701fc3a"
OLD_SOURCE_SHA = "14ea5b78402de77040abddc84f4874593d7f82a0dfc0d94a03aefeef51930a25"
SELECTOR_SHA = "ee181e77476161cf6c03f5642caa30a3b1b1e89a1e776205d200ba1e8a947be1"
GEOMETRY = WORK / "temporal_refractive_geometry_diagnosis_v2"
GEOMETRY_SHA = "149d159aea1a1f370f912aebfc616c3bc6d31b24732e704bb6d8e557c36b1fb4"
PREFIX = WORK / "temporal_known_gap64_preparation_v1"
PREFIX_SHA = "447a713a715ccb900883bcebff42ab629fbd55f422b36a0e633ad6ed7f544c8e"
REFERENCE = WORK / "continuation_transient_200_concurrent_v1"
REFERENCE_SHA = "163135a1f70c25bcd8251b5f5077c4b6eecc5b707e6df09d44466d4b8b5c05ce"
ORIGINAL = WORK / "persistent_mouth_transient_200_v1"
ORIGINAL_SHA = "7084161d4e43fbfdb1e5621afbd53246f1371a57748d912f77fa78d4b3ca18e1"
SEED_SHA = "576254e497bf7dff8d8346e8ab591164ffca65783f56d4d2d660ed8b74f9c56a"
METADATA_SHA = "7c089f5b3da8239bde679cb159c71866260f6fe8e290e7035bdfd7c2b154e7d1"


def json_member(root, table, name, evidence, expected=None):
    path = root / name
    return read_json(bind(path, member(table, path, expected)["sha256"], evidence))


def seed_contract(seed, metadata, summary, origin):
    """Establish original constructor slot order, without using fitted poses."""
    initial = np.asarray(seed["pose_xyz"], np.float64)
    context = metadata["context"]
    require(
        seed["frame"] == START
        and seed["initialization_count"] == 1
        and seed["later_baseline_input"] is False
        and initial.shape == (3, 3, 3)
        and np.isfinite(initial).all()
        and signature(initial) == context["initial_pose"]
        and signature(np.linalg.norm(np.diff(initial, axis=1), axis=-1))
        == context["initial_lengths"],
        "original registered seed/context differs",
    )
    require(
        context["views"] == list(VIEWS)
        and context["world_units"] == "mm"
        and context["implementation"] == "AcceleratedContinuousPersistentNodeTracker"
        and metadata["initialize_count"] == 1
        and metadata["restore_count"] == 0
        and metadata["processed_frames"] == 200
        and metadata["previous_frame"] == STOP - 1
        and summary["original_candidate_21_arrays_byte_exact"] is True
        and summary["original_candidate_loaded_only_after_full_inference"] is True
        and origin["reference"]["seal"] == REFERENCE_SHA
        and origin["reference"]["metadata"] == METADATA_SHA
        and origin["reference"]["start"] == START
        and origin["reference"]["stop"] == STOP
        and origin["retrospective_reference"] is True,
        "reference/constructor slot contract differs",
    )
    return dict(
        schema="semantic_node200_original_seed_contract_v1",
        initial_pose_signature=signature(initial),
        constructor_body_slot_order=[0, 1, 2],
        node_order=["mouth", "pectoral", "tail"],
        original_seed_source_rows=seed["source"],
        original_seed_source_rows_are_whole_fish_registration=True,
        query_source_rows_are_per_node_and_need_not_equal_seed_rows=True,
        reference_candidate_sha256=origin["reference"]["candidate"],
        qualification_reference_horizon=[START, STOP],
        retrospective_conditional_body_labels=True,
        constructor_seed_is_original_registration=True,
        later_reference_poses_used_as_tracker_seeds=False,
        biological_ID_truth=False,
        original_seed_limitation=seed["limitation"],
    )


def metadata_inputs(evidence):
    """Small JSON/source contract; return unopened payload references."""
    records = {}
    for name, root, filename, digest, stage in (
        ("old", old.OLD, "staging.provenance.json", old.OLD_SHA, "prepare_three_body_sequence"),
        (
            "compact",
            old.COMPACT,
            "compact_sequence.provenance.json",
            old.COMPACT_SHA,
            "three_body_compact_source_sequence",
        ),
        (
            "prefix",
            PREFIX,
            "preparation.provenance.json",
            PREFIX_SHA,
            "temporal_known_gap64_preparation_v1",
        ),
        (
            "geometry",
            GEOMETRY,
            "diagnosis.provenance.json",
            GEOMETRY_SHA,
            "temporal_refractive_geometry_diagnosis_v2",
        ),
        (
            "reference",
            REFERENCE,
            "continuation.provenance.json",
            REFERENCE_SHA,
            "three_body_persistent_continuation",
        ),
        (
            "original",
            ORIGINAL,
            "experiment.provenance.json",
            ORIGINAL_SHA,
            "three_body_persistent_node_recovery_algorithm_arm",
        ),
    ):
        records[name] = header(root / filename, digest, stage, evidence)
    tables = {
        k: {
            g: members(v, g)
            for g in (("outputs",) if k in ("old", "original") else ("inputs", "outputs"))
        }
        for k, v in records.items()
    }
    oo, ci, co = tables["old"]["outputs"], tables["compact"]["inputs"], tables["compact"]["outputs"]
    po, gi, go = (
        tables["prefix"]["outputs"],
        tables["geometry"]["inputs"],
        tables["geometry"]["outputs"],
    )
    ri, ro = tables["reference"]["inputs"], tables["reference"]["outputs"]
    manifest = json_member(old.OLD, oo, "manifest.json", evidence, old.OLD_MANIFEST_SHA)
    compact = json_member(old.COMPACT, co, "manifest.json", evidence, old.MANIFEST_SHA)
    audit = json_member(old.COMPACT, co, "decode_audit.json", evidence, old.AUDIT_SHA)
    member(ci, old.OLD / "manifest.json", old.OLD_MANIFEST_SHA)
    detector = old.validate_metadata(manifest, compact, audit, records["old"])
    old.legacy_asset_roles(records["old"], manifest)
    require(compact["backgrounds"] == manifest["backgrounds"], "background descriptor drift")
    prefix_plan = json_member(PREFIX, po, "plan.json", evidence)
    prefix_rows = json_member(PREFIX, po, "native_evidence.json", evidence)
    shared = prefix_plan["shared_SL_calibration_sha256"]
    require(
        len(shared) == 7 and sum(Path(p).suffix == ".npz" for p in shared) == 3,
        "SL3/calibration4 required",
    )
    require(
        [(r["frame_idx"], r["view"]) for r in prefix_rows]
        == [(n, v) for n in range(START, START + 64) for v in VIEWS],
        "prefix192 differs",
    )
    member(gi, PREFIX / "preparation.provenance.json", PREFIX_SHA)
    member(gi, REFERENCE / "continuation.provenance.json", REFERENCE_SHA)
    origin = json_member(GEOMETRY, go, "known_gap/reference_origin.json", evidence)
    qualification = json_member(GEOMETRY, go, "initial_node_query_qualification.json", evidence)
    node_sets = json_member(GEOMETRY, go, "initial_node_sets.json", evidence)
    metadata = json_member(REFERENCE, ro, "checkpoint/metadata.json", evidence, METADATA_SHA)
    summary = json_member(REFERENCE, ro, "summary.json", evidence)
    seed = json_member(
        ORIGINAL, tables["original"]["outputs"], "seed_registration.json", evidence, SEED_SHA
    )
    member(ri, ORIGINAL / "seed_registration.json", SEED_SHA)
    member(ri, ORIGINAL / "experiment.provenance.json", ORIGINAL_SHA)
    require(
        records["reference"]["params"]["bindings"]["source_range"] == [START, STOP],
        "original reference input interval differs",
    )
    contract = seed_contract(seed, metadata, summary, origin)
    refs = []
    for path, digest in shared.items():
        for table in (gi, ri, tables["prefix"]["inputs"]):
            member(table, path, digest)
        refs.append(member(ri, path, digest))
    for category in ("raw", "masks", "backgrounds"):
        for view in VIEWS:
            info = manifest[category][view]
            path = native.checked_relative(old.OLD, info["path"])
            for table in (oo, ri):
                member(table, path, info["sha256"])
            if category == "raw":
                require(path.suffix.lower() == ".avi", "only audited legacy FFV1 accepted")
            refs.append(member(oo, path, info["sha256"]))
    for name in ("known_gap/reference_selected21.npz", f"known_gap/sources/native_{START}.npz"):
        refs.append(member(go, GEOMETRY / name))
    # Pin current loaded observation/decoder code to historical consumed originals.
    code = source_code()
    for path in (
        Path(sys.modules["observations"].__file__),
        Path(sys.modules["prepare_sequence"].__file__),
    ):
        member(ri, path, code[str(path.resolve())])
    for module in (native, sys.modules["chunk_masks"], prior, old):
        path = Path(module.__file__).resolve()
        member(tables["prefix"]["inputs"], path, code[str(path)])
    for path, digest in code.items():
        for table in (ri, tables["prefix"]["inputs"]):
            if path in table:
                member(table, path, digest)
    for path, digest in code.items():
        bind(path, digest, evidence)
    return dict(
        manifest=manifest,
        compact=compact,
        shared=shared,
        references=refs,
        prefix_rows=prefix_rows,
        origin=origin,
        qualification=qualification,
        node_sets=node_sets,
        seed=seed,
        metadata=metadata,
        seed_contract=contract,
        detector=detector,
    )


def retry_metadata(evidence):
    """Bind the failed transport attempt; never adopt its numerical outputs."""
    failed = read_json(bind(FAILED / "preparation.provenance.json", FAILED_SHA, evidence))
    require(
        failed["stage"] == "semantic_node200_preparation_v1"
        and failed["status"] == "failed"
        and failed.get("finished_at")
        and failed["error"] == {"type": "ValueError", "message": "lossless FFV1 codec required"},
        "fixed V1 transport failure differs",
    )
    member(members(failed, "inputs"), FAILED_SOURCE, FAILED_SOURCE_SHA)
    bind(FAILED_SOURCE, FAILED_SOURCE_SHA, evidence)
    return dict(
        schema="semantic_node200_FFV1_alias_retry_v1",
        failed_seal_sha256=FAILED_SHA,
        failed_source_sha256=FAILED_SOURCE_SHA,
        reason="FFMPEG reports lowercase ffv1 for the fixed AVI FFV1 stream",
        exact_reported_FourCC_aliases=["FFV1", "ffv1"],
        all600_new_preparation_planned=True,
        v1_payloads_adopted=False,
        pixel_SHA_native_mapping_query_and_geometry_policy_unchanged=True,
    )


def capture_metadata(capture, view):
    value = int(capture.get(cv2.CAP_PROP_FOURCC))
    return dict(
        view=view,
        backend=capture.getBackendName() if capture.isOpened() else None,
        reported_fourcc_integer=value,
        reported_fourcc_bytes=list(value.to_bytes(4, "little", signed=False)),
        reported_frame_count=capture.get(cv2.CAP_PROP_FRAME_COUNT),
        reported_fps=capture.get(cv2.CAP_PROP_FPS),
        reported_shape_hw=[
            capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
            capture.get(cv2.CAP_PROP_FRAME_WIDTH),
        ],
        RIFF_header_reparsed_by_this_preparation=False,
        stream_identity_authority="pinned full AVI SHA plus all200 canonical decoded BGR hashes",
    )


def read200(capture, hashes, *, shape=(1200, 1200)):
    """Clip offsets0..199 map to native333519..333718; no original-video fallback."""
    require(len(hashes) == len(FRAMES), "all200 native hashes required")
    # Frozen native.validate_capture requires the full348600 MP4 extent; this
    # separate transport is the registered200-frame FFV1 clip, never that MP4.
    require(capture.isOpened(), "cannot open audited FFV1")
    require(abs(capture.get(cv2.CAP_PROP_FPS) - 166) <= 1e-6, "FFV1 fps differs")
    require(
        (capture.get(cv2.CAP_PROP_FRAME_HEIGHT), capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        == tuple(shape),
        "FFV1 image dimensions differ",
    )
    require(capture.get(cv2.CAP_PROP_FRAME_COUNT) == len(FRAMES), "FFV1 extent differs")
    require(
        int(capture.get(cv2.CAP_PROP_FOURCC))
        in {cv2.VideoWriter_fourcc(*"FFV1"), cv2.VideoWriter_fourcc(*"ffv1")},
        "lossless FFV1 codec required",
    )
    require(capture.set(cv2.CAP_PROP_POS_FRAMES, 0), "clip initial seek failed")
    require(abs(capture.get(cv2.CAP_PROP_POS_FRAMES)) <= 1e-6, "clip initial offset differs")
    for offset, (n, digest) in enumerate(zip(FRAMES, hashes, strict=True)):
        ok, image = capture.read()
        require(
            ok
            and isinstance(image, np.ndarray)
            and image.dtype == np.uint8
            and image.shape == (*shape, 3),
            "FFV1 frame dtype/shape/EOF differs",
        )
        require(
            abs(capture.get(cv2.CAP_PROP_POS_FRAMES) - offset - 1) <= 1e-6,
            "clip/native mapping differs",
        )
        require(signature(image)["sha256"] == digest, "canonical native BGR differs")
        yield n, image


def observation(data, frame, count, points):
    require(type(frame) is int and START <= frame < STOP, "native outside fixed200")
    require(
        isinstance(points, np.ndarray)
        and points.dtype == np.float32
        and points.ndim == 3
        and points.shape[1:] == (3, 2),
        "raw SL f32[D,3,2] required",
    )
    point_before = signature(points)
    packed, decoded = native.select_packed_frame(data, frame - START, frame, count)
    arrays = dict(
        frame_idx=np.asarray(frame, np.int64),
        masks=decoded if decoded is not None else np.empty((0, 1200, 1200), bool),
        mask_available=np.asarray(decoded is not None),
        detections=points.copy(),
        **{"packed_" + k: v for k, v in packed.items()},
    )
    require(point_before == signature(points), "SL input changed")
    return arrays


def initial_queries(source, selected, origin, qualification, node_sets):
    """Consume saved strict qualification; compare all raw pixels/source labels."""
    require(
        set(selected) == set(origin["selected_array_signatures"]) and len(selected) == 21,
        "sealed selected21 key set differs",
    )
    require(
        all(signature(v) == origin["selected_array_signatures"][k] for k, v in selected.items()),
        "selected21 arrays differ",
    )
    require(
        np.array_equal(selected["frame_idx"], np.arange(START, START + 64)),
        "selected reference native range differs",
    )
    require(
        source["frame_idx"].shape == ()
        and source["frame_idx"].dtype == np.int64
        and int(source["frame_idx"]) == START,
        "initial source native differs",
    )
    for key in ("sources", "observed"):
        require(
            signature(source[key]) == signature(selected[key][0]), "initial source/selected drift"
        )
    rows = [r for r in qualification if r["window"] == "known_gap"]
    sets = [r for r in node_sets if r["window"] == "known_gap"]
    require(len(rows) == 27 and len(sets) == 9, "all27/9 initial qualification required")
    row_map = {(r["body"], r["node"], r["view"]): r for r in rows}
    require(
        set(row_map) == {(b, n, v) for b in range(3) for n in range(3) for v in VIEWS},
        "initial qualification duplicates/missing",
    )
    points = {v: source[v + "_points"] for v in VIEWS}
    for b in range(3):
        for n in range(3):
            for vi, view in enumerate(VIEWS):
                row = row_map[b, n, view]
                index = int(source["sources"][b, vi, n])
                eligible = bool(source["observed"][b, n] and index >= 0)
                require(
                    row["native"] == START
                    and row["source_row"] == index
                    and row["observed"] == bool(source["observed"][b, n])
                    and row["eligible"] == eligible,
                    "initial qualification status drift",
                )
                if index >= 0:
                    xy = points[view][index, n]
                    require(
                        row["source_array_key"] == view + "_points"
                        and row["source_array_indices"] == [index, n]
                        and row["source_xy_dtype"] == xy.dtype.str
                        and row["source_xy_sha256"] == signature(xy)["sha256"]
                        and np.array_equal(np.asarray(row["source_xy"], np.float32), xy),
                        "initial raw SL byte/label drift",
                    )
                    require(
                        np.isfinite(source["ray_distance_mm"][b, vi, n])
                        and source["ray_distance_mm"][b, vi, n] <= 2.0
                        and np.isfinite(source["pixel_error_px"][b, vi, n])
                        and source["pixel_error_px"][b, vi, n] <= 12.0,
                        "saved strict2mm/12px qualification differs",
                    )
    set_map = {(r["body"], r["node"]): r for r in sets}
    require(
        set(set_map) == {(b, n) for b in range(3) for n in range(3)}, "node set duplicates/missing"
    )
    for (b, n), row in set_map.items():
        views = [v for v in VIEWS if row_map[b, n, v]["eligible"]]
        require(
            row["qualified_views"] == views
            and row["count"] == len(views)
            and row["at_least_two"] == (len(views) >= 2),
            "node set qualification differs",
        )
    before = {k: signature(v) for k, v in source.items()}
    result = selector.select_queries(
        points, source["sources"], source["observed"], native_frame=START
    )
    require(
        before == {k: signature(v) for k, v in source.items()}, "query selection mutated source"
    )
    for arrays in result["views"].values():
        arrays["query_index"] = np.arange(len(arrays["xy"]), dtype=np.int32)
    return result, rows, sets


def source_code():
    result = old.source_code()
    result[str(Path(__file__).resolve())] = sha(__file__)
    result[str(Path(old.__file__).resolve())] = OLD_SOURCE_SHA
    result[str(Path(selector.__file__).resolve())] = SELECTOR_SHA
    result[str(PLAN)] = PLAN_SHA
    result[str(FAILED_SOURCE)] = FAILED_SOURCE_SHA
    return result


def run(output):
    output = Path(output).resolve()
    require(output.is_relative_to(WORK) and not output.exists(), "new workspace output required")
    prior.no_model_imports()
    cv2.setNumThreads(2)
    require(
        sha(PLAN) == PLAN_SHA
        and sha(old.__file__) == OLD_SOURCE_SHA
        and sha(selector.__file__) == SELECTOR_SHA,
        "frozen source/plan drift",
    )
    output.mkdir(parents=True)
    evidence, outputs, captures, timings = {}, {}, {}, {}
    started, clock = now(), time.perf_counter()
    code = source_code()

    def record(path):
        outputs[path.relative_to(output).as_posix()] = dict(
            path=str(path), sha256=sha(path), size_bytes=path.stat().st_size
        )

    def write(name, value):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, value)
        record(path)

    def dump(name, value):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            np.savez_compressed(stream, **value)
        record(path)

    try:
        for path, digest in code.items():
            bind(path, digest, evidence)
            target = output / "code_snapshot" / Path(path).relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            require(sha(target) == digest, "code snapshot drift")
            record(target)
        t = time.perf_counter()
        inputs = metadata_inputs(evidence)
        write("retry.json", retry_metadata(evidence))
        prior.verify_payloads(inputs, evidence)
        timings["input_verification_seconds"] = time.perf_counter() - t
        write("original_seed.json", inputs["seed"])
        write("seed_contract.json", inputs["seed_contract"])
        write("original_context.json", inputs["metadata"]["context"])
        write("reference_origin.json", inputs["origin"])
        write("legacy_detector_condition.json", inputs["detector"])
        write(
            "plan.json",
            dict(
                schema=STAGE,
                plan_sha256=PLAN_SHA,
                native_interval=[START, STOP],
                frames=list(FRAMES),
                views=list(VIEWS),
                source_hashes=code,
                query_selector_sha256=SELECTOR_SHA,
                shared_SL_calibration_sha256=inputs["shared"],
                geometry_seal_sha256=GEOMETRY_SHA,
                prefix_preparation_seal_sha256=PREFIX_SHA,
                compact_seal_sha256=old.COMPACT_SHA,
                storage_audit_sha256=old.AUDIT_SHA,
                legacy_manifest_sha256=old.OLD_MANIFEST_SHA,
                raw_images=600,
                observations=600,
                query_slots=27,
                node_sets=9,
                query_upper_bound=27,
                source_storage="audited legacy FFV1 AVI, all600 canonical native BGR hashes",
                implementation_decision=(
                    "Original plan's no-AVI assumption superseded by Root: compact is MP4 "
                    "hardlink, no600PNG cache. Use audited lossless200 AVI; never fallback to MP4."
                ),
                original_MP4_opened=False,
                new_D2=False,
                biological_ID_truth=False,
                retrospective_conditional_labels=True,
                future_reference_poses_as_seeds=False,
                background_scope=(
                    "original offline median, including other recording times; reused unchanged"
                ),
            ),
        )
        source_path = GEOMETRY / f"known_gap/sources/native_{START}.npz"
        with np.load(source_path, allow_pickle=False) as z:
            source = {k: z[k].copy() for k in z.files}
        with np.load(GEOMETRY / "known_gap/reference_selected21.npz", allow_pickle=False) as z:
            selected = {k: z[k].copy() for k in z.files}
        queries, qualification, node_sets = initial_queries(
            source, selected, inputs["origin"], inputs["qualification"], inputs["node_sets"]
        )
        dump("initial_source_archive.npz", source)
        dump("initial_reference21.npz", {k: v[:1].copy() for k, v in selected.items()})
        write("initial_qualification.json", qualification)
        write("initial_qualified_node_sets.json", node_sets)
        write("query_slots.json", queries["slots"])
        write("node_sets.json", queries["node_sets"])
        query_rows = []
        for view in VIEWS:
            arrays = queries["views"][view]
            name = f"queries/{view}.npz"
            dump(name, arrays)
            query_rows.append(
                dict(
                    view=view,
                    frame_idx=START,
                    path=name,
                    sha256=outputs[name]["sha256"],
                    query_count=len(arrays["xy"]),
                    array_signatures={k: signature(v) for k, v in arrays.items()},
                    body_slot_labels="conditional original constructor slots; not physical ID",
                )
            )
        write("query_rows.json", query_rows)
        sl_paths = [Path(p) for p in inputs["shared"] if Path(p).suffix == ".npz"]
        t = time.perf_counter()
        nodes = CachedNodes(sl_paths[0].parents[3], native.EXPERIMENT, FRAMES)
        require(
            {str(p.resolve()) for p in nodes.paths.values()}
            == {str(p.resolve()) for p in sl_paths},
            "SL loader paths drift",
        )
        for view in VIEWS:
            require(
                signature(nodes.at(view, START)) == signature(source[view + "_points"]),
                "original SL/qualified raw pixels differ",
            )
        timings["SL_load_seconds"] = time.perf_counter() - t
        native_rows, backgrounds, codec_rows = [], [], []
        prefix = {(r["frame_idx"], r["view"]): r for r in inputs["prefix_rows"]}
        timings.update(
            mask_archive_load_seconds=0.0,
            payload_save_including_SHA_seconds=0.0,
            raw_decode_seconds=0.0,
            mask_selection_seconds=0.0,
        )
        for view in VIEWS:
            info = inputs["manifest"]["backgrounds"][view]
            bpath = old.OLD / info["path"]
            image = cv2.imread(str(bpath), cv2.IMREAD_COLOR)
            require(
                image is not None and image.dtype == np.uint8 and image.shape == (1200, 1200, 3),
                "original background shape/dtype drift",
            )
            target = output / f"backgrounds/{view}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(bpath, target)
            require(sha(target) == info["sha256"], "encoded background changed")
            record(target)
            backgrounds.append(
                dict(
                    view=view,
                    path=target.relative_to(output).as_posix(),
                    encoded_sha256=info["sha256"],
                    decoded_bgr_signature=signature(image),
                    source_frames=info["frames"],
                    future_background_information=True,
                )
            )
            t = time.perf_counter()
            mask_path = old.OLD / inputs["manifest"]["masks"][view]["path"]
            with np.load(mask_path, allow_pickle=False) as z:
                data = {k: z[k].copy() for k in z.files}
            require(
                set(data) == set(native.SCHEMA_DTYPES)
                and all(data[k].dtype == dt for k, dt in native.SCHEMA_DTYPES.items())
                and np.array_equal(data["frame_idx"], np.arange(START, STOP)),
                "packed13 archive differs",
            )
            data_signatures = {k: signature(v) for k, v in data.items()}
            timings["mask_archive_load_seconds"] += time.perf_counter() - t
            capture = cv2.VideoCapture(str(old.OLD / inputs["manifest"]["raw"][view]["path"]))
            captures[view] = capture
            codec_rows.append(capture_metadata(capture, view))
            write("video_transport_metadata.json", codec_rows)
            iterator = iter(read200(capture, inputs["manifest"]["raw"][view]["frame_bgr_sha256"]))
            for expected in FRAMES:
                t = time.perf_counter()
                frame, image = next(iterator)
                timings["raw_decode_seconds"] += time.perf_counter() - t
                require(frame == expected, "frame order drift")
                t = time.perf_counter()
                count = inputs["manifest"]["masks"][view]["instances_per_frame"][frame - START]
                obs = observation(data, frame, count, nodes.at(view, frame))
                timings["mask_selection_seconds"] += time.perf_counter() - t
                bgr_sha, mask_sha = signature(image)["sha256"], signature(obs["masks"])["sha256"]
                if (frame, view) in prefix:
                    row = prefix[frame, view]
                    require(
                        row["bgr_sha256"] == bgr_sha
                        and row["mask_sha256"] == mask_sha
                        and row["mask_available"] == bool(obs["mask_available"])
                        and row["mask_count"] == count,
                        "old64 exact native prefix differs",
                    )
                t = time.perf_counter()
                pngname = f"raw/raw_{frame}_{view}.png"
                png = output / pngname
                png.parent.mkdir(parents=True, exist_ok=True)
                require(cv2.imwrite(str(png), image), "PNG write failed")
                record(png)
                restored = cv2.imread(str(png), cv2.IMREAD_COLOR)
                require(
                    restored is not None and signature(restored) == signature(image),
                    "PNG roundtrip differs",
                )
                obsname = f"observations/observation_{frame}_{view}.npz"
                dump(obsname, obs)
                native_rows.append(
                    dict(
                        frame_idx=frame,
                        view=view,
                        shape_hw=[1200, 1200],
                        native_offset=frame - START,
                        png_path=pngname,
                        png_sha256=outputs[pngname]["sha256"],
                        observation_path=obsname,
                        observation_sha256=outputs[obsname]["sha256"],
                        bgr_sha256=bgr_sha,
                        bgr_signature=signature(image),
                        mask_sha256=mask_sha,
                        mask_count=count,
                        mask_available=bool(obs["mask_available"]),
                        detections_sha256=signature(obs["detections"])["sha256"],
                        observation_array_signatures={k: signature(v) for k, v in obs.items()},
                        manifest_sha256=old.OLD_MANIFEST_SHA,
                    )
                )
                timings["payload_save_including_SHA_seconds"] += time.perf_counter() - t
            require(next(iterator, None) is None, "unexpected extra planned frame")
            require(
                data_signatures == {k: signature(v) for k, v in data.items()},
                "packed13 inputs changed during view preparation",
            )
            capture.release()
            del captures[view]
            print(
                json.dumps(
                    dict(
                        event="semantic200_view_prepared",
                        view=view,
                        frames=200,
                        queries=len(queries["views"][view]["xy"]),
                    ),
                    allow_nan=False,
                ),
                flush=True,
            )
        native_rows.sort(key=lambda r: (r["frame_idx"], VIEWS.index(r["view"])))
        require(
            [(r["frame_idx"], r["view"]) for r in native_rows]
            == [(n, v) for n in FRAMES for v in VIEWS],
            "600 native/view membership drift",
        )
        write("native_evidence.json", native_rows)
        write("backgrounds.json", backgrounds)
        prior.no_model_imports()
        require(code == source_code(), "loaded source set changed")
        t = time.perf_counter()
        for path, digest in evidence.items():
            require(sha(path) == digest, "input changed: " + path)
        timings["final_input_guard_seconds"] = time.perf_counter() - t
        timings["total_before_summary_seconds"] = time.perf_counter() - clock
        summary = dict(
            schema=STAGE,
            status="ok",
            native_interval=[START, STOP],
            frames=list(FRAMES),
            views=list(VIEWS),
            native_frames=200,
            raw_images=600,
            decoded_mask_stacks=600,
            total_masks=sum(r["mask_count"] for r in native_rows),
            query_slots=27,
            node_sets=9,
            queries_per_view={r["view"]: r["query_count"] for r in query_rows},
            queries=sum(r["query_count"] for r in query_rows),
            prefix192_BGR_mask_exact=True,
            original_MP4_opened=False,
            legacy_FFV1_used=True,
            transport_metadata=codec_rows,
            retry_of_failed_v1_sha256=FAILED_SHA,
            model_loaded=False,
            GPU=False,
            model_forward_calls=0,
            fit_calls=0,
            advance_calls=0,
            initialize_calls=0,
            full_candidate_payloads_read=False,
            selected64_reference21_read=True,
            retrospective_conditional_labels=True,
            biological_ID_truth=False,
            future_reference_poses_used_as_seeds=False,
            timings=timings,
        )
        write("summary.json", summary)
        prior.output_guard(output, outputs)
        write_json(
            output / "preparation.provenance.json",
            dict(
                stage=STAGE,
                status="ok",
                started_at=started,
                finished_at=now(),
                inputs=evidence,
                outputs=outputs,
                environment=dict(
                    python=sys.version,
                    numpy=np.__version__,
                    opencv=cv2.__version__,
                    opencv_threads=cv2.getNumThreads(),
                    model_imported=False,
                ),
            ),
        )
        return summary
    except BaseException as exc:
        write_json(
            output / "preparation.provenance.json",
            dict(
                stage=STAGE,
                status="failed",
                started_at=started,
                finished_at=now(),
                inputs=evidence,
                outputs=outputs,
                error=dict(type=type(exc).__name__, message=str(exc)),
            ),
        )
        raise
    finally:
        for capture in captures.values():
            capture.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
