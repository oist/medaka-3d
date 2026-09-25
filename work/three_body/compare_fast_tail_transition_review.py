"""Disabled CPU comparison of a real imported nine-mouth/six-link human review.

Three final116200 candidates and their actual beam14 mappings are read only.
Human spatial correspondence includes finite latent competitors; saved measured
support, existing reviews and conditional beam ambiguity are separate evidence.
No SL, images, masks, checkpoint39, graph, solver or GPU execution is needed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
from pathlib import Path

import import_long_review as human
import numpy as np
import score_variant_female_five_minute_anchors as variants

from medaka3d import calib
from medaka3d.eval import encounter_review
from medaka3d.types import VIEWS

prior, depth = variants.prior, variants.depth
old, comparison, proof = prior.control, variants.comparison, variants.proof
require = proof.require
REPO, WORK = old.REPO, old.WORK
FIRST, START, STOP = 49800, 66400, 116200
CLIP = [90040, 90400]
STAGE = "fast_tail_transition_human_comparison_v1"
EXECUTION_ENABLED = False
IMPORT_STAGE = "fast_tail_transition_review_document_import_v1"
IMPORT_SOURCE = WORK / "import_fast_tail_transition_review.py"
IMPORT_FALSE_SHA = "de2bea445002986fb646653fc4363d1d9f37c97bb43f4b1b37efc4093080f45b"
IMPORT_TRUE_SHA = "54135664aa73a2da2aa6b04486dc6619c501d048041374f77fe4ee9183e5ab75"
VARIANTS_FALSE_SHA = "59c989a179576049796d495038044c962e6c4b83c79b0ecd6d5def0823ac072c"
PACK_ROOT = WORK / "fast_tail_transition_raw_review_pack_v1"
PACK_SHA = "8622bd5655ecc34058c5e3a7a2fe64e7ff357a3d9eb508cfca4e258405fb0c7f"
PACK_STAGE = "fast_tail_transition_raw_review_pack_v1"
FIELDS = prior.FIELDS
MAPPING_FIELDS = variants.MAPPING_FIELDS
ARMS = ("fast_tail", *comparison.ARMS)
# Explicit semantic/helper pins, not a dynamic import-closure inventory.
PINS = {
    "work/three_body/import_long_review.py": (
        "4ee9a2af648231658714c017ff3d48232700f09a701092f7a70a75869421bcc5"
    ),
    "work/three_body/prepare_long_raw_review.py": (
        "c1a799d248f9b0f3b0e90329fb034e73af1a6a2c74fab9270dd8df84d07c66f0"
    ),
    "labs/20260909_three_body/evaluate_review.py": (
        "212ff70e92675f7880fd9ca4f02cf31815282b064a62315c31eaeb6ef0e66190"
    ),
    "labs/20260909_three_body/review_truth.py": (
        "9dd5fe7b96d4a8d84954604a2e647be6bfa2d9652ad78378c51618c5fa70dbd9"
    ),
    "src/medaka3d/eval/encounter_review.py": (
        "49538a9d76180642fce84b4ad90e7c7675ee759aa8f00f573b2e6a60c714d2f9"
    ),
    "src/medaka3d/calib.py": "60849004cb53febe69f437e7afa8b18cb2b65213ba76ce7c921f7022d2be4274",
    "src/medaka3d/types.py": "71b002e8dbfe7efb139520bc387c9cc0aa097ce55efb24aadd1a6080da71e33a",
    "work/three_body/score_female_five_minute_anchors.py": (
        "b50e7668b91c6474edfe9d5d2150deeda8a823c11c7a08d5c451e6709f616165"
    ),
    "work/three_body/run_fast_tail_five_minute_identity_beam.py": (
        "9bf62d84cf2eff9cb2dc19ed2dd6beab901633b8221ca6491ab5f0b2bc3a55e1"
    ),
    "work/three_body/run_sparse_instance_five_minute_identity_beam.py": (
        "d433ce526804f5d72594b942c6ab93e9d8e6a2b9b6c1ae46e3f276b6cc925631"
    ),
    "work/three_body/compare_fast_tail_continuation.py": (
        "2e4a749d5ea13116f05a622afd47478af311525fbd856b32a116bd9ac765acba"
    ),
    "work/three_body/continue_fast_tail_five_minutes.py": (
        "d90e9eb49038dac4fd1e18068334ba89eb8df1a813a37c6aec071c20db1a61a6"
    ),
    "work/three_body/compare_expanded_depth_five_minutes.py": (
        "db30f8fd32a58a60b77ef1daa50dbf42da7a0d181eb4b43da47cdf2b2ee2694e"
    ),
    "work/three_body/compare_sparse_instance_five_minutes.py": (
        "1ec0893d7dd2bbc746004f71d9b81abc60096cc0173bf44871f56be948e1c8dc"
    ),
    "work/three_body/compare_instance_rescore_10s.py": (
        "bb7653e5f7d71f2b4c078418dee93433fb4a1fc3fdefd7f38ef9c4694ae9040a"
    ),
    "work/three_body/compare_temporal_point64_v2.py": (
        "2461f383d7c69223c98cb9371058373361a7fb30a409a37d617b60fc2db53790"
    ),
    "work/three_body/temporal_point_core.py": (
        "3ec308ca7877338f8677c0703d6f180a9d5658c4858b9868451e1152b7d4e25b"
    ),
}


def flag_source_digest(data, expected_false_sha):
    """Accept only the reviewed executable flag change; record the actual bytes."""
    old_flag, new_flag = b"EXECUTION_READY = False", b"EXECUTION_READY = True"
    lines = data.splitlines(keepends=True)
    matches = [i for i, line in enumerate(lines) if line.rstrip(b"\r\n") in (old_flag, new_flag)]
    require(len(matches) == 1, "one execution flag required")
    index = matches[0]
    # A helper's quoted flag examples are source, not executable assignments.
    # Preserve them, all other bytes, and the original LF/CRLF convention.
    lines[index] = lines[index].replace(new_flag, old_flag)
    require(
        hashlib.sha256(b"".join(lines)).hexdigest() == expected_false_sha,
        "source changed beyond declared execution flag",
    )
    return hashlib.sha256(data).hexdigest()


def source_code():
    code = {}
    for name, expected in PINS.items():
        path = REPO / name
        require(proof.sha(path) == expected, "frozen helper differs: " + name)
        code[str(path)] = expected
    for path, expected in (
        (IMPORT_SOURCE, IMPORT_FALSE_SHA),
        (Path(depth.__file__).resolve(), variants.DEPTH_REVIEWED_FALSE_SHA),
        (Path(variants.__file__).resolve(), VARIANTS_FALSE_SHA),
    ):
        data = path.read_bytes()
        code[str(path)] = flag_source_digest(data, expected)
    code[str(Path(__file__).resolve())] = proof.sha(__file__)
    return code


def workspace(path):
    path = Path(path).resolve()
    require(path.is_relative_to(WORK.resolve()), "input/output outside work directory")
    return path


def member(root, seal, name, evidence, *, binary=False):
    require(not Path(name).is_absolute() and ".." not in Path(name).parts, "unsafe member")
    root, path = workspace(root), workspace(Path(root) / name)
    row = seal["outputs"][name]
    require(
        path.is_relative_to(root)
        and Path(row["path"]).resolve() == path
        and type(row["size_bytes"]) is int
        and path.stat().st_size == row["size_bytes"],
        "sealed member path/size differs",
    )
    evidence.add(path, row["sha256"])
    return path.read_bytes() if binary else human.document(path.read_bytes(), name)


def seal_at(root, filename, digest, stage, evidence):
    path = workspace(root) / filename
    require(proof.valid_sha(digest), "explicit successful seal SHA required")
    evidence.add(path, digest)
    seal = human.document(path.read_bytes(), filename)
    require(seal.get("stage") == stage and seal.get("status") == "ok", "successful stage required")
    return seal


def checked_review(parsed):
    checked = human.parse_review(parsed.review_bytes, parsed.manifest_bytes, clip_range=CLIP)
    require(
        parsed.review == checked.review
        and parsed.manifest == checked.manifest
        and np.array_equal(parsed.anchor_frames, checked.anchor_frames)
        and np.array_equal(parsed.human_points, checked.human_points, equal_nan=True)
        and np.array_equal(parsed.continuity, checked.continuity),
        "parsed fields differ from preserved human bytes",
    )
    return checked


def validate_import_summary(parsed, summary, proposal):
    """Actual new importer schema; never treat its staged links as scored ones."""
    parsed = checked_review(parsed)
    endpoints, links, changes = [], [], []
    for t, anchor in enumerate(parsed.review["anchors"]):
        proposed = proposal["proposed_anchor_frames"][t]
        changes.append(
            dict(
                phase=anchor["phase"],
                proposed_frame=proposed,
                returned_frame=anchor["frame"],
                changed=anchor["frame"] != proposed,
                returned_frame_choice=anchor["frame_choice"],
            )
        )
        for animal in human.ANIMALS:
            mark = anchor["animals"][animal]
            endpoints.append(
                dict(
                    phase=anchor["phase"],
                    frame=anchor["frame"],
                    anonymous_animal=animal,
                    visibility=mark["visibility"],
                    original_points=mark["points"],
                    model_slot=None,
                    physical_identity_certified=False,
                )
            )
    for animal in human.ANIMALS:
        value = parsed.review["continuity"][animal]
        for i, key in enumerate(human.EDGES):
            links.append(
                dict(
                    anonymous_animal=animal,
                    edge=key,
                    start_frame=int(parsed.anchor_frames[i]),
                    stop_frame=int(parsed.anchor_frames[i + 1]),
                    human_status=value[key],
                    human_basis=value["basis"],
                    human_notes=value.get("notes", ""),
                    trajectory_constraint_created=False,
                )
            )
    expected = dict(
        schema=IMPORT_STAGE,
        status="ok",
        review_status_preserved=parsed.review["status"],
        review_clip_range=CLIP,
        source_frame_range=[89640, 91300],
        pack_seal_sha256=PACK_SHA,
        parser_source_sha256=PINS["work/three_body/import_long_review.py"],
        original_review_sha256=hashlib.sha256(parsed.review_bytes).hexdigest(),
        external_input_count=1,
        original_review_bytes_preserved=True,
        endpoints=endpoints,
        links=links,
        proposed_to_returned_anchor_changes=changes,
        original_anchor_proposal=proposal,
        biological_naming_preserved_not_applied=parsed.review.get("biological_naming"),
        anonymous_identity_scope="within this clip only; A/B/C are human entry choices",
        pack_outputs_verified=1096,
        original_video_stat_or_payload_reads=0,
        SLEAP_mask_candidate_checkpoint_reads=0,
        image_decodes=0,
        GPU_used=False,
        identity_solver_calls=0,
        trajectory_modified=False,
        constraints_created=False,
        labels_filled=False,
        physical_identity_certified=False,
        all_five_minute_identity_truth=False,
        human_truth_created=False,
    )
    comparison.common.exact_fields(summary, expected, "imported review schema/content differs")
    require(len(endpoints) == 9 and len(links) == 6, "nine endpoints/six links required")


def load_imported_review(root, digest, evidence):
    seal = seal_at(root, "import.provenance.json", digest, IMPORT_STAGE, evidence)
    depth.edge(seal, IMPORT_SOURCE, IMPORT_TRUE_SHA)
    evidence.add(IMPORT_SOURCE, IMPORT_TRUE_SHA)
    require(
        set(seal["outputs"])
        == {
            "original_review.json",
            "source_manifest.json",
            "original_anchor_selection.json",
            "summary.json",
        },
        "new importer output schema differs",
    )
    raw = member(root, seal, "original_review.json", evidence, binary=True)
    manifest_bytes = member(root, seal, "source_manifest.json", evidence, binary=True)
    parsed = human.parse_review(raw, manifest_bytes, clip_range=CLIP)
    summary = member(root, seal, "summary.json", evidence)
    proposal = member(root, seal, "original_anchor_selection.json", evidence)
    validate_import_summary(parsed, summary, proposal)
    external = seal["explicit_review_inputs"]
    require(
        len(external) == 1 and list(external.values()) == [hashlib.sha256(raw).hexdigest()],
        "one preserved explicit review input required",
    )
    depth.edge(seal, PACK_ROOT / "review.provenance.json", PACK_SHA)
    pack = seal_at(PACK_ROOT, "review.provenance.json", PACK_SHA, PACK_STAGE, evidence)
    pack_summary = member(PACK_ROOT, pack, "summary.json", evidence)
    require(
        pack["outputs"]["source_manifest.json"]["sha256"]
        == hashlib.sha256(manifest_bytes).hexdigest()
        == pack_summary["source_manifest_sha256"]
        and pack_summary["review_clip_range"] == CLIP
        and pack_summary["consumed_record"]["index"] == 24,
        "review/source pack binding differs",
    )
    return dict(
        parsed=parsed,
        summary=summary,
        pack_summary=pack_summary,
        import_seal_sha256=digest,
        original_import_root=str(workspace(root)),
    )


def old_identity_headers(reference, evidence):
    root = prior.BEAM_ROOT
    seal = seal_at(root, "identity.provenance.json", prior.BEAM_SEAL_SHA, old.STAGE, evidence)
    old.edge(seal, Path(old.__file__), prior.CONTROL_SHA)
    old.edge(seal, old.COMPARISON_ROOT / "comparison.provenance.json", old.COMPARISON_SEAL_SHA)
    old.edge(seal, Path(reference["path"]), reference["sha256"])
    plan, summary, scalars = [
        member(root, seal, name, evidence)
        for name in ("plan.json", "summary.json", "beam_scalars.json")
    ]
    for doc in (plan, summary):
        comparison.common.exact_fields(
            doc,
            dict(
                schema=old.STAGE,
                native_interval=[START, STOP],
                native_frames=STOP - START,
                comparison_candidate_reference=reference,
                raw21_signatures=reference["arrays"],
                comparison_seal_sha256=old.COMPARISON_SEAL_SHA,
                milestone_summary_sha256=old.MILESTONE_SHA,
                beam_width=256,
                gaps_conditional_on_retained_graph=True,
                gaps_are_probabilities=False,
                physical_ID_truth=False,
                global_optimality_certified=False,
            ),
            "OldFast beam metadata differs",
        )
    require(
        plan["initial_P"] == [0, 1, 2]
        and plan["dp_config"] == old.CONFIG
        and plan["node_indices"] == [1, 0]
        and plan["source_sha256"].get(str(Path(old.__file__).resolve())) == prior.CONTROL_SHA,
        "OldFast beam source/objective differs",
    )
    comparison.common.exact_fields(
        summary,
        dict(
            status="ok",
            terminal_horizon=STOP,
            solve_calls=1,
            intermediate_ID_resets=0,
            original_raw21_unchanged=True,
            raw_sources_review_unchanged=True,
            review_flags_cleared=False,
        ),
        "OldFast beam preservation differs",
    )
    return dict(
        root=root,
        seal=seal,
        scalars=scalars,
        summary=summary,
        seal_sha256=prior.BEAM_SEAL_SHA,
        source_sha256=prior.CONTROL_SHA,
    )


def load_headers(common_sha, sparse_id_sha, depth_id_sha, evidence):
    """Real successful stage contracts only; all gates precede payload decoding."""
    old_binding = old.comparison_headers(old.COMPARISON_SEAL_SHA, evidence)
    old_ref = old_binding["comparison_candidate_reference"]
    common_seal, _ = depth.comparison_headers(common_sha, evidence)
    plan = member(depth.COMPARISON_ROOT, common_seal, "plan.json", evidence)
    references = {"fast_tail": old_ref, **plan["candidate_references"]}
    bundles = {"fast_tail": old_identity_headers(old_ref, evidence)}
    sparse_root = WORK / depth.frozen.STAGE
    sparse_seal = seal_at(
        sparse_root, "identity.provenance.json", sparse_id_sha, depth.frozen.STAGE, evidence
    )
    sparse_summary = member(sparse_root, sparse_seal, "summary.json", evidence)
    for arm, root, sha, source, source_sha, body_root, body_sha, body_source in (
        (
            ARMS[1],
            sparse_root,
            sparse_id_sha,
            variants.SPARSE_SOURCE,
            variants.SPARSE_EXECUTION_SHA,
            depth.frozen.COMPARISON_ROOT,
            sparse_summary["comparison_seal_sha256"],
            depth.frozen.COMPARISON_SHA,
        ),
        (
            ARMS[2],
            WORK / depth.STAGE,
            depth_id_sha,
            Path(depth.__file__),
            variants.DEPTH_EXECUTION_SHA,
            depth.COMPARISON_ROOT,
            common_sha,
            variants.COMPARISON_SHA,
        ),
    ):
        bundles[arm] = variants.identity_headers(
            root,
            sha,
            source,
            source_sha,
            arm,
            references[arm],
            body_root,
            body_sha,
            body_source,
            evidence,
        )
    descriptors = {"fast_tail": dict(old_binding["candidate_archive_references"]["milestone"])}
    for arm in ARMS[1:]:
        descriptors[arm] = copy.deepcopy(common_seal["outputs"][f"candidates/{arm}.npz"])
        require(
            Path(descriptors[arm]["path"]).resolve()
            == (depth.COMPARISON_ROOT / f"candidates/{arm}.npz").resolve(),
            "comparison copy path differs",
        )
    return dict(
        references=references,
        descriptors=descriptors,
        bundles=bundles,
        common_seal=common_seal,
        common_sha=common_sha,
    )


def validate_lineage(imported, indices, producers):
    """Small consumed metadata only. Canonical videos are never stat'ed here."""
    require(set(indices) == set(producers) == set(ARMS), "three final producers required")
    first = indices[ARMS[0]]
    require(all(index == first for index in indices.values()), "different native input index")
    require(set(first) == {"original_binding", "prefix_chunks"}, "actual input_index schema")
    pack = imported["pack_summary"]
    chunks = [row for row in first["prefix_chunks"] if row["index"] == 24]
    require(len(chunks) == 1 and chunks[0] == pack["consumed_record"], "review chunk24 differs")
    chunk, manifest = chunks[0], imported["parsed"].manifest
    require(
        chunk["start"] == 89640
        and chunk["stop"] == 91300
        and chunk["manifest_sha256"] == pack["source_manifest_sha256"]
        and manifest["raw_storage"] == "original_source_video",
        "source native range/storage differs",
    )
    shared = first["original_binding"]["inputs"]["seed"]["SL_and_calibration_sha256"]
    require(len(shared) == 7, "registered SL3/calibration4 required")
    videos = imported["summary"]["source_video_bindings"]
    require(set(videos) == set(VIEWS), "three review source videos required")
    for producer in producers.values():
        depth.edge(producer, Path(pack["source_manifest"]), pack["source_manifest_sha256"])
        for path, digest in shared.items():
            depth.edge(producer, Path(path), digest)  # SHA edges only; SL is not opened.
        for view in VIEWS:
            video, raw, hardlink = videos[view], manifest["raw"][view], chunk["raw_hardlinks"][view]
            require(
                video["sha256"] == raw["sha256"] == hardlink["sha256"]
                and os.path.abspath(video["canonical_path"])
                == os.path.abspath(hardlink["source_path"])
                and os.path.abspath(video["staged_path"]) == os.path.abspath(hardlink["path"])
                and raw["source_offset_frames"] == 89640
                and raw["source_total_frames"] == 348600,
                "review canonical/staged/native source differs",
            )
            # Deliberately lexical: do not resolve/stat/hash the video itself.
            canonical = os.path.abspath(video["canonical_path"])
            require(producer["inputs"].get(canonical) == video["sha256"], "canonical source edge")
    return shared


def load_rig_and_lineage(imported, headers, evidence):
    indices, producers = {}, {}
    for arm in ARMS:
        if arm == "fast_tail":
            root, stage, sha = old.TRACKING_ROOT, old.comparison.runner.STAGE, old.TRACKING_SEAL_SHA
        else:
            ref = headers["references"][arm]
            root = Path(ref["path"]).parent
            stage, sha = root.name, ref["tracking_seal_sha256"]
        producers[arm] = seal_at(root, "tracking.provenance.json", sha, stage, evidence)
        indices[arm] = member(root, producers[arm], "input_index.json", evidence)
    shared = validate_lineage(imported, indices, producers)
    calibration = {p: sha for p, sha in shared.items() if Path(p).suffix.lower() == ".json"}
    manifests = [Path(p) for p in calibration if Path(p).name == "calibration_manifest.json"]
    require(len(calibration) == 4 and len(manifests) == 1, "four registered calibration JSONs")
    root = manifests[0].parent
    for path, sha in calibration.items():
        require(Path(path).resolve().is_relative_to(REPO), "calibration outside repository")
        evidence.add(Path(path), sha)
    manifest = human.document(manifests[0].read_bytes(), "calibration manifest")
    model_paths = []
    for row in manifest["model_files"]:
        require(
            not Path(row["path"]).is_absolute() and ".." not in Path(row["path"]).parts,
            "calibration model path escape",
        )
        path = (root / row["path"]).resolve()
        require(calibration.get(str(path)) == row["sha256"], "calibration model binding differs")
        model_paths.append(str(path))
    require(set(calibration) == {str(manifests[0]), *model_paths}, "unbound calibration read")
    return calib.load_rig(root), dict(
        shared_SL_calibration_sha256=shared,
        calibration_sha256=calibration,
        source_video_bindings=imported["summary"]["source_video_bindings"],
        source_chunk_index=24,
        input_indices_equal=True,
        SL_payload_revalidated_here=False,
        strict_support_inherited_from_successful_comparisons=dict(ray_mm=2.0, native_pixel=12.0),
    )


def selected_indices(frames, *, first=START, stop=STOP):
    frames = np.asarray(frames)
    require(
        frames.shape == (3,)
        and frames.dtype.kind in "iu"
        and np.all(np.diff(frames) > 0)
        and np.all((frames >= first) & (frames < stop)),
        "three ordered native anchors required; no shifting or extrapolation",
    )
    return frames - first


def validate_rows(rows, frames):
    require(
        set(rows) == set(FIELDS) and np.array_equal(rows["frame_idx"], frames),
        "exact five selected anchor fields required",
    )
    selected_indices(rows["frame_idx"])
    for key, shape, kind in (
        ("pose_xyz", (3, 3, 3, 3), "f"),
        ("sources", (3, 3, 3, 3), "iu"),
        ("observed", (3, 3, 3), "b"),
        ("identity_review", (3, 3), "b"),
    ):
        value = rows[key]
        require(
            isinstance(value, np.ndarray) and value.shape == shape and value.dtype.kind in kind,
            "anchor field shape/type: " + key,
        )
    require(
        not np.isinf(rows["pose_xyz"]).any() and np.all(rows["sources"] >= -1),
        "invalid pose/source values",
    )
    require(
        not np.any(rows["observed"] & ~np.isfinite(rows["pose_xyz"]).all(axis=-1))
        and not np.any(rows["observed"] & ((rows["sources"] >= 0).sum(axis=2) < 2)),
        "saved observed node contradicts finite/source support",
    )


def load_anchor_rows(descriptor, signatures, frames, evidence):
    indices = selected_indices(frames, first=FIRST)
    path = workspace(descriptor["path"])
    require(path.stat().st_size == descriptor["size_bytes"], "candidate archive size differs")
    evidence.add(path, descriptor["sha256"])
    rows = {}
    with np.load(path, allow_pickle=False) as archive:
        require(
            len(archive.files) == len(set(archive.files)) == 21
            and set(archive.files) == set(signatures),
            "candidate21 archive membership",
        )
        for key in FIELDS:
            value = archive[key]
            require(proof.signature(value) == signatures[key], "candidate field bytes: " + key)
            if key == "frame_idx":
                require(
                    np.array_equal(value, np.arange(FIRST, STOP)), "retained candidate frame gap"
                )
            rows[key] = value[indices].copy()
    validate_rows(rows, frames)
    for value in rows.values():
        value.flags.writeable = False
    return rows


def load_mapping(bundle, frames, evidence):
    indices = selected_indices(frames)
    root, seal = bundle["root"], bundle["seal"]
    name = "identity_mapping_and_ambiguity.npz"
    descriptor = seal["outputs"][name]
    path = workspace(root / name)
    require(
        Path(descriptor["path"]).resolve() == path
        and path.stat().st_size == descriptor["size_bytes"],
        "mapping output descriptor",
    )
    evidence.add(path, descriptor["sha256"])
    with np.load(path, allow_pickle=False) as archive:
        require(
            len(archive.files) == len(set(archive.files)) == 14
            and set(archive.files) == set(depth.frozen.MAPPING_FIELDS),
            "actual beam14 schema",
        )
        arrays = {k: archive[k] for k in MAPPING_FIELDS}
    prior.validate_mapping(arrays, tolerance=bundle["scalars"]["numerical_tie_tolerance"])
    # The four terminal members do not have a frame axis; they are not decoded/sliced.
    result = {k: value[indices].copy() for k, value in arrays.items()}
    for value in result.values():
        value.flags.writeable = False
    return result


def local_links(spatial, continuity):
    """Local human assertions survive an unconfirmed chain to entry."""
    rows, counts = [], dict(same=0, different=0, unscored=0)
    for animal, name in enumerate(human.ANIMALS):
        for edge, key in enumerate(human.EDGES):
            left, right = spatial["anchors"][edge : edge + 2]
            a, b = left["animals"][animal]["slot"], right["animals"][animal]["slot"]
            state = str(continuity[animal, edge])
            reason = (
                "human continuity " + state
                if state != "confirmed"
                else (
                    "endpoint correspondence unavailable"
                    if a is None or b is None
                    else "matched local endpoints"
                )
            )
            outcome = (
                "unscored"
                if reason != "matched local endpoints"
                else ("same" if a == b else "different")
            )
            counts[outcome] += 1
            rows.append(
                dict(
                    anonymous_animal=name,
                    edge=key,
                    start_frame=left["frame"],
                    stop_frame=right["frame"],
                    human_status=state,
                    left_label=a,
                    right_label=b,
                    outcome=outcome,
                    reason=reason,
                )
            )
    return dict(
        requested_links=6,
        counts=counts,
        links=rows,
        entry_identity_inferred=False,
        interior_identity_certified=False,
        physical_swap_events_counted=False,
    )


def score_rows(parsed, rows, rig):
    parsed = checked_review(parsed)
    validate_rows(rows, parsed.anchor_frames)
    before = {k: proof.signature(v) for k, v in rows.items()}
    projected = human.evaluate_review.project_mouth(rig, rows, parsed.anchor_frames)
    args = (rows["frame_idx"], projected)
    tail = (parsed.anchor_frames, parsed.human_points, parsed.continuity)
    spatial = encounter_review.score_encounter_review(
        *args,
        np.ones((3, 3), bool),
        np.zeros((3, 3), bool),
        *tail,
    )
    supported = encounter_review.score_encounter_review(
        *args,
        rows["observed"][:, :, 0],
        rows["identity_review"],
        *tail,
    )
    result = dict(
        human_spatial=spatial["raw"],
        saved_observed=supported["raw"],
        saved_observed_and_review=supported["accepted"],
        local_human_endpoint_links=local_links(spatial["raw"], parsed.continuity),
        saved_node_observed=rows["observed"].tolist(),
        saved_source_rows=rows["sources"].tolist(),
        saved_identity_review=rows["identity_review"].tolist(),
        all_latent_mouth_projections=proof.safe_json(projected.tolist()),
        marked_view_minimum=1,
        match_radius_px=5.0,
        all_marked_views_must_match_same_slot=True,
        unique_across_all_maximum_cardinality_matchings=True,
        strict_support_recomputed_here=False,
        human_edge_states=spatial["human_edge_states"],
        scope=spatial["scope"],
    )
    for key in ("human_spatial", "saved_observed", "saved_observed_and_review"):
        score = result[key]
        require(
            score["requested_anchors"] == 9
            and score["requested_animal_intervals"] == 6
            and sum(score["interval_counts"].values()) == 6,
            "nine/six denominator changed",
        )
    require(before == {k: proof.signature(v) for k, v in rows.items()}, "raw anchor fields changed")
    return result


def evaluate_arm(parsed, rows, rig, mapping):
    require(
        set(mapping) == set(MAPPING_FIELDS)
        and np.array_equal(mapping["frame_idx"], parsed.anchor_frames),
        "mapping anchor rows differ",
    )
    before = {k: proof.signature(v) for k, v in mapping.items()}
    reordered = prior.reorder_rows(rows, mapping["permutation_by_id"])
    raw, automatic = score_rows(parsed, rows, rig), score_rows(parsed, reordered, rig)
    require(before == {k: proof.signature(v) for k, v in mapping.items()}, "mapping mutated")
    return dict(
        raw_slots=raw,
        automatic_initial_slot_IDs=automatic,
        raw_slot_relation="raw_slot = permutation_by_id[anchor_index, automatic_ID]",
        permutation_by_id=mapping["permutation_by_id"].tolist(),
        conditional_ID_alternative_status=mapping[
            "conditional_identity_alternative_status"
        ].tolist(),
        conditional_ID_alternative_gap=[
            ["+Infinity" if np.isposinf(v) else float(v) for v in row]
            for row in mapping["conditional_identity_alternative_gap"]
        ],
        conditional_gaps_are_probabilities=False,
        conditional_gaps_did_not_filter_human_spatial=True,
        global_optimality_certified=False,
        biological_naming_applied=False,
        raw_review_flags_cleared=False,
        tracks_modified=False,
    )


def run(args):
    require(EXECUTION_ENABLED, "source-only: human comparison execution is disabled")
    output = workspace(args.output)
    require(not output.exists(), "fresh output required")
    old.cpu_only()
    evidence, out, started = proof.Evidence(), proof.Output(output), proof.utc()
    try:
        code = source_code()
        for path, sha in code.items():
            evidence.add(path, sha)
        imported = load_imported_review(args.import_root, args.import_sha, evidence)
        parsed = imported["parsed"]
        headers = load_headers(args.comparison_sha, args.sparse_id_sha, args.depth_id_sha, evidence)
        rig, lineage = load_rig_and_lineage(imported, headers, evidence)
        results = {}
        for arm in ARMS:
            rows = load_anchor_rows(
                headers["descriptors"][arm],
                headers["references"][arm]["arrays"],
                parsed.anchor_frames,
                evidence,
            )
            mapping = load_mapping(headers["bundles"][arm], parsed.anchor_frames, evidence)
            results[arm] = evaluate_arm(parsed, rows, rig, mapping)
            out.write(f"anchors/{arm}.npz", rows)
            out.write(f"identity_anchors/{arm}.npz", mapping)
        out.write("original_review.json", parsed.review_bytes)
        out.write("source_manifest.json", parsed.manifest_bytes)
        out.write("import_summary.json", imported["summary"])
        out.write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                native_interval=[START, STOP],
                terminal_horizon=STOP,
                review_clip_range=CLIP,
                anchor_frames=parsed.anchor_frames.tolist(),
                requested_endpoints_per_arm=9,
                requested_links_per_arm=6,
                results=results,
                import_root=imported["original_import_root"],
                import_seal_sha256=args.import_sha,
                original_review_sha256=hashlib.sha256(parsed.review_bytes).hexdigest(),
                review_status_preserved=parsed.review["status"],
                biological_naming_preserved_not_applied=parsed.review.get("biological_naming"),
                comparison_seals=dict(
                    old=old.COMPARISON_SEAL_SHA, sparse_depth=args.comparison_sha
                ),
                candidate_references=headers["references"],
                consumed_candidate_archives=headers["descriptors"],
                identity_bindings={
                    arm: dict(seal_sha256=b["seal_sha256"], source_sha256=b["source_sha256"])
                    for arm, b in headers["bundles"].items()
                },
                lineage=lineage,
                source_sha256=code,
                source_inventory_scope="explicit used helpers",
                candidate_archives_decoded=3,
                candidate_fields_decoded=5,
                candidate_member_count=21,
                unread_candidate_fields_numeric_verification_inherited=True,
                mapping_archives_decoded=3,
                mapping_fields_decoded=5,
                mapping_member_count=14,
                terminal_mapping_arrays_decoded=False,
                strict_support_recomputed_here=False,
                SL_payload_reads=0,
                image_video_mask_reads=0,
                checkpoint39_reads=0,
                identity_graph_reads=0,
                identity_solves=0,
                GPU_used=False,
                trajectory_modified=False,
                reviews_cleared=False,
                human_labels_created=False,
                biological_identity_assigned=False,
                physical_swap_event_count_certified=False,
                all_five_minute_identity_truth=False,
                dense_swap_zero_certified=False,
                automatic_adoption=False,
                endpoint_only_diagnostics_are_not_interior_certificates=True,
            ),
        )
        old.cpu_only()
        require(source_code() == code, "used source changed")
        evidence.final_guard()
        out.final_guard()
        status = dict(status="ok")
    except BaseException as error:
        status = dict(status="failed", error_type=type(error).__name__, error=str(error))
        out.write("failure.json", status)
        raise
    finally:
        with (output / "comparison.provenance.json").open("xb") as stream:
            stream.write(
                proof.json_bytes(
                    dict(
                        stage=STAGE,
                        started_at=started,
                        finished_at=proof.utc(),
                        inputs=evidence.paths,
                        outputs=out.records,
                        **status,
                    )
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("import-root", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    for option in ("import-sha", "comparison-sha", "sparse-id-sha", "depth-id-sha"):
        parser.add_argument("--" + option, required=True)
    run(parser.parse_args())
