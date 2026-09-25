"""Fixed three-bank CPU ablation of the complete-SLEAP-group eligibility gate.

The saved candidate bank and its parent path costs stay fixed. One camera's
sixth-term contribution changes; this produces diagnostic rankings, not a new
temporal trajectory. The reflection case is retained regardless of its result.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import sys
from pathlib import Path

import instance_body_score as core
import mask_only_instance_observation as preparer
import numpy as np
import qualify_sparse_instance_saved_10s as parity
import run_instance_rescore_10s as codec
import score_saved_capacity_island as source
import sparse_instance_body_score as sparse

proof, WORK, REPO, require = source.proof, source.WORK, source.REPO, source.require
prepare_sequence, chunk_masks, calib = parity.prepare_sequence, parity.chunk_masks, parity.calib
local, relative, read = source.local, source.relative, source.read
VIEWS = ("top_cam", "side1_cam", "side2_cam")
CASES = ((90258, "side2_cam"), (90259, "side2_cam"), (90141, "side1_cam"))
MASK_VIEWS = ("side1_cam", "side2_cam")
STAGE = "mask_only_saved_transition_cost_ablation_v1"
PRODUCER = WORK / "sparse_instance_five_minute_continuation_v1"
RAW_REVIEW = WORK / "fast_tail_five_minute_id_transition_raw_review_v2"
CHECKPOINT = PRODUCER / "checkpoints/through_091300"
GATE_SNAPSHOT = WORK / "sparse_instance_six_frame_observation_gate_snapshot_v1.json"
COST_SNAPSHOT = WORK / "sparse_instance_six_frame_cost_discrimination_v1.json"
GATE_SHA = "63619f9bbff8e64dbfd9dcdaa1b9f676e1b17c74577259684c89b5509d53e355"
COST_SHA = "2179933708850a03fac129632e3bedb106e9cd00ee0e83922d216b631e7f38f8"
RAW_SHA = "0063b4be2dd9b37200d279c9042dff4862e38d84f9c01446f6559beb146a1924"
CHECKPOINT_SHA = "d1fb9abca83edd772841ac90740454d88370b121f6efc0524bb9187925437bf4"
EXECUTION_READY = True
PINS = {
    "mask_only_instance_observation.py": (
        "6b2cf8b5f5b94bfd211d4a1b133823d91bb56c14ebf91683d460d231c214402b"
    ),
    "score_saved_capacity_island.py": (
        "a64107b63347565135daaaa070a2d0abfde7a72d4024343ebc56d0a72a27821f"
    ),
    "qualify_sparse_instance_saved_10s.py": (
        "801ce45028523e77d23196a65222f22afb8db16bc3a7a99594facfac91779b2a"
    ),
    "qualify_sparse_instance_body_score.py": (
        "b183ca5c537c6cbb85b7d6f5839972b11a5e7c26f7cc0f551724f8c56418ddc1"
    ),
    "run_instance_rescore_10s.py": (
        "91a66646f89a6abb59b6d60bc1b357b383167c21eb9d40bc47491802d05e9544"
    ),
    "sparse_instance_body_score.py": preparer.REFERENCE_SPARSE_SHA256,
    "instance_body_score.py": sparse.REFERENCE_SOURCE_SHA256,
}
BANK_FIELDS = (
    "schema",
    "frame_idx",
    "policy",
    "distance_backend",
    "candidate_index",
    "candidate_pose_xyz",
    "candidate_parent",
    "candidate_plan",
    "candidate_kind",
    "candidate_permutation",
    "base_local_cost",
    "original_five_loss_parts",
    "sixth_instance_term",
    "adjusted_local_cost",
    "parent_path_cost",
    "adjusted_path_cost",
    "capped_contributions_by_view",
    "views",
)
MASK_EVIDENCE = (
    "mask_instance_count",
    "mask_area_pixels",
    "pair_intersection_pixels_01_02_12",
    "pair_mask_distance_px_01_02_12",
)


def exact(a, b, message):
    require(proof.signature(a) == proof.signature(b), message)


def validate_policy(policy):
    expected = dict(
        schema="instance_rescore_policy_v1",
        effective_from_native=76360,
        bodies_per_view=3,
        fixed_camera_denominator=3,
        cap_per_view=9.0,
        weight=3.0,
        camera_order=list(VIEWS),
        unavailable_view_contribution=0.0,
        nonfinite_projection_contribution=9.0,
        post_fit_only=True,
        gradient_objective_unchanged=True,
        original_five_loss_parts_unchanged=True,
        permanent_IDs_assigned=False,
        clears_identity_review=False,
        observation_config=json.loads(json.dumps(dataclasses.asdict(core.DEFAULT_CONFIG))),
    )
    require(all(policy.get(k) == v for k, v in expected.items()), "saved scoring policy differs")


def source_code(binding, raw_seal, evidence):
    """Bind loaded local Python helpers to the existing records or explicit new pins."""
    expected = {**raw_seal["inputs"], **binding["source_sha256"]}
    expected.update({str(WORK / name): digest for name, digest in PINS.items()})
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.suffix != ".py" or not any(
            path.is_relative_to(root) for root in (WORK, REPO / "src", REPO / "labs")
        ):
            continue
        if path == Path(__file__).resolve():
            evidence.add(path)
            continue
        require(proof.valid_sha(expected.get(str(path))), "unbound local source: " + str(path))
        evidence.add(path, expected[str(path)])
    require(not codec.torch.cuda.is_initialized(), "CPU diagnostic initialized CUDA")


def sealed_json(root, seal, name, evidence):
    row = seal["outputs"][name]
    path = relative(root, name)
    require(
        local(row["path"]) == path and path.stat().st_size == row["size_bytes"],
        "sealed output path/size differs",
    )
    return read(path, evidence, row["sha256"])


def validate_observation_binding(binding, raw, manifest, stage):
    chunk = raw["source_chunk"]
    require(
        (chunk["index"], chunk["start"], chunk["stop"]) == (24, 89640, 91300),
        "fixed native chunk differs",
    )
    inputs = binding["original_binding"]["inputs"]
    require(
        raw["shared_SL_calibration_sha256"] == inputs["seed"]["SL_and_calibration_sha256"],
        "bank and observed source calibration/SL bindings differ",
    )
    require(
        manifest["binding"] == stage["params"]["binding"] == inputs["detector_binding"],
        "bank and observed detector binding differs",
    )
    require(
        stage["status"] == "ok" and stage["stage"] == "prepare_independent_mask_chunk",
        "successful independent-mask preparation required",
    )
    outputs = {k: dict(path=v["path"], sha256=v["sha256"]) for k, v in stage["outputs"].items()}
    require(outputs == chunk["output_hashes"], "consumed chunk output references differ")
    require(
        manifest["frames"] == list(range(89640, 91300))
        and manifest["source_interval_half_open"] == [89640, 91300]
        and manifest["fps"] == 166
        and manifest["image_shape_hw"] == [1200, 1200],
        "mask native extent differs",
    )


def load_headers(evidence):
    """Small JSON/source reads only; no candidate, mask, calibration payload or SL read."""
    gates = read(GATE_SNAPSHOT, evidence, GATE_SHA)
    costs = read(COST_SNAPSHOT, evidence, COST_SHA)
    require(
        gates["schema"] == "sparse_instance_six_frame_observation_gate_snapshot_v1"
        and costs["schema"] == "sparse_instance_six_frame_cost_discrimination_v1"
        and local(gates["root"]) == PRODUCER,
        "fixed snapshot source differs",
    )
    require(
        gates["completed_checkpoint_manifest_sha256"] == CHECKPOINT_SHA,
        "completed checkpoint reference differs",
    )
    manifest = read(CHECKPOINT / "manifest.json", evidence, CHECKPOINT_SHA)
    require(
        manifest["schema"] == "medaka_tracker_checkpoint_v1" and manifest["status"] == "ok",
        "completed checkpoint header required",
    )
    metadata_path = relative(CHECKPOINT, manifest["metadata"]["file"])
    require(
        metadata_path.name == "metadata.json"
        and metadata_path.stat().st_size == manifest["metadata"]["bytes"],
        "checkpoint metadata descriptor differs",
    )
    metadata = read(metadata_path, evidence, manifest["metadata"]["sha256"])
    require(
        metadata["previous_frame"] == 91299
        and metadata["processed_frames"] == 41500
        and metadata["initialize_count"] == 1
        and metadata["restore_count"] == 6,
        "saved checkpoint native extent/lifecycle differs",
    )
    binding = manifest["bindings"]
    require(
        binding["schema"] == "sparse_instance_five_minute_continuation_v1_checkpoint"
        and binding["process_interval"] == [78020, 116200],
        "saved producer stage differs",
    )
    validate_policy(binding["scoring_policy"])
    raw_seal = read(RAW_REVIEW / "render.provenance.json", evidence, RAW_SHA)
    require(
        raw_seal["stage"] == "fast_tail_five_minute_id_transition_raw_review_v1"
        and raw_seal["status"] == "ok",
        "successful fixed raw source evidence required",
    )
    raw = sealed_json(RAW_REVIEW, raw_seal, "source_and_frame_evidence.json", evidence)
    chunk = raw["source_chunk"]
    chunk_path = local(chunk["output_hashes"]["manifest"]["path"])
    require(
        raw_seal["inputs"].get(str(chunk_path)) == chunk["manifest_sha256"],
        "raw evidence manifest edge differs",
    )
    mask_manifest = read(chunk_path, evidence, chunk["manifest_sha256"])
    stage_path = chunk_path.parent / "staging.provenance.json"
    require(
        raw_seal["inputs"].get(str(stage_path)) == chunk["provenance_sha256"],
        "raw evidence staging edge differs",
    )
    stage = read(stage_path, evidence, chunk["provenance_sha256"])
    validate_observation_binding(binding, raw, mask_manifest, stage)
    require(
        chunk_path.parent
        == local(
            Path(binding["original_binding"]["inputs"]["input_stage_root"]) / chunk["attempt_path"]
        ),
        "mask staging root differs",
    )
    context = dict(
        binding=binding, raw=raw, raw_seal=raw_seal, manifest=mask_manifest, chunk_path=chunk_path
    )
    for view in MASK_VIEWS:
        mask_descriptor(context, view)
    source_code(binding, raw_seal, evidence)
    gate_rows = {r["frame"]: r for r in gates["records"]}
    cost_rows = {r["frame"]: r for r in costs["records"]}
    raw_rows = {r["frame"]: r for r in raw["records"]}
    headers = {}
    for frame, view in CASES:
        row = gate_rows[frame]
        cost = cost_rows[frame]
        require(
            row["diagnostic_json_sha256"] == cost["diagnostic_json_sha256"],
            "gate/cost snapshot JSON references disagree",
        )
        saved = read(
            PRODUCER / f"diagnostics/f{frame}.json", evidence, row["diagnostic_json_sha256"]
        )
        require(
            saved["tree"]["frame_idx"] == frame
            and saved["tree"]["policy"] == binding["scoring_policy"]
            and saved["tree"]["distance_backend"] == binding["sparse_distance_backend"],
            "saved frame/policy/backend differs",
        )
        require(
            saved["archive"]["sha256"] == cost["NPZ_sha256"]
            and local(saved["archive"]["path"]) == PRODUCER / f"diagnostics/f{frame}.npz",
            "snapshot companion reference differs",
        )
        for camera in VIEWS:
            actual = saved["tree"]["views"][camera]
            require(
                {k: actual[k] for k in row["views"][camera]} == row["views"][camera],
                "saved view gate differs from snapshot",
            )
        headers[frame] = dict(
            saved=saved, snapshot=cost, raw_observation=raw_rows[frame]["observations"][view]
        )
    return dict(**context, headers=headers)


def array_descriptors(tree):
    result = {}

    def visit(item):
        if isinstance(item, dict) and "__array__" in item:
            require(
                set(item) == {"__array__", "signature"} and item["__array__"] not in result,
                "duplicate/invalid array descriptor",
            )
            result[item["__array__"]] = item["signature"]
        elif isinstance(item, dict):
            for value in item.values():
                visit(value)
        elif isinstance(item, list):
            for value in item:
                visit(value)

    visit(tree)
    return result


def capture_archive(path, digest, captures, *, keys=None, all_keys=None, size=None):
    path = local(path)
    require(str(path) not in captures, "archive would be captured twice")
    before = parity.file_stat(path)
    payload = path.read_bytes()
    require(
        hashlib.sha256(payload).hexdigest() == digest
        and parity.file_stat(path) == before
        and (size is None or len(payload) == size),
        "captured archive SHA/stat/size differs",
    )
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        require(
            len(archive.files) == len(set(archive.files))
            and (all_keys is None or set(archive.files) == set(all_keys)),
            "archive member set differs",
        )
        arrays = {k: archive[k] for k in (archive.files if keys is None else keys)}
    for array in arrays.values():
        require(not array.dtype.hasobject, "object payload forbidden")
        array.flags.writeable = False
    record = dict(
        path=str(path),
        sha256=digest,
        file_stat=before,
        payload_reads=1,
        decoded_arrays={k: proof.signature(v) for k, v in arrays.items()},
    )
    captures[str(path)] = (record, arrays)
    return arrays


def load_bank(header, captures):
    saved = header["saved"]
    tree = {key: saved["tree"][key] for key in BANK_FIELDS}
    descriptors = array_descriptors(tree)
    row = saved["archive"]
    arrays = capture_archive(
        row["path"],
        row["sha256"],
        captures,
        keys=descriptors,
        all_keys=array_descriptors(saved["tree"]),
        size=row["size_bytes"],
    )
    bank = codec.restore_diagnostic(tree, arrays)
    validate_bank(bank)
    require(
        len(bank["candidate_index"]) == header["snapshot"]["candidate_count"],
        "snapshot candidate count differs",
    )
    return bank


def validate_bank(bank):
    validate_policy(bank["policy"])
    require(bank["schema"] == "instance_rescore_frame_diagnostics_v1", "wrong candidate schema")
    count = len(bank["candidate_index"])
    require(count > 0, "empty candidate bank")
    exact(
        bank["candidate_index"], np.arange(count, dtype=np.int64), "candidate index order differs"
    )
    shapes = dict(
        candidate_pose_xyz=(count, 3, 3, 3),
        candidate_parent=(count,),
        candidate_plan=(count, 3, 3, 3),
        candidate_permutation=(count, 3),
        original_five_loss_parts=(count, 5),
        capped_contributions_by_view=(count, 3),
    )
    for key in (
        "base_local_cost",
        "sixth_instance_term",
        "adjusted_local_cost",
        "parent_path_cost",
        "adjusted_path_cost",
    ):
        shapes[key] = (count,)
    for key, shape in shapes.items():
        value = bank[key]
        dtype = (
            "<i4"
            if key == "candidate_plan"
            else "<i8"
            if key in ("candidate_parent", "candidate_permutation")
            else "<f8"
        )
        require(
            value.shape == shape and value.dtype.str == dtype,
            "candidate field shape/type differs: " + key,
        )
        if key != "candidate_pose_xyz":
            require(np.isfinite(value).all(), "nonfinite saved cost/index field: " + key)
    require(
        len(bank["candidate_kind"]) == count and np.all(bank["candidate_parent"] >= 0),
        "candidate kind/parent extent differs",
    )
    require(
        np.all(np.sort(bank["candidate_permutation"], axis=1) == np.arange(3)),
        "candidate permutation is not bijective",
    )
    contributions = bank["capped_contributions_by_view"]
    require(
        np.all((contributions >= 0) & (contributions <= 9)), "saved view contribution out of bounds"
    )
    for k, view in enumerate(VIEWS):
        exact(
            contributions[:, k],
            bank["views"][view]["capped_view_contribution"],
            "saved per-view column differs",
        )
    sixth = (
        bank["policy"]["weight"]
        * contributions.sum(axis=1)
        / bank["policy"]["fixed_camera_denominator"]
    )
    exact(sixth, bank["sixth_instance_term"], "saved sixth term arithmetic differs")
    local_cost = bank["base_local_cost"] + sixth
    exact(local_cost, bank["adjusted_local_cost"], "saved local cost arithmetic differs")
    exact(
        bank["parent_path_cost"] + local_cost,
        bank["adjusted_path_cost"],
        "saved path cost arithmetic differs",
    )


def replace_view_contribution(bank, view, scored):
    """Keep other view columns and parent costs exact; no temporal state update."""
    validate_bank(bank)
    column = VIEWS.index(view)
    compact = parity.compact_view(scored, bank["policy"]["cap_per_view"])
    contribution = compact["capped_view_contribution"]
    require(
        contribution.shape == bank["base_local_cost"].shape, "new score candidate extent differs"
    )
    updated = bank["capped_contributions_by_view"].copy()
    updated[:, column] = contribution
    other = [k for k in range(3) if k != column]
    exact(
        updated[:, other],
        bank["capped_contributions_by_view"][:, other],
        "unchanged view contribution changed",
    )
    sixth = (
        bank["policy"]["weight"] * updated.sum(axis=1) / bank["policy"]["fixed_camera_denominator"]
    )
    local_cost = bank["base_local_cost"] + sixth
    path_cost = bank["parent_path_cost"] + local_cost
    require(
        np.isfinite(local_cost).all() and np.isfinite(path_cost).all(),
        "new candidate costs are nonfinite",
    )
    costs = dict(
        old_local_cost=bank["adjusted_local_cost"],
        new_local_cost=local_cost,
        old_fixed_parent_path_cost=bank["adjusted_path_cost"],
        new_fixed_parent_path_cost=path_cost,
    )
    rankings = {
        key: dict(
            stable_candidate_order=np.argsort(value, kind="stable").tolist(),
            exact_minimum_tied_candidates=np.flatnonzero(value == value.min()).tolist(),
        )
        for key, value in costs.items()
    }
    return dict(
        new_view=compact,
        new_capped_contributions_by_view=updated,
        new_sixth_instance_term=sixth,
        **costs,
    ), rankings


def mask_descriptor(context, view):
    """Actual raw-review edge: sealed evidence contains captured-mask records."""
    descriptor = context["manifest"]["masks"][view]
    path = relative(context["chunk_path"].parent, descriptor["path"])
    role = context["raw"]["source_chunk"]["output_hashes"][view + "_masks"]
    captured = context["raw"]["captured_mask_records"][view]
    require(
        local(role["path"]) == path == local(captured["path"])
        and role["sha256"] == descriptor["sha256"] == captured["sha256"],
        "consumed/staged/captured mask edge differs",
    )
    require(
        parity.file_stat(path) == captured["file_stat"]
        and set(captured["arrays"]) == set(chunk_masks.SCHEMA_DTYPES),
        "captured mask stat/array membership differs",
    )
    return path, descriptor["sha256"], captured


def load_masks_and_calibration(context, evidence, captures):
    manifest = context["manifest"]
    packed = {}
    for view in MASK_VIEWS:
        path, digest, captured = mask_descriptor(context, view)
        arrays = capture_archive(path, digest, captures, all_keys=chunk_masks.SCHEMA_DTYPES)
        require(
            {k: proof.signature(v) for k, v in arrays.items()} == captured["arrays"],
            "captured mask decoded signatures differ from sealed raw review",
        )
        require(
            all(arrays[k].dtype == v for k, v in chunk_masks.SCHEMA_DTYPES.items()),
            "packed mask dtype differs",
        )
        require(
            np.array_equal(arrays["frame_idx"], manifest["frames"])
            and np.array_equal(arrays["image_shape_hw"], [1200, 1200])
            and arrays["view_available"].shape == arrays["frame_mask_sha256"].shape == (1660,),
            "packed native mask extent differs",
        )
        packed[view] = arrays
    shared = context["raw"]["shared_SL_calibration_sha256"]
    paths = {local(p): digest for p, digest in shared.items() if Path(p).suffix == ".json"}
    require(len(paths) == 4, "exactly four calibration files required")
    manifests = [p for p in paths if p.name == "calibration_manifest.json"]
    require(len(manifests) == 1, "one calibration manifest required")
    for path, digest in paths.items():
        require(
            context["raw_seal"]["inputs"].get(str(path)) == digest,
            "calibration source edge differs",
        )
        evidence.add(path, digest)
    config = json.loads(manifests[0].read_text(encoding="utf8"))
    model_paths = {relative(manifests[0].parent, r["path"]) for r in config["model_files"]}
    require(
        len(model_paths) == 3 and model_paths | set(manifests) == set(paths),
        "calibration membership differs",
    )
    rig = calib.load_rig(manifests[0].parent)
    require(tuple(rig.views) == VIEWS and rig.world_units == "mm", "calibrated views/units differ")
    return packed, rig


def write_result(out, name, value):
    tree, arrays = codec.split_diagnostic(value)
    record = out.write(name + ".npz", arrays)
    with np.load(record["path"], allow_pickle=False) as archive:
        restored = {k: archive[k] for k in archive.files}
    codec.restore_diagnostic(tree, restored)
    out.write(name + ".json", dict(tree=tree, archive=record))
    return record


def aligned_detector_scores(packed, index, count):
    """Read already captured framewise scores, without using them in any score/gate."""
    offsets = packed["frame_offsets"]
    require(0 <= index < len(offsets) - 1, "detector score native index out of bounds")
    lo, hi = map(int, offsets[index : index + 2])
    require(
        0 <= lo <= hi <= len(packed["scores"]) and hi - lo == count,
        "detector score/mask extent differs",
    )
    ids = packed["instance_id"][lo:hi]
    exact(ids, np.arange(count, dtype=np.int64), "detector score instance order differs")
    scores = packed["scores"][lo:hi]
    require(
        scores.dtype == np.float32 and scores.shape == (count,) and np.isfinite(scores).all(),
        "detector scores malformed",
    )
    return dict(
        scores=scores.copy(),
        within_frame_instance_id=ids.copy(),
        used_for_eligibility_or_ranking=False,
        physical_IDs=False,
    )


def execute(args):
    require(EXECUTION_READY and args.execute, "reviewed CPU execution is not enabled")
    output = args.output.resolve()
    require(output.is_relative_to(WORK) and not output.exists(), "fresh workspace output required")
    out, evidence, captures = proof.Output(output), proof.Evidence(), {}
    started, records = proof.utc(), []
    try:
        out.write(
            "plan.json",
            dict(
                schema=STAGE,
                cases=[
                    dict(
                        frame=f,
                        view=v,
                        role="reflection_negative_control" if f == 90141 else "development_case",
                    )
                    for f, v in CASES
                ],
                gate_snapshot_sha256=GATE_SHA,
                cost_snapshot_sha256=COST_SHA,
                completed_checkpoint_sha256=CHECKPOINT_SHA,
                raw_source_evidence_sha256=RAW_SHA,
                frozen_observation_config=dataclasses.asdict(core.DEFAULT_CONFIG),
                changed_rule=preparer.ELIGIBILITY_POLICY,
                thresholds_retuned=False,
                camera_order=list(VIEWS),
                weight=3.0,
                cap_per_view=9.0,
                bodies_per_view=3,
                fixed_camera_denominator=3,
                path_arithmetic="saved_parent + (saved_base + new_sixth)",
                actual_temporal_selection=False,
                final_horizon_selection_read=False,
                original_live_snapshot_is_final_success=False,
                selected_frames_are_holdout=False,
                physical_ID_truth=False,
                tracking_adoption=False,
            ),
        )
        context = load_headers(evidence)
        packed, rig = load_masks_and_calibration(context, evidence, captures)
        for frame, view in CASES:
            header = context["headers"][frame]
            bank = load_bank(header, captures)
            native_index = frame - context["raw"]["source_chunk"]["start"]
            arrays = packed[view]
            decoded = prepare_sequence.decode_frame_masks(arrays, native_index)
            mask_sha = prepare_sequence.array_sha256(decoded)
            raw = header["raw_observation"]
            available = bool(arrays["view_available"][native_index])
            require(
                mask_sha
                == bytes(arrays["frame_mask_sha256"][native_index]).decode("ascii")
                == raw["mask_sha256"]
                and len(decoded) == raw["mask_count"]
                and available == raw["mask_available"]
                and (available or len(decoded) == 0),
                "native mask SHA/count/availability differs",
            )
            observation = preparer.prepare_mask_only_instance_observation(
                decoded if available else None
            )
            detector_scores = aligned_detector_scores(arrays, native_index, len(decoded))
            prior = bank["views"][view]
            for key in MASK_EVIDENCE:
                require(
                    observation.evidence.get(key) == prior["observation_evidence"].get(key),
                    "mask-only geometry gate evidence differs",
                )
            scored = sparse.score_sparse_instance_bodies(
                bank["candidate_pose_xyz"], rig.camera(view).project, observation
            )
            updated, rankings = replace_view_contribution(bank, view, scored)
            require(prepare_sequence.array_sha256(decoded) == mask_sha, "decoded mask changed")
            result = dict(
                frame=frame,
                view=view,
                input_candidate_bank=bank,
                new_score=scored,
                candidate_cost_ablation=updated,
                rankings=rankings,
                candidate_bank_or_parents_modified=False,
                physical_ID_truth=False,
                actual_temporal_selection=False,
                unchanged_view_contributions_exact=True,
                detector_scores_diagnostic_only=detector_scores,
            )
            archive = write_result(out, f"cases/f{frame}_{view}", result)
            records.append(
                dict(
                    frame=frame,
                    view=view,
                    old_eligible=prior["eligible"],
                    old_reason=prior["reason"],
                    new_eligible=observation.eligible,
                    new_reason=observation.reason,
                    candidate_count=len(bank["candidate_index"]),
                    rankings=rankings,
                    mask_sha256=mask_sha,
                    result_archive=archive,
                    capped_candidates=int(updated["new_view"]["cap_applied"].sum()),
                    unchanged_view_contributions_exact=True,
                    detector_scores_in_mask_order=detector_scores["scores"].tolist(),
                    detector_scores_used_for_eligibility_or_ranking=False,
                )
            )
            print(json.dumps(records[-1]), flush=True)
        for record, arrays in captures.values():
            require(
                parity.file_stat(Path(record["path"])) == record["file_stat"],
                "captured input file stat changed",
            )
            require(
                {k: proof.signature(v) for k, v in arrays.items()} == record["decoded_arrays"],
                "captured input arrays changed",
            )
        require(
            len(captures) == 5 and not codec.torch.cuda.is_initialized(),
            "bounded capture count/CUDA differs",
        )
        out.write("input_capture_records.json", [r for r, _ in captures.values()])
        out.write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                records=records,
                candidate_bank_decodes=3,
                packed_mask_archive_reads=2,
                native_mask_decodes=3,
                calibration_files_read=4,
                SLEAP_payload_reads=0,
                raw_video_reads=0,
                checkpoint_payload_reads=0,
                GPU_used=False,
                temporal_tracking_calls=0,
                candidate_cost_ablation_only=True,
                original_five_parts_unchanged=True,
                parent_path_costs_unchanged=True,
                unchanged_view_contributions_exact=True,
                full5_or_physical_ID_accuracy_established=False,
                thresholds_retuned=False,
                limitations=[
                    "Development-selected candidate banks, not held-out truth.",
                    "Rankings condition on the original fitted pool and original parent histories.",
                    (
                        "Three separated masks can include a reflection; "
                        "eligibility is not fish verification."
                    ),
                    "Lower image cost does not establish correct anatomy or physical identity.",
                ],
                final_captured_payload_guard=(
                    "same file stat and unchanged decoded arrays; no second file payload hash"
                ),
            ),
        )
        evidence.final_guard()
        out.final_guard()
        status, error = "ok", None
    except BaseException as exc:
        status, error = "failed", repr(exc)
        raise
    finally:
        seal = dict(
            stage=STAGE,
            status=status,
            error=error,
            started_at=started,
            finished_at=proof.utc(),
            inputs={**evidence.paths, **{p: r["sha256"] for p, (r, _) in captures.items()}},
            outputs=out.records,
        )
        (output / "scoring.provenance.json").write_bytes(proof.json_bytes(seal))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--execute", action="store_true")
    return result


if __name__ == "__main__":
    execute(parser().parse_args())
