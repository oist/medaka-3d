"""CPU-only reranking of eleven saved fit banks; no new trajectory or ID claim.

The original parent cost and five-part fitted local cost remain fixed. Replace
the sixth term with bounded partial-bundle penalties, once with mask weight zero
and once with the fixed row+mask policy. No original candidate, SLEAP cache,
video, checkpoint, fitter, tracker or identity solver is read or executed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import io
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import render_fast_tail_five_minute_id_transition as frozen
import run_instance_rescore_10s as codec

proof, require, prior = frozen.proof, frozen.require, frozen.prior
WORK, REPO, VIEWS = frozen.WORK, frozen.REPO, frozen.VIEWS
STAGE = "partial_body_saved_bank_comparison_v1"
FRAMES = (82720, 82727, 82728, 82750)
NEGATIVE_FRAMES = (90141, 90258, 90259)
ALL_FRAMES = FRAMES + NEGATIVE_FRAMES
ARMS = ("sparse_instance_rescore", "expanded_depth_instance_rescore")
TRACKING_ROOTS = dict(
    zip(
        ARMS,
        (
            WORK / "sparse_instance_five_minute_continuation_v1",
            WORK / "expanded_depth_five_minute_continuation_v1",
        ),
        strict=True,
    )
)
TRACKING_SHAS = dict(
    zip(
        ARMS,
        (
            "596b23afe5fb572a66a72b96b0b1f5ba9b3971bfef4dd43cb8ee6b32459e32ad",
            "44da9cd5e55b02992156251e97d3a8e331d535399116bddcbaa3135f31f486ec",
        ),
        strict=True,
    )
)
VISUAL_ROOT = WORK / "depth_identity_transition_raw_review_v1"
VISUAL_SHA = "9f3721806523e5b656f317684ffffa19cc5c08b94341a5e353d1bd5af383deea"
EVIDENCE_SHA = "b0176b780b77c7205911fae1839aa8771bece0bb5fa368ce4afac902f4435f74"
VISUAL_SOURCE_SHA = "26dcca519ee7364a55e9c40af6579eec13d9bdb3c253dc5d3ef66cc552d5312a"
SCORER_SHA = "fb99820558d317a70b4d170f67c8a94c5cf3ead9e319f068568afa651d6054b6"
NEGATIVE_ROOT = WORK / "fast_tail_five_minute_id_transition_raw_review_v2"
NEGATIVE_SHA = "0063b4be2dd9b37200d279c9042dff4862e38d84f9c01446f6559beb146a1924"
NEGATIVE_EVIDENCE_SHA = "5405ce02b34cd4c1673487f7427d101ccf333cfef869dc740a9acb426ea70d41"
NEGATIVE_CHUNK_SHA = "17a9cf8721d1dc4868b7bbc9881e0222e47b6b84a8955ccb0b4b8b29e840cd92"
CODEC_SHA = "91a66646f89a6abb59b6d60bc1b357b383167c21eb9d40bc47491802d05e9544"
PACK_HELPER_SHA = "169c8a7dcd7e3ab932412a968ddae4e04574d99c4e517ee986cdbe4aa544236e"
CHUNK_SHA = "52d88451c0c1162c004acef567992802e5452ea8505fcd86a848f9ab356074ce"
MODES = ("baseline", "row_only", "row_plus_mask")
NORMALIZER = 15.0  # Shared coefficient: mask_weight=0 does not amplify semantic gain.


def case_keys():
    return [(a, f) for f in FRAMES for a in ARMS] + [(ARMS[0], f) for f in NEGATIVE_FRAMES]


def edge(seal, path, digest):
    require(seal["inputs"].get(str(Path(path).resolve())) == digest, "consumed input edge differs")


def read(path, digest, evidence):
    evidence.add(path, digest)
    return proof.read_json(path)


def member(root, seal, name, evidence):
    path = prior.native.checked_relative(root, name)
    record = seal["outputs"][name]
    require(
        str(path) == record["path"] and path.stat().st_size == record["size_bytes"],
        "sealed member path/size differs",
    )
    evidence.add(path, record["sha256"])
    return path


def load_headers(evidence):
    visual = read(VISUAL_ROOT / "render.provenance.json", VISUAL_SHA, evidence)
    require(
        visual["status"] == "ok" and visual["stage"] == "depth_identity_transition_raw_review_v1",
        "successful saved visual evidence required",
    )
    edge(visual, WORK / "render_depth_identity_transition_review.py", VISUAL_SOURCE_SHA)
    path = member(VISUAL_ROOT, visual, "source_and_frame_evidence.json", evidence)
    require(
        visual["outputs"]["source_and_frame_evidence.json"]["sha256"] == EVIDENCE_SHA,
        "fixed frame evidence differs",
    )
    doc = proof.read_json(path)
    plan = proof.read_json(member(VISUAL_ROOT, visual, "plan.json", evidence))
    require(
        doc["status"] == "ok"
        and doc["terminal_horizon"] == 116200
        and doc["arms"] == list(ARMS)
        and doc["physical_ID_truth"] is False
        and doc["physical_ID_mapping_applied"] is False,
        "same-final raw-slot evidence required",
    )
    records = {r["frame"]: r for r in doc["records"] if r["frame"] in FRAMES}
    require(
        len(records) == len(FRAMES) and all(r["frame"] == f for f, r in records.items()),
        "four fixed frame observations required",
    )
    chunk = doc["source_chunks"]["19"]
    require(
        (chunk["index"], chunk["start"], chunk["stop"]) == (19, 81340, 83000)
        and chunk["manifest_sha256"] == CHUNK_SHA,
        "one exact consumed chunk required",
    )
    tracking = {}
    for arm in ARMS:
        root, digest = TRACKING_ROOTS[arm], TRACKING_SHAS[arm]
        edge(visual, root / "tracking.provenance.json", digest)
        require(
            doc["tracking_sha256"][arm] == digest
            and doc["candidate_references"][arm]["tracking_seal_sha256"] == digest,
            "frame evidence producer differs",
        )
        seal = read(root / "tracking.provenance.json", digest, evidence)
        require(
            seal["status"] == "ok" and seal["stage"] == root.name,
            "successful final producer required",
        )
        tracking[arm] = seal
    for r in records.values():
        require(
            set(r["observations"]) == set(VIEWS) and set(r["methods"]) == set(ARMS),
            "three-view two-arm evidence required",
        )
        for ob in r["observations"].values():
            require(
                ob["source_chunk_index"] == 19
                and ob["canonical_video_frame"] == r["frame"]
                and ob["chunk_local_offset"] == r["frame"] - 81340,
                "saved native SL/mask position differs",
            )
    old_seal = read(NEGATIVE_ROOT / "render.provenance.json", NEGATIVE_SHA, evidence)
    require(
        old_seal["status"] == "ok"
        and old_seal["stage"] == "fast_tail_five_minute_id_transition_raw_review_v1",
        "successful negative-control observation evidence required",
    )
    old = proof.read_json(
        member(NEGATIVE_ROOT, old_seal, "source_and_frame_evidence.json", evidence)
    )
    require(
        old_seal["outputs"]["source_and_frame_evidence.json"]["sha256"] == NEGATIVE_EVIDENCE_SHA
        and old["status"] == "ok"
        and old["terminal_horizon"] == 116200,
        "fixed negative-control evidence differs",
    )
    old_chunk = old["source_chunk"]
    require(
        (old_chunk["index"], old_chunk["start"], old_chunk["stop"]) == (24, 89640, 91300)
        and old_chunk["manifest_sha256"] == NEGATIVE_CHUNK_SHA,
        "negative-control chunk differs",
    )
    require(
        old["shared_SL_calibration_sha256"] == doc["shared_SL_calibration_sha256"],
        "negative/main calibration or SL source differs",
    )
    old_records = {r["frame"]: r for r in old["records"] if r["frame"] in NEGATIVE_FRAMES}
    require(set(old_records) == set(NEGATIVE_FRAMES), "all three negative controls required")
    for frame, row in old_records.items():
        require(set(row["observations"]) == set(VIEWS), "negative-control views differ")
        # Its saved candidate is OldFast, so never treat it as final Sparse truth.
        records[frame] = dict(frame=frame, observations=row["observations"], methods={})
    return dict(
        visual=visual,
        visual_plan=plan,
        evidence=doc,
        records=records,
        tracking=tracking,
        observation_groups={
            19: dict(chunk=chunk, frames=FRAMES, captures=doc["captured_mask_records"]["19"]),
            24: dict(
                chunk=old_chunk, frames=NEGATIVE_FRAMES, captures=old["captured_mask_records"]
            ),
        },
    )


def source_code(context, evidence):
    require(proof.valid_sha(SCORER_SHA), "new scorer source must be reviewed and pinned")
    code = dict(context["visual_plan"]["source_code"])
    expected = {
        str(Path(codec.__file__).resolve()): CODEC_SHA,
        str(Path(frozen.__file__).resolve()): PACK_HELPER_SHA,
        str(WORK / "partial_body_bundle_score.py"): SCORER_SHA,
    }
    for path, digest in expected.items():
        require(path not in code or code[path] == digest, "reused source differs from visual input")
        code[path] = digest
    code[str(Path(__file__).resolve())] = proof.sha(__file__)
    for path, digest in code.items():
        evidence.add(Path(path), digest)
    return code


def load_banks(context, evidence):
    banks, references = {}, {}
    for arm in ARMS:
        root, seal = TRACKING_ROOTS[arm], context["tracking"][arm]
        for frame in FRAMES + (NEGATIVE_FRAMES if arm == ARMS[0] else ()):
            name = f"diagnostics/f{frame}.json"
            doc = proof.read_json(member(root, seal, name, evidence))
            archive_name = f"diagnostics/f{frame}.npz"
            record = seal["outputs"][archive_name]
            require(doc["archive"] == record, "companion/archive producer binding differs")
            path = prior.native.checked_relative(root, archive_name)
            require(
                str(path) == record["path"] and path.stat().st_size == record["size_bytes"],
                "saved bank path/size differs",
            )
            # One capture/decode; the final Evidence guard only rehashes bytes.
            before_stat = frozen.observations.file_stat(path)
            payload = path.read_bytes()
            require(
                hashlib.sha256(payload).hexdigest() == record["sha256"]
                and frozen.observations.file_stat(path) == before_stat,
                "saved bank SHA/stat differs",
            )
            require(
                str(path) not in evidence.paths or evidence.paths[str(path)] == record["sha256"],
                "bank capture disagrees with earlier input",
            )
            evidence.paths[str(path)] = record["sha256"]
            with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
                require(
                    len(archive.files) == len(set(archive.files)), "duplicate bank array member"
                )
                value = codec.restore_diagnostic(
                    doc["tree"], {k: archive[k] for k in archive.files}
                )
            validate_bank(value, frame)
            banks[(arm, frame)] = value
            references[f"{arm}/{frame}"] = dict(
                companion=seal["outputs"][name],
                archive=record,
                arrays={
                    k: v["signature"]
                    for k, v in doc["tree"].items()
                    if isinstance(v, dict) and "__array__" in v
                },
            )
    return banks, references


def validate_bank(bank, frame):
    require(
        bank["schema"] == "instance_rescore_frame_diagnostics_v1"
        and bank["frame_idx"] == frame
        and bank["post_fit_only"] is True
        and bank["temporal_state"] is False,
        "actual fitted bank scope differs",
    )
    n = len(bank["candidate_index"])
    require(
        np.array_equal(bank["candidate_index"], np.arange(n))
        and bank["candidate_pose_xyz"].shape == (n, 3, 3, 3)
        and bank["candidate_plan"].shape == (n, 3, 3, 3)
        and bank["candidate_parent"].shape == (n,)
        and bank["original_five_loss_parts"].shape == (n, 5),
        "bank shapes differ",
    )
    require(
        bank["candidate_pose_xyz"].dtype == np.float64
        and bank["candidate_plan"].dtype == np.int32
        and bank["candidate_parent"].dtype == np.int64
        and len(bank["candidate_kind"]) == n,
        "bank dtypes differ",
    )
    for key in (
        "base_local_cost",
        "sixth_instance_term",
        "adjusted_local_cost",
        "parent_path_cost",
        "adjusted_path_cost",
    ):
        require(
            bank[key].shape == (n,)
            and bank[key].dtype == np.float64
            and np.isfinite(bank[key]).all(),
            "finite saved cost vector required",
        )
    require(np.isfinite(bank["candidate_pose_xyz"]).all(), "finite saved poses required")
    require(
        np.array_equal(
            bank["base_local_cost"] + bank["sixth_instance_term"], bank["adjusted_local_cost"]
        )
        and np.array_equal(
            bank["parent_path_cost"] + bank["adjusted_local_cost"], bank["adjusted_path_cost"]
        ),
        "original cost bookkeeping differs",
    )


def load_observations(context):
    """Capture six archives serially; retain only seven target masks per view."""
    decoded, captures, points = {}, {}, {}
    for frame in ALL_FRAMES:
        points[frame], decoded[frame] = {}, {}
        for view in VIEWS:
            row = context["records"][frame]["observations"][view]
            p = np.asarray(row["SLEAP_points"], dtype=np.float32)
            require(
                proof.signature(p) == row["SL_points_signature"],
                "saved SL point reconstruction differs",
            )
            p.flags.writeable = False
            points[frame][view] = p
    for index, group in context["observation_groups"].items():
        chunk = group["chunk"]
        captures[str(index)] = {}
        for view in VIEWS:
            record = chunk["output_hashes"][view + "_masks"]
            path = Path(record["path"]).resolve()
            require(path.is_relative_to(REPO), "mask escaped workspace")
            for seal in context["tracking"].values():
                edge(seal, path, record["sha256"])
            require(
                group["captures"][view]["sha256"] == record["sha256"]
                and group["captures"][view]["path"] == str(path),
                "visual mask capture differs",
            )
            packed, captures[str(index)][view] = frozen.capture_masks(
                path, record["sha256"], np.arange(chunk["start"], chunk["stop"], dtype=np.int64)
            )
            for frame in group["frames"]:
                pos = frame - chunk["start"]
                mask = prior.prepare_sequence.decode_frame_masks(packed, pos)
                digest = prior.prepare_sequence.array_sha256(mask)
                saved = context["records"][frame]["observations"][view]
                require(
                    digest
                    == bytes(packed["frame_mask_sha256"][pos]).decode()
                    == saved["mask_sha256"]
                    and bool(packed["view_available"][pos]) == saved["mask_available"]
                    and len(mask) == saved["mask_count"],
                    "saved mask stack differs",
                )
                mask.flags.writeable = False
                decoded[frame][view] = mask if saved["mask_available"] else None
            require(
                {k: proof.signature(v) for k, v in packed.items()}
                == captures[str(index)][view]["arrays"],
                "captured mask arrays changed",
            )
            del packed
    return decoded, points, captures


def load_rig(context, evidence):
    shared = context["evidence"]["shared_SL_calibration_sha256"]
    calibration = {p: d for p, d in shared.items() if Path(p).suffix == ".json"}
    require(len(calibration) == 4, "registered rig4 required")
    for path, digest in calibration.items():
        edge(context["visual"], Path(path), digest)
        for seal in context["tracking"].values():
            edge(seal, Path(path), digest)
        evidence.add(Path(path), digest)
    manifests = [Path(p) for p in calibration if Path(p).name == "calibration_manifest.json"]
    require(len(manifests) == 1, "one rig manifest required")
    manifest = proof.read_json(manifests[0])
    models = {
        str(prior.native.checked_relative(manifests[0].parent, r["path"]))
        for r in manifest["model_files"]
    }
    require(models | {str(manifests[0])} == set(calibration), "rig membership differs")
    rig = frozen.observations.calib.load_rig(manifests[0].parent)
    require(tuple(rig.views) == tuple(VIEWS) and rig.world_units == "mm", "rig views/units differ")
    return rig


def exact_final_matches(bank, method):
    if method is None:
        return dict(
            available=False,
            reason=(
                "negative control has no registered final Sparse pose; "
                "old visual candidate belongs to OldFast"
            ),
        )
    pose = np.asarray(method["pose_xyz_mm"], np.float64)
    sources = np.asarray(method["sources"], np.int32)
    require(pose.shape == sources.shape == (3, 3, 3), "saved final body shape differs")
    pose_matches = np.all(bank["candidate_pose_xyz"] == pose, axis=(1, 2, 3))
    plan_matches = np.all(
        (sources[None] < 0) | (bank["candidate_plan"] == sources[None]), axis=(1, 2, 3)
    )
    require(pose_matches.any(), "final backtrace pose absent from actual fit bank")
    require((pose_matches & plan_matches).any(), "final supported sources absent from pose matches")
    return dict(
        available=True,
        exact_pose_candidate_indices=np.flatnonzero(pose_matches),
        exact_pose_and_supported_source_candidate_indices=np.flatnonzero(
            pose_matches & plan_matches
        ),
        multiple_matches_are_unresolved=True,
    )


def rank_costs(local, parent):
    require(
        local.shape == parent.shape
        and local.ndim == 1
        and not np.isnan(local).any()
        and not np.isneginf(local).any()
        and np.isfinite(parent).all()
        and np.isfinite(local).any(),
        "same-bank costs require at least one finite candidate; no NaN or negative infinity",
    )
    path = parent + local
    require(np.isfinite(path[np.isfinite(local)]).all(), "finite path cost overflow")
    return dict(
        local_cost=local,
        path_cost=path,
        best_local_cost=float(local.min()),
        best_path_cost=float(path.min()),
        local_order=np.argsort(local, kind="stable"),
        path_order=np.argsort(path, kind="stable"),
        local_minimum_indices=np.flatnonzero(local == local.min()),
        path_minimum_indices=np.flatnonzero(path == path.min()),
        minimum_ties_are_exact_float64=True,
    )


def bounded_term(costs, normalizer):
    require(
        costs.ndim == 2
        and costs.shape[1] == 3
        and not np.isnan(costs).any()
        and np.all(costs >= 0)
        and normalizer == NORMALIZER,
        "nonnegative per-view bundle costs required",
    )
    per_view = np.clip(costs / normalizer, 0.0, 1.0)
    per_view[np.isposinf(costs)] = np.inf
    return per_view, per_view.sum(axis=1)


def compare_bank(bank, method, view_observations, rig, scorer, configs):
    n = len(bank["candidate_index"])
    modes = {
        "baseline": dict(
            rank_costs(bank["adjusted_local_cost"].copy(), bank["parent_path_cost"]),
            sixth_term=bank["sixth_instance_term"].copy(),
        )
    }
    projections = {v: rig.camera(v).project(bank["candidate_pose_xyz"]) for v in VIEWS}
    for mode, config in configs.items():
        scores = {}
        for view in VIEWS:
            obs = view_observations[mode][view]
            scores[view] = scorer.score_partial_body_bundles(
                bank["candidate_pose_xyz"], rig.camera(view).project, obs
            )
        raw = np.stack([np.asarray(scores[v]["best_cost"], np.float64) for v in VIEWS], axis=1)
        require(raw.shape == (n, 3), "one score per candidate/view required")
        normalizer = NORMALIZER
        require(
            all(scores[v]["maximum_gain"] == config.maximum_gain for v in VIEWS)
            and np.all(raw[np.isfinite(raw)] <= config.maximum_gain + config.tie_atol),
            "scorer ceiling differs",
        )
        per_view, term = bounded_term(raw, normalizer)
        modes[mode] = dict(
            rank_costs(bank["base_local_cost"] + term, bank["parent_path_cost"]),
            sixth_term=term,
            raw_costs_by_view=raw,
            normalized_costs_by_view=per_view,
            normalizer=normalizer,
            weight_per_view=1.0,
            scorer_ceiling=config.maximum_gain,
            valid_cost_ceiling_per_view=config.maximum_gain / normalizer,
            invalid_projection_candidates=np.isposinf(raw).any(axis=1),
            positive_infinity_preserved=True,
            config=asdict(config),
            scores=scores,
        )
    return dict(
        schema=STAGE + "_case",
        frame_idx=bank["frame_idx"],
        baseline_bank=copy.deepcopy(bank),
        projections=projections,
        modes=modes,
        final_selected_match=exact_final_matches(bank, method),
        parent_history_fixed=True,
        applies_only_to_saved_fit_candidates=True,
        reoptimized_fit=False,
        new_temporal_backtrace=False,
        physical_ID_truth=False,
    )


def write_case(out, name, value):
    tree, arrays = codec.split_diagnostic(value)
    archive = out.write(name + ".npz", arrays)
    with np.load(archive["path"], allow_pickle=False) as data:
        codec.restore_diagnostic(tree, {k: data[k] for k in data.files})
    out.write(name + ".json", dict(archive=archive, tree=tree))
    return archive


def small_json(value):
    """Explicit arrays for small reports; nonfinite scientific values use the codec."""
    if isinstance(value, np.ndarray):
        return small_json(value.tolist())
    if isinstance(value, np.generic):
        return small_json(value.item())
    if isinstance(value, dict):
        return {str(k): small_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [small_json(v) for v in value]
    require(
        not isinstance(value, float) or np.isfinite(value),
        "nonfinite small JSON requires explicit descriptor",
    )
    return value


def render_selection(result):
    bank = result["baseline_bank"]
    selected = {}
    for mode, score in result["modes"].items():
        index = int(score["path_order"][0])
        selected[mode] = dict(
            candidate_index=index,
            path_minimum_indices=score["path_minimum_indices"].tolist(),
            local_minimum_indices=score["local_minimum_indices"].tolist(),
            best_local_cost=score["best_local_cost"],
            best_path_cost=score["best_path_cost"],
            path_cost=float(score["path_cost"][index]),
            local_cost=float(score["local_cost"][index]),
            parent_path_cost=float(bank["parent_path_cost"][index]),
            candidate_parent=int(bank["candidate_parent"][index]),
            candidate_kind=bank["candidate_kind"][index],
            candidate_plan=bank["candidate_plan"][index].tolist(),
            pose_xyz=bank["candidate_pose_xyz"][index].tolist(),
            projections={v: prior.finite_json(p[index]) for v, p in result["projections"].items()},
            support_not_recomputed=True,
            plan_is_assigned_views_not_direct_support=True,
        )
    return dict(
        arm=result["arm"],
        frame_idx=result["frame_idx"],
        selections=selected,
        final_selected_match=result["final_selected_match"],
        display_tie_policy="first index in stable path order; complete exact minimum set retained",
    )


def run(output):
    output = Path(output).resolve()
    require(output.is_relative_to(WORK) and not output.exists(), "fresh workspace output required")
    require(
        "torch" not in sys.modules or not sys.modules["torch"].cuda.is_initialized(), "CPU only"
    )
    out, evidence, started = proof.Output(output), proof.Evidence(), proof.utc()
    try:
        context = load_headers(evidence)
        code = source_code(context, evidence)
        scorer = importlib.import_module("partial_body_bundle_score")
        configs = dict(
            row_only=replace(scorer.DEFAULT_CONFIG, mask_weight=0.0),
            row_plus_mask=scorer.DEFAULT_CONFIG,
        )
        banks, references = load_banks(context, evidence)
        masks, points, captures = load_observations(context)
        rig = load_rig(context, evidence)
        before = {key: codec.split_diagnostic(value)[0] for key, value in banks.items()}
        plan = dict(
            schema=STAGE,
            frames=list(FRAMES),
            negative_control_frames=list(NEGATIVE_FRAMES),
            cases=case_keys(),
            arms=list(ARMS),
            modes=list(MODES),
            source_code=code,
            terminal_horizon=116200,
            visual_seal_sha256=VISUAL_SHA,
            visual_evidence_sha256=EVIDENCE_SHA,
            candidate_bank_references=references,
            tracking_sha256=TRACKING_SHAS,
            configs={k: asdict(v) for k, v in configs.items()},
            replacement=(
                "parent_path_cost + base_local_cost + sum_views(clip(best_cost / normalizer,0,1))"
            ),
            normalizers=dict(row_only=15.0, row_plus_mask=15.0),
            scorer_ceilings=dict(row_only=12.0, row_plus_mask=15.0),
            weight_per_view=1.0,
            invalid_positive_infinite_view_cost_preserved=True,
            invalid_projection_candidates_cannot_win=True,
            all_invalid_candidate_bank_fails_explicitly=True,
            fixed_parent_history=True,
            old_sixth_replaced_not_added=True,
            cross_arm_absolute_path_costs_comparable=False,
            negative_controls_included=True,
            negative_visual_sha256=NEGATIVE_SHA,
            negative_evidence_sha256=NEGATIVE_EVIDENCE_SHA,
            selected_frames_are_holdout=False,
            raw_video_reads=0,
            original_SLEAP_cache_reads=0,
            original_candidate_reads=0,
            checkpoint_or_DAG_reads=0,
            fit_calls=0,
            tracker_calls=0,
            ID_solver_calls=0,
            GPU_used=False,
            physical_ID_truth=False,
            whole_trajectory_or_swap_success_claimed=False,
        )
        out.write("plan.json", plan)
        summaries, render_rows = [], []
        for frame in ALL_FRAMES:
            prepared = {
                mode: {
                    v: scorer.prepare_partial_bundle_observation(
                        masks[frame][v], points[frame][v], config
                    )
                    for v in VIEWS
                }
                for mode, config in configs.items()
            }
            for arm in ARMS if frame in FRAMES else ARMS[:1]:
                result = compare_bank(
                    banks[(arm, frame)],
                    context["records"][frame]["methods"].get(arm),
                    prepared,
                    rig,
                    scorer,
                    configs,
                )
                result.update(arm=arm, bank_reference=references[f"{arm}/{frame}"])
                render_rows.append(render_selection(result))
                archive = write_case(out, f"cases/{arm}_{frame}", result)
                summaries.append(
                    dict(
                        arm=arm,
                        frame_idx=frame,
                        candidates=len(banks[(arm, frame)]["candidate_index"]),
                        archive=archive,
                        final_selected_match=result["final_selected_match"],
                        rankings={
                            mode: {
                                k: v
                                for k, v in data.items()
                                if k
                                in (
                                    "local_minimum_indices",
                                    "path_minimum_indices",
                                    "local_order",
                                    "path_order",
                                    "best_local_cost",
                                    "best_path_cost",
                                )
                            }
                            for mode, data in result["modes"].items()
                        },
                    )
                )
        require(
            before == {key: codec.split_diagnostic(value)[0] for key, value in banks.items()},
            "original banks changed",
        )
        for group in captures.values():
            for record in group.values():
                require(
                    frozen.observations.file_stat(Path(record["path"])) == record["file_stat"],
                    "mask archive changed",
                )
        out.write(
            "render_payload.json",
            small_json(
                dict(
                    schema=STAGE + "_render_payload_v1",
                    modes=list(MODES),
                    cases=render_rows,
                    original_observation_evidence=[
                        dict(
                            root=str(VISUAL_ROOT),
                            seal_sha256=VISUAL_SHA,
                            evidence_sha256=EVIDENCE_SHA,
                            frames=list(FRAMES),
                        ),
                        dict(
                            root=str(NEGATIVE_ROOT),
                            seal_sha256=NEGATIVE_SHA,
                            evidence_sha256=NEGATIVE_EVIDENCE_SHA,
                            frames=list(NEGATIVE_FRAMES),
                        ),
                    ],
                    physical_ID_truth=False,
                    raw_slot_labels_only=True,
                    new_trajectory_created=False,
                )
            ),
        )
        out.write(
            "summary.json",
            small_json(
                dict(
                    schema=STAGE,
                    status="ok",
                    cases=summaries,
                    captured_mask_records=captures,
                    saved_candidate_banks=11,
                    target_mask_decodes=21,
                    mask_archive_captures=6,
                    baseline_bank_unchanged=True,
                    physical_ID_truth=False,
                    new_trajectory_created=False,
                    limitations=[
                        "Fixed original parent histories and already fitted candidate pools only.",
                        "Observation-based scores are not human identity ground truth.",
                        "Minimum-cost sets and exact final-pose matches may be ambiguous.",
                        "No original SLEAP, video, candidate21, checkpoint39 "
                        "or ID mapping is decoded.",
                    ],
                )
            ),
        )
        evidence.final_guard()
        out.final_guard()
        require(
            "torch" not in sys.modules or not sys.modules["torch"].cuda.is_initialized(),
            "CUDA initialized",
        )
        seal = dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=out.records,
        )
    except BaseException as error:
        seal = dict(
            stage=STAGE,
            status="failed",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=out.records,
            error=repr(error),
        )
        (output / "comparison.provenance.json").write_bytes(proof.json_bytes(seal))
        raise
    (output / "comparison.provenance.json").write_bytes(proof.json_bytes(seal))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)


if __name__ == "__main__":
    main()
