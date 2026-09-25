"""Completed same76360 three-way comparison; CPU validation, never tracking.

The old two all9960 optical audits and seven C34 rows are reused by exact seals,
candidate bytes and SL/calibration bindings. Only bounded support is reprojected.
All three terminal39 checkpoints are semantically decoded without restore or
advance; all21 saved winners are independently backtraced. Sparse female marks
are reused evaluation data, not new or independent three-animal identity truth.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "labs/20260909_three_body"), str(REPO / "src")]

import audit_bounded_first_chunk as interim
import compare_directional_one_minute as prior
import numpy as np
from continuous_tracker import AcceleratedContinuousPersistentNodeTracker
from directional_chest_tracker import DirectionalChestContinuousTracker, array_signature

from medaka3d.eval.generative import MouthAnchor
from medaka3d.provenance import provenance

checks = prior.checks
FIRST, START, STOP, CHUNK = 49800, 66400, 76360, 1660
STAGE = "bounded_one_minute_comparison_v1"
METHODS = ("original", "directional", "bounded")
NODES = ("mouth", "chest", "tail")
WORK = REPO / "work/three_body"
PARENT = WORK / "long_continuation_through66400_v1"
ORIGINAL = WORK / "long_remaining_supervisor_v1/phases/f066400-076360/tracking"
DIRECTIONAL = WORK / "directional_one_minute_66400-76360_v1"
OLD_COMPARISON = WORK / "directional_one_minute_comparison_v1"
CLASSICAL = WORK / "directional_minute_all7_classical_comparison_v1"
FEMALE = WORK / "long_female_endpoint_160s_v1"
PRIOR_SHA = "a30db73faeb57d86bc5f2211bec125c269fe6181d922260ec60e20c46f828333"
CLASSICAL_SHA = "c3b58e087d4b2961d2f710ad5884a4417a93b329233150186b04a99e81c0126d"
FEMALE_SHA = "e5894da372b2f1b577e8c0ac2d6930140231a6674db90594501dda58d1fba5b6"
ANCHOR_FRAMES = (66889, 68186, 69363, 70234, 70835, 71288, 72371, 73027, 73953, 74786, 75987)
LAMBDAS = ("0", "0.1", "0.25", "0.5", "0.767", "2", "8")
PINS = {
    Path(
        prior.__file__
    ).resolve(): "066f702d0c64ddf048fe5d9c65e6add64f92daf40025e90b50a698355e6707c3",
    Path(
        interim.__file__
    ).resolve(): "7be951a77f39e8f2dea5e6cf2aeae1a10663036ede0e131a1462beb67c45759e",
    WORK / "run_bounded_mouth_one_minute.py": interim.RUNNER_SHA,
}


def bind(path, digest, hashes):
    return interim.bind(path, digest, hashes)


def input_binding(record, path, digest):
    path = Path(path).resolve()
    items = [v for v in record["inputs"].values() if Path(v["path"]).resolve() == path]
    checks.require(items and all(v["sha256"] == digest for v in items), "input binding differs")


def successful(root, filename, stage, hashes, expected=None):
    previous = dict(hashes)
    if expected is not None:
        bind(root / filename, expected, hashes)
    record = checks.verify(root, filename, stage, hashes)
    checks.require(
        all(hashes[path] == digest for path, digest in previous.items()),
        "conflicting evidence for previously bound file",
    )
    checks.require(bool(record.get("finished_at")), "stage lacks completion time")
    return record


def validate_completion(summary):
    checks.require(
        summary.get("schema") == "bounded_mouth_one_minute_v1"
        and summary.get("status") == "ok"
        and summary.get("one_minute_completed") is True
        and summary.get("native_interval") == [START, STOP]
        and summary.get("native_frames") == STOP - START
        and summary.get("logical_initialization_count") == 1
        and summary.get("restore_count") == 4
        and summary.get("external_pose_resets") == 0
        and summary.get("checkpoint_count") == 6
        and summary.get("diagnostic_frames") == 321
        and summary.get("periodic_checkpoints_are_not_reinitializations") is True,
        "completed bounded minute/lifecycle required",
    )


def semantic_decode(cls, arrays, metadata, rig, seed):
    """Same frozen decoder as interim; full endpoint is fixed to76360 here."""
    config = copy.deepcopy(metadata["context"]["tracker_config"])
    checks.require(config["device"] == "cuda", "saved CUDA context required")
    config["device"] = "cpu"
    tracker = cls(
        rig,
        seed,
        interim.TrackerConfig(**config),
        recovery_config=interim.NodeRecoveryConfig(**metadata["context"]["recovery_config"]),
        persistence_config=interim.PersistentNodeConfig(
            **metadata["context"]["persistence_config"]
        ),
    )
    actual = tracker._context()
    claimed = copy.deepcopy(actual)
    claimed["tracker_config"]["device"] = "cuda"
    checks.require(claimed == metadata["context"], "context differs beyond validation device")
    adapted = copy.deepcopy(metadata)
    adapted["context"] = actual
    before = array_signature(arrays)
    history, states, previous, _ = tracker._decode_state(arrays, adapted)
    checks.require(
        array_signature(arrays) == before
        and tracker.initialize_count == tracker.processed_frames == 0,
        "semantic validation mutated inputs or started tracker",
    )
    checks.require(
        len(arrays) == 39
        and len(history) == STOP - FIRST
        and len(states) == 6
        and previous.frame_idx == STOP - 1,
        "decoded full endpoint or six active states differs",
    )
    return dict(
        array_count=39,
        history_frames=len(history),
        history_rows=sum(map(len, history)),
        active_states=6,
        previous_frame=previous.frame_idx,
        frozen_semantic_decoder_pass=True,
        all_history_ages_review_velocity_cost_checked=True,
        validation_context_device="cpu",
        saved_context_device="cuda",
        restore_calls=0,
        advance_calls=0,
        limit="Unretained candidate pools/optimizer and score margins are not replayed.",
    )


def audit_bounded(branch, binding, parent, parent_meta, hashes):
    prior.require_bound(
        branch,
        ("summary.json", "checkpoint_progress.json", "final_checkpoint.json", "state_import.json"),
        hashes,
    )
    summary = checks.read(branch / "summary.json")
    validate_completion(summary)
    migration = checks.read(branch / "state_import.json")
    rows = checks.read(branch / "checkpoint_progress.json")
    checks.require(len(rows) == 6, "six checkpoint records required")
    previous, audits, fixed = parent, [], None
    for row, stop in zip(rows, range(START + CHUNK, STOP + CHUNK, CHUNK), strict=True):
        path = branch / "checkpoints" / f"through_{stop:06d}"
        checks.require(
            Path(row["path"]).resolve() == path
            and hashes.get(str(path / "manifest.json")) == row["manifest_sha256"],
            "recorded checkpoint manifest/path differs",
        )
        current = checks.read(path / "manifest.json")["bindings"]
        checks.require(
            current["schema"] == "bounded_mouth_one_minute_checkpoint_binding_v1"
            and current["original_binding"] == binding
            and current["native_interval"] == [START, STOP]
            and current["state_import_sha256"] == hashes[str(branch / "state_import.json")]
            and current["experimental_code_sha256"].get(
                str(WORK / "run_bounded_mouth_one_minute.py")
            )
            == interim.RUNNER_SHA,
            "bounded checkpoint scientific/source/parent binding differs",
        )
        if fixed is None:
            fixed = current
            for path_name, digest in current["experimental_code_sha256"].items():
                source = bind(path_name, digest, hashes)
                checks.require(source.is_relative_to(REPO), "source outside workspace")
                snap = branch / "code_snapshot" / source.relative_to(REPO)
                checks.require(hashes.get(str(snap)) == digest, "unsealed code snapshot")
        checks.require(current == fixed, "checkpoint binding drift")
        arrays, meta, _ = prior.load_bound_checkpoint(path, current, hashes)
        prior.validate_lifecycle(meta, stop, 4)
        interim.verify_bounded_import(migration, parent, parent_meta, meta)
        checks.require(
            row["arrays"] == 39
            and all(
                row[k] == meta[k]
                for k in ("previous_frame", "processed_frames", "initialize_count", "restore_count")
            ),
            "checkpoint progress metadata differs",
        )
        prior.verify_history_prefix(previous, arrays)
        audits.append(
            dict(
                through_frame=stop,
                manifest_sha256=row["manifest_sha256"],
                all_retained_history_prefix_exact=True,
                restore_count=4,
            )
        )
        previous = arrays
    checks.require(
        checks.read(branch / "final_checkpoint.json") == rows[-1], "final pointer differs"
    )
    return prior.load_candidate(branch, arrays, hashes), arrays, meta, audits, summary


def reused_support(candidate, method, saved):
    part = {k: candidate[k][START - FIRST :] for k in ("frame_idx", "sources", "observed")}
    result = saved["methods"][method]
    check = result["all_native_source_geometry"]
    checks.require(
        prior.short.stats(candidate, START, STOP) == result["full_minute"],
        "reused audit candidate counts/cost differ",
    )
    checks.require(
        check["native_frames"] == STOP - START
        and check["frame_range"] == [START, STOP]
        and check["blocks"] == 6
        and check["block_frames"] == CHUNK
        and check["ray_gate_mm"] == 2.0
        and check["native_pixel_gate"] == 12.0
        and check["all_sources_and_observed_flags_pass"] is True
        and check["temporal_subsampling"] is False
        and check["source_rows_checked"] == int((part["sources"] >= 0).sum())
        and check["observed_nodes_checked"] == int(part["observed"].sum()),
        "reused all9960 source support audit differs",
    )
    return copy.deepcopy(check)


def node_metrics(candidate, start=START, stop=STOP):
    """Missing spans and endpoint averages; adjacent steps never cross a gap."""
    frames, observed, pose = (candidate[k] for k in ("frame_idx", "observed", "pose_xyz"))
    checks.require(np.all(np.diff(frames) == 1), "native contiguous frames required")
    keep = (frames >= start) & (frames < stop)
    checks.require(int(keep.sum()) == stop - start, "complete evaluation interval required")
    runs, events, rows = [], [], []
    for fish in range(3):
        for node, label in enumerate(NODES):
            last, missing = None, None
            for t, native in enumerate(frames):
                native = int(native)
                if not observed[t, fish, node]:
                    if missing is None:
                        missing = native
                    continue
                if missing is not None:
                    if max(missing, start) < min(native, stop):
                        runs.append(
                            dict(
                                slot=fish,
                                node=label,
                                start=missing,
                                stop=native,
                                missing_frames_in_interval=min(native, stop) - max(missing, start),
                                terminal_open=False,
                            )
                        )
                    if start <= native < stop:
                        elapsed = None if last is None else native - int(frames[last])
                        distance = (
                            None
                            if last is None
                            else float(np.linalg.norm(pose[t, fish, node] - pose[last, fish, node]))
                        )
                        events.append(
                            dict(
                                frame_idx=native,
                                slot=fish,
                                node=label,
                                missing_frames=native - missing,
                                previous_direct_frame=None if last is None else int(frames[last]),
                                previous_latent_to_reacquired_step_mm=None
                                if t == 0
                                else float(
                                    np.linalg.norm(pose[t, fish, node] - pose[t - 1, fish, node])
                                ),
                                direct_endpoint_distance_mm=distance,
                                direct_endpoint_elapsed_frames=elapsed,
                                direct_endpoint_average_mm_per_s=None
                                if elapsed is None
                                else distance * 166 / elapsed,
                                identity_review=bool(candidate["identity_review"][t, fish]),
                                biological_identity_confirmed=False,
                            )
                        )
                    missing = None
                last = t
            if missing is not None and max(missing, start) < stop:
                runs.append(
                    dict(
                        slot=fish,
                        node=label,
                        start=missing,
                        stop=int(frames[-1]) + 1,
                        missing_frames_in_interval=stop - max(missing, start),
                        terminal_open=True,
                    )
                )
            pair = observed[1:, fish, node] & observed[:-1, fish, node] & keep[1:] & keep[:-1]
            delta = np.linalg.norm(np.diff(pose[:, fish, node], axis=0), axis=-1)
            direct, latent = delta[pair], delta[keep[1:]]
            matching = [r for r in runs if r["slot"] == fish and r["node"] == label]
            longest = max((r["missing_frames_in_interval"] for r in matching), default=0)
            rows.append(
                dict(
                    slot=fish,
                    node=label,
                    supported_frames=int(observed[keep, fish, node].sum()),
                    missing_frames=int((~observed[keep, fish, node]).sum()),
                    longest_missing_frames_in_interval=longest,
                    longest_missing_runs=[
                        r for r in matching if r["missing_frames_in_interval"] == longest
                    ],
                    directly_supported_adjacent_steps=int(pair.sum()),
                    directly_supported_adjacent_steps_gt10mm=int((direct > 10).sum()),
                    maximum_directly_supported_adjacent_step_mm=float(direct.max())
                    if direct.size
                    else None,
                    maximum_latent_adjacent_step_mm=float(latent.max()) if latent.size else None,
                )
            )
    return dict(
        by_slot_node=rows,
        missing_runs=runs,
        reacquisitions=events,
        direct_step_scope="Both endpoints inside interval, adjacent native and directly supported.",
        latent_step_scope=(
            "Ending frame inside interval; includes the start-minus-one boundary and missing "
            "predictions. Not observed fish speed."
        ),
    )


def support_delta(a, b, start=START, stop=STOP):
    checks.require(prior.same(a["frame_idx"], b["frame_idx"]), "different horizons")
    keep = (a["frame_idx"] >= start) & (a["frame_idx"] < stop)
    x, y = a["observed"][keep], b["observed"][keep]
    gain, loss = y & ~x, x & ~y
    changes = []
    for t, fish, node in np.argwhere(gain | loss):
        changes.append(
            dict(
                frame_idx=int(a["frame_idx"][keep][t]),
                slot=int(fish),
                node=NODES[node],
                change="gain" if gain[t, fish, node] else "loss",
            )
        )
    return dict(
        gained_by_slot_node=gain.sum(0).tolist(),
        lost_by_slot_node=loss.sum(0).tolist(),
        gained_by_node=gain.sum((0, 1)).tolist(),
        lost_by_node=loss.sum((0, 1)).tolist(),
        review_changed_by_slot=(a["identity_review"][keep] != b["identity_review"][keep])
        .sum(0)
        .tolist(),
        slot_scope="Same numeric slots only; no biological identity alignment",
    ), changes


def classical_rows(summary, traces, old_summary):
    checks.require(
        summary["schema"] == "directional_minute_all7_classical_comparison_v1"
        and summary["status"] == "ok"
        and summary["native_interval"] == [START, STOP]
        and summary["native_frames"] == STOP - START
        and summary["requested_female_anchors"] == 11
        and summary["requested_female_intervals"] == 10
        and summary["all_seven_lambdas_retained"] is True,
        "classical scope differs",
    )
    rows = summary["methods"]
    expected = {"original", "directional", *("c34_lambda_" + v for v in LAMBDAS)}
    checks.require(
        len(rows) == 9 and {r["name"] for r in rows} == expected, "all seven fixed C34s required"
    )
    for row in rows:
        checks.require(
            row["native_frames"] == STOP - START and row["fish_frame_denominator"] == 29880,
            "C34/forward denominator differs",
        )
        for scored in row["female"].values():
            checks.require(
                scored["requested_anchors"] == 11
                and scored["requested_intervals"] == 10
                and tuple(a["frame"] for a in scored["anchors"]) == ANCHOR_FRAMES,
                "female anchors dropped or changed",
            )
        if row["name"] in traces:
            name = row["name"]
            part = traces[name]["observed"][START - FIRST :]
            checks.require(
                row["observed_nodes_total"] == part.sum((0, 1)).tolist()
                and row["observed_nodes_by_slot"] == part.sum(0).tolist()
                and row["female"]["strict_raw"] == old_summary["methods"][name]["female"]["raw"]
                and row["female"]["strict_with_original_acceptance"]
                == old_summary["methods"][name]["female"]["with_original_review"],
                "classical forward row differs from completed minute",
            )
    return copy.deepcopy(rows)


def load_inputs(branch, expected_sha, hashes):
    # All three producer success gates precede any candidate/checkpoint payload.
    bounded_seal = successful(
        branch,
        "tracking.provenance.json",
        "bounded_mouth_one_minute_continuous",
        hashes,
        expected_sha,
    )
    prior.require_bound(branch, ("summary.json",), hashes)
    validate_completion(checks.read(branch / "summary.json"))
    prior_seal = successful(
        OLD_COMPARISON,
        "comparison.provenance.json",
        "directional_one_minute_comparison",
        hashes,
        PRIOR_SHA,
    )
    checks.require(
        prior_seal["params"]["all_bound_files_unchanged"] is True, "prior comparison guard missing"
    )
    prior.require_bound(OLD_COMPARISON, ("summary.json", "checkpoint_audit.json"), hashes)
    saved = checks.read(OLD_COMPARISON / "summary.json")
    checks.require(
        saved["schema"] == "directional_one_minute_comparison_v1"
        and saved["status"] == "ok"
        and saved["native_frames"] == 9960
        and saved["same_parent_and_horizon"] is True
        and saved["all21_original_and_directional_arrays_exact_backtrace"] is True
        and saved["original_history_and_all_six_checkpoint_prefixes_exact"] is True
        and saved["checkpoint_snapshots_and_earlier_manifest_digests_verified"] is True
        and saved["restores_during_one_minute"] == saved["external_pose_resets"] == 0,
        "prior exact-horizon audit incomplete",
    )
    producers = {}
    for name, root, filename, stage in (
        ("original", ORIGINAL, "continuation.provenance.json", "three_body_long_continuation"),
        (
            "directional",
            DIRECTIONAL,
            "tracking.provenance.json",
            "directional_one_minute_continuous",
        ),
    ):
        producers[name] = successful(root, filename, stage, hashes)
        for path in (root / filename, root / "candidate.npz"):
            input_binding(prior_seal, path, hashes[str(path)])
    binding = producers["original"]["params"]["bindings"]
    checks.require(
        producers["original"]["params"]["through_frame"] == STOP
        and Path(producers["original"]["params"]["resume_phase"]).resolve() == PARENT,
        "wrong original parent/horizon",
    )
    parent_seal = successful(
        PARENT, "continuation.provenance.json", "three_body_long_continuation", hashes
    )
    checks.require(parent_seal["params"]["bindings"] == binding, "parent binding differs")
    for record in (prior_seal, producers["directional"], bounded_seal):
        input_binding(
            record,
            PARENT / "continuation.provenance.json",
            hashes[str(PARENT / "continuation.provenance.json")],
        )
    for path, digest in binding["code_sha256"].items():
        bind(path, digest, hashes)
    parent, parent_meta, _ = prior.load_bound_checkpoint(PARENT / "checkpoint", binding, hashes)
    prior.validate_lifecycle(parent_meta, START, 3)
    traces, states, metas = {}, {}, {}
    for name, root in (("original", ORIGINAL), ("directional", DIRECTIONAL)):
        path = root / ("checkpoint" if name == "original" else "checkpoints/through_076360")
        fixed = checks.read(path / "manifest.json")["bindings"]
        if name == "original":
            checks.require(fixed == binding, "original checkpoint binding differs")
        else:
            checks.require(
                fixed["original_binding"] == binding and fixed["native_interval"] == [START, STOP],
                "directional binding differs",
            )
            migration = checks.read(root / "state_import.json")
            prior.verify_import(migration, parent, parent_meta)
            checks.require(
                fixed["state_import_sha256"] == hashes[str(root / "state_import.json")],
                "directional import SHA differs",
            )
            for source, digest in fixed["experimental_code_sha256"].items():
                source = bind(source, digest, hashes)
                checks.require(
                    hashes.get(str(root / "code_snapshot" / source.relative_to(REPO))) == digest,
                    "directional snapshot differs",
                )
        states[name], metas[name], _ = prior.load_bound_checkpoint(path, fixed, hashes)
        prior.validate_lifecycle(metas[name], STOP, 4)
        if name == "directional":
            checks.require(
                metas[name]["context"] == migration["new_context"], "directional context differs"
            )
        prior.verify_history_prefix(parent, states[name])
        traces[name] = prior.load_candidate(root, states[name], hashes)
    traces["bounded"], states["bounded"], metas["bounded"], audits, bounded_summary = audit_bounded(
        branch, binding, parent, parent_meta, hashes
    )
    checks.require(
        bounded_summary["direct_node_support"]
        == traces["bounded"]["observed"][START - FIRST :].sum((0, 1)).tolist()
        and bounded_summary["identity_review_fish_frames"]
        == int(traces["bounded"]["identity_review"][START - FIRST :].sum()),
        "bounded producer counts differ",
    )
    shared = binding["inputs"]["seed"]["SL_and_calibration_sha256"]
    checks.require(len(shared) == 7, "SL3/calibration4 required")
    for path, digest in shared.items():
        bind(path, digest, hashes)
        for record in (prior_seal, *producers.values(), bounded_seal):
            input_binding(record, path, digest)
    paths = {
        v: next(Path(p) for p in shared if Path(p).name == prior.EXPERIMENT + "__" + v + ".npz")
        for v in prior.VIEWS
    }
    calibration = [Path(p) for p in shared if Path(p).name == "calibration_manifest.json"]
    checks.require(
        len(set(paths.values())) == 3 and len(calibration) == 1, "SL/calibration paths differ"
    )
    rig, nodes = (
        prior.load_rig(calibration[0].parent),
        prior.RegisteredNodes(paths, np.arange(START, STOP)),
    )
    seed_info = binding["inputs"]["seed"]
    seed_path = bind(seed_info["registration_path"], seed_info["registration_sha256"], hashes)
    seed = np.asarray(checks.read(seed_path)["pose_xyz"], np.float64)
    decoded = {
        name: semantic_decode(cls, states[name], metas[name], rig, seed)
        for name, cls in (
            ("original", AcceleratedContinuousPersistentNodeTracker),
            ("directional", DirectionalChestContinuousTracker),
            ("bounded", interim.bounded.BoundedMouthContinuousTracker),
        )
    }
    return traces, decoded, nodes, rig, audits, saved, prior_seal, shared


def reused_evaluation(traces, old_summary, prior_seal, shared, hashes):
    female_seal = successful(
        FEMALE,
        "evaluation.provenance.json",
        "long_female_endpoint_evaluation_160s",
        hashes,
        FEMALE_SHA,
    )
    input_binding(prior_seal, FEMALE / "evaluation.provenance.json", FEMALE_SHA)
    prior.require_bound(FEMALE, ("anchor_matches.json", "original_female_sex_gt.csv"), hashes)
    checks.require(
        hashes[str(FEMALE / "original_female_sex_gt.csv")] == prior.GT_SHA,
        "annotation version differs",
    )
    raw = [r for r in checks.read(FEMALE / "anchor_matches.json") if START <= r["frame"] < STOP]
    checks.require(tuple(r["frame"] for r in raw) == ANCHOR_FRAMES, "fixed11 anchor list differs")
    anchors = [MouthAnchor(**r) for r in raw]
    classical_seal = successful(
        CLASSICAL,
        "comparison.provenance.json",
        "directional_minute_all7_classical_comparison_v1",
        hashes,
        CLASSICAL_SHA,
    )
    input_binding(classical_seal, OLD_COMPARISON / "comparison.provenance.json", PRIOR_SHA)
    input_binding(classical_seal, FEMALE / "evaluation.provenance.json", FEMALE_SHA)
    prior.require_bound(CLASSICAL, ("summary.json",), hashes)
    summary = checks.read(CLASSICAL / "summary.json")
    for path, digest in shared.items():
        input_binding(female_seal, path, digest)
        input_binding(classical_seal, path, digest)
    for name, root in (("original", ORIGINAL), ("directional", DIRECTIONAL)):
        input_binding(classical_seal, root / "candidate.npz", hashes[str(root / "candidate.npz")])
        checks.require(
            prior.female_scores(traces[name], anchors) == old_summary["methods"][name]["female"],
            "saved anchor scoring differs",
        )
    rows = classical_rows(summary, traces, old_summary)
    return anchors, rows


def main(args):
    checks.require(
        type(args.owned_session) is int
        and args.owned_session > 0
        and type(args.owned_exit_code) is int
        and args.owned_exit_code == 0,
        "explicit ROOT-observed owned session CLOSED0 required",
    )
    branch, output = args.branch.resolve(strict=True), args.output.resolve()
    checks.require(output.is_relative_to(WORK) and not output.exists(), "new work output required")
    code = prior.source_registry.local_code()
    hashes, outputs = dict(code), {}
    for path, digest in PINS.items():
        bind(path, digest, hashes)
    output.mkdir(parents=True)
    with provenance(STAGE, output, name="comparison") as record:

        def emit(name, value=None):
            path = output / name
            if value is not None:
                path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
            outputs[str(path)] = checks.sha(path)
            record.output(name, path, sha256=outputs[str(path)])

        for path, digest in code.items():
            target = output / "code_snapshot" / Path(path).relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(path).read_bytes())
            checks.require(checks.sha(target) == digest, "snapshot changed")
            emit(str(target.relative_to(output)))
        traces, decoded, nodes, rig, audits, old_summary, prior_seal, shared = load_inputs(
            branch, args.expected_seal_sha256, hashes
        )
        anchors, classical = reused_evaluation(traces, old_summary, prior_seal, shared, hashes)
        methods = {}
        for name in METHODS:
            candidate = traces[name]
            part = {
                k: candidate[k][START - FIRST :]
                for k in ("frame_idx", "pose_xyz", "sources", "observed")
            }
            if name == "bounded":

                def save_support(meta, arrays):
                    a, b = meta["frame_range"]
                    filename = f"bounded_support_f{a:06d}-{b:06d}.npz"
                    np.savez_compressed(output / filename, **arrays)
                    emit(filename)

                geometry = prior.support_audit.verify_support_partitioned(
                    part, nodes, rig, on_block=save_support
                )
            else:
                geometry = reused_support(candidate, name, old_summary)
            review = candidate["identity_review"][START - FIRST :]
            methods[name] = dict(
                full_minute=prior.short.stats(candidate, START, STOP),
                direct_by_slot_node=part["observed"].sum(0).tolist(),
                accepted_by_slot_node=(part["observed"] & ~review[:, :, None]).sum(0).tolist(),
                review_by_slot=review.sum(0).tolist(),
                selected_kind_counts=dict(
                    Counter(candidate["proposal_kind"][START - FIRST :].tolist())
                ),
                all_native_source_geometry=geometry,
                support_recomputed_here=name == "bounded",
                support_reused_comparison_sha256=PRIOR_SHA if name != "bounded" else None,
                female=prior.female_scores(candidate, anchors),
            )
            metrics = node_metrics(candidate)
            methods[name]["node_metrics"] = metrics["by_slot_node"]
            emit(f"{name}_gaps_and_reacquisitions.json", metrics)
        deltas = {}
        for a, b in combinations(METHODS, 2):
            key = a + "_to_" + b
            deltas[key], changes = support_delta(traces[a], traces[b])
            emit(key + "_gain_loss.json", changes)
        candidate_paths = {
            name: str(root / "candidate.npz")
            for name, root in (
                ("original", ORIGINAL),
                ("directional", DIRECTIONAL),
                ("bounded", branch),
            )
        }
        result = dict(
            schema=STAGE,
            status="ok",
            native_interval=[START, STOP],
            native_frames=9960,
            horizon_stop=STOP,
            same_parent_and_horizon=True,
            one_minute_completed=True,
            candidate_paths=candidate_paths,
            candidate_sha256={name: hashes[path] for name, path in candidate_paths.items()},
            all39_arrays_semantically_decoded=True,
            all21_arrays_exact_backtrace=True,
            original66400_parent_and_migration_verified=True,
            bounded_all_six_checkpoint_prefixes_exact=True,
            logical_initialization_count=1,
            checkpoint_restore_count=4,
            restores_during_one_minute=0,
            external_pose_resets=0,
            owned_process_completion=dict(
                session=args.owned_session,
                exit_code=0,
                scope="ROOT-observed owned tool exit; not an OS witness created by this comparator",
            ),
            methods=methods,
            support_delta=deltas,
            checkpoint_validation=decoded,
            shared_SL_calibration_sha256=shared,
            reused_original_directional_comparison_sha256=PRIOR_SHA,
            reused_classical_comparison_sha256=CLASSICAL_SHA,
            classical_rows=classical,
            all_seven_C34_conditions_retained=True,
            requested_female_anchors=11,
            requested_female_intervals=10,
            independent_ID_truth=False,
            all_three_ID_truth=False,
            adopted=False,
            full_thirty_minute_comparison=False,
            raw_pixel_score_included=False,
            limitations=[
                "Fullminute raw pixel evaluation is a separate artifact and task.",
                "All slots are numeric state slots, not cross-method biological identity matches.",
                "Sticky review is inherited, never cleared by geometric recovery.",
                "Sparse female P/S/U cannot certify dense or male identities "
                "or absence of return swaps.",
                "C34 uses whole-recording future information and different initialization; "
                "saved chest unchanged, mouth/tail derived.",
                "Old optical audits, C34 rows and annotation matching are reused; "
                "no repeated old support/SL matching/raw decoding. Saved female anchors "
                "are scored against all three candidates here.",
                "No fit, advance, restore or GPU calls. Decoder verifies retained state, "
                "not dropped candidate pools.",
            ],
        )
        emit("checkpoint_audit.json", audits)
        emit("summary.json", result)
        frame_rows = []
        for t in range(START - FIRST, STOP - FIRST):
            row = dict(frame_idx=int(traces["bounded"]["frame_idx"][t]))
            for name, data in traces.items():
                for node, label in enumerate(NODES):
                    row[name + "_" + label + "_supported"] = int(data["observed"][t, :, node].sum())
                row[name + "_review_fish"] = int(data["identity_review"][t].sum())
                row[name + "_selected_kind"] = str(data["proposal_kind"][t])
            frame_rows.append(row)
        with (output / "frames.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(frame_rows[0]))
            writer.writeheader()
            writer.writerows(frame_rows)
        emit("frames.csv")
        lines = [
            "# 同76360終端・3方式の1分比較",
            "",
            "[66400,76360) の9960 native frames、各nodeの分母は29,880魚フレーム。",
            "",
            "|方式|口/胸/尾の直接支持|全3胸部支持frame|review魚frame|",
            "|---|---|---:|---:|",
        ]
        for name in METHODS:
            value = methods[name]["full_minute"]
            lines.append(
                f"|{name}|{value['direct_node_support']}|{value['all_three_chest_frames']}|{value['identity_review_fish_frames']}|"
            )
        lines += [
            "",
            "3終端39配列の意味検査と21 backtraceを確認。新分岐の6 checkpointで"
            "共通親履歴・移行・初期化1/restore4/区間内restore0を確認。",
            "旧2方式の全9960支持監査は同候補SHAで再利用し、bounded全9960だけ2mm/12pxで再投影した。全slot/nodeの増減・欠測・再取得は別JSONに全件保存。",
            "連続直接支持stepと欠測後の端点平均を区別する。幾何回復でsticky reviewを解除しない。",
            "",
            "既存全7 C34行と11女性端点/10区間を保持。C34の保存胸部は変更せず、口尾は派生診断。"
            "全動画の将来情報・初期化が異なるため因果的対照ではない。",
            "全3ID正解・30分完了・方式採用は認定しない。新しい1分raw pixel評価は別工程。",
            "",
        ]
        (output / "RESULT.md").write_text("\n".join(lines), encoding="utf-8")
        emit("RESULT.md")
        checks.require(code == prior.source_registry.local_code(), "loaded code set changed")
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
        for path, digest in {**hashes, **outputs}.items():
            checks.require(checks.sha(path) == digest, "bound source/input/output changed")
        record.param("all_bound_source_input_snapshot_output_SHA_unchanged", True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", type=Path, required=True)
    parser.add_argument("--expected-seal-sha256", required=True)
    parser.add_argument("--owned-session", type=int, required=True)
    parser.add_argument("--owned-exit-code", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(main(parser.parse_args()), indent=2))
