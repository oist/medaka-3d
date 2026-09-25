"""Committed68060 same-horizon CPU audit, while the new minute may continue.

No unfinished new candidate/checkpoint, fitter, GPU, raw video or truth file is
opened. Completed old-run output bytes are hashed, but later checkpoint/candidate
arrays are not deserialized. The mutable commit-list uses its stable first record.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_REPO / "labs/20260909_three_body"), str(_REPO / "src")]

import bounded_mouth_tracker as bounded
import compare_directional_one_minute as comparison
import numpy as np
from directional_chest_tracker import DirectionalChestContinuousTracker, array_signature
from long_baseline_support import EXPERIMENT, RegisteredNodes
from node_recovery import NodeRecoveryConfig
from persistent_node_tracker import PersistentNodeConfig
from tracker import TrackerConfig

from medaka3d.calib import load_rig
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS

checks, REPO = comparison.checks, comparison.REPO
FIRST, START, STOP = 49800, 66400, 68060
STAGE = "bounded_mouth_first_chunk_interim_audit_v1"
OLD = REPO / "work/three_body/directional_one_minute_66400-76360_v1"
PARENT = REPO / "work/three_body/long_continuation_through66400_v1"
OLD_SEAL_SHA = "020ec9bbb078739dcad9d841e06fe95a90352ba992e71bb3e6a4f67c3926c614"
OLD_MANIFEST_SHA = "2f1d0122254c1c1c3605cf6010dcf39f82c82c214b616c2bc5bc2fe11cb9461b"
RUNNER_SHA = "b817b19fa451d47a914c34ba2715ee6efb4be9c1a39bba68ee771788ef7e5e72"
NODES = ("mouth", "pectoral", "tail")


def bind(path, digest, hashes):
    path = Path(path).resolve(strict=True)
    checks.require(checks.sha(path) == digest, "bound file changed: " + str(path))
    checks.require(str(path) not in hashes or hashes[str(path)] == digest, "conflicting bound SHA")
    hashes[str(path)] = digest
    return path


def input_member(seal, path, hashes):
    path = Path(path).resolve(strict=True)
    item = seal["inputs"].get(str(path))
    checks.require(item is not None and Path(item["path"]).resolve() == path, "unbound input path")
    return bind(path, item["sha256"], hashes)


def committed(branch, expected_sha, hashes):
    checks.require(
        isinstance(expected_sha, str)
        and len(expected_sha) == 64
        and all(c in "0123456789abcdef" for c in expected_sha),
        "expected SHA required",
    )
    rows = checks.read(branch / "checkpoint_progress.json")
    checks.require(isinstance(rows, list) and len(rows) >= 1, "first checkpoint not committed")
    row = rows[0]
    root = branch / "checkpoints/through_068060"
    checks.require(
        Path(row["path"]).resolve() == root and row["manifest_sha256"] == expected_sha,
        "first checkpoint path/digest differs",
    )
    bind(root / "manifest.json", expected_sha, hashes)
    # The codec manifest below verifies and guards these two payloads independently.
    for name in ("metadata.json", "arrays.npz"):
        path = root / name
        bind(path, checks.sha(path), hashes)
    fixed = checks.read(root / "manifest.json")["bindings"]
    checks.require(
        fixed["schema"] == "bounded_mouth_one_minute_checkpoint_binding_v1"
        and fixed["native_interval"] == [START, comparison.STOP],
        "branch binding scope",
    )
    runner = REPO / "work/three_body/run_bounded_mouth_one_minute.py"
    checks.require(
        fixed["experimental_code_sha256"].get(str(runner)) == RUNNER_SHA,
        "unqualified producer implementation",
    )
    for path, digest in fixed["experimental_code_sha256"].items():
        source = bind(path, digest, hashes)
        checks.require(source.is_relative_to(REPO), "source escaped repository")
        bind(branch / "code_snapshot" / source.relative_to(REPO), digest, hashes)
    migration = bind(branch / "state_import.json", fixed["state_import_sha256"], hashes)
    arrays, meta, _ = comparison.load_bound_checkpoint(root, fixed, hashes)
    comparison.validate_lifecycle(meta, STOP, 4)
    checks.require(
        row["arrays"] == 39
        and all(
            row[k] == meta[k]
            for k in ("previous_frame", "processed_frames", "initialize_count", "restore_count")
        ),
        "committed header differs from decoded checkpoint",
    )
    return arrays, meta, fixed, checks.read(migration), copy.deepcopy(row)


def verify_bounded_import(migration, parent, parent_meta, new_meta):
    checks.require(
        migration["schema"] == "bounded_mouth_original_state_import_v1"
        and migration["status"] == "ok",
        "successful explicit bounded migration required",
    )
    comparison.verify_import(migration["directional_validation"], parent, parent_meta)
    checks.require(
        migration["arrays"] == array_signature(parent)
        and migration["original_metadata"] == parent_meta
        and migration["old_context"] == parent_meta["context"]
        and migration["new_context"] == new_meta["context"],
        "wrong original state origin",
    )
    expected = copy.deepcopy(parent_meta)
    expected["restore_count"] += 1
    expected["context"] = migration["new_context"]
    checks.require(migration["imported_metadata"] == expected, "migration changed other metadata")
    context = copy.deepcopy(migration["new_context"])
    policy = context.pop("bounded_mouth_policy")
    checks.require(
        policy["enabled"] is True and policy["clears_identity_review"] is False,
        "bounded policy disabled or clears review",
    )
    context["implementation"] = migration["directional_context"]["implementation"]
    checks.require(
        context
        == migration["directional_context"]
        == migration["directional_validation"]["new_context"],
        "undeclared context change",
    )
    checks.require(
        migration["all_arrays_byte_preserved"] is True
        and migration["original_metadata_unchanged"] is True
        and migration["logical_initialize_count"] == 1
        and migration["new_begin_calls"] == 0
        and migration["added_restore_count"] == 1
        and migration["review_cleared_by_migration"] is False,
        "migration preservation flags differ",
    )


def semantic_decode(cls, arrays, metadata, rig, seed):
    """Read-only frozen validation; explicitly adapt only device in a copied context."""
    config = copy.deepcopy(metadata["context"]["tracker_config"])
    checks.require(config["device"] == "cuda", "original CUDA context required")
    config["device"] = "cpu"
    tracker = cls(
        rig,
        seed,
        TrackerConfig(**config),
        recovery_config=NodeRecoveryConfig(**metadata["context"]["recovery_config"]),
        persistence_config=PersistentNodeConfig(**metadata["context"]["persistence_config"]),
    )
    actual = tracker._context()
    claimed = copy.deepcopy(actual)
    claimed["tracker_config"]["device"] = "cuda"
    checks.require(claimed == metadata["context"], "saved core context differs beyond device")
    adapted = copy.deepcopy(metadata)
    adapted["context"] = actual
    before = array_signature(arrays)
    history, states, previous, _ = tracker._decode_state(arrays, adapted)
    checks.require(
        array_signature(arrays) == before
        and tracker.initialize_count == 0
        and tracker.processed_frames == 0,
        "read-only validation mutated tracker/input",
    )
    checks.require(
        len(history) == STOP - FIRST and len(states) == 6 and previous.frame_idx == STOP - 1,
        "decoded boundary/state differs",
    )
    return dict(
        array_count=len(arrays),
        history_frames=len(history),
        history_rows=sum(map(len, history)),
        active_states=len(states),
        previous_frame=previous.frame_idx,
        frozen_semantic_decoder_pass=True,
        validation_context_device="cpu",
        saved_context_device="cuda",
        all_history_ages_review_velocity_cost_checked=True,
        restore_calls=0,
        advance_calls=0,
    )


def compare_support(old, new):
    checks.require(
        checks.equal_bytes(old["frame_idx"], new["frame_idx"]), "different native frames"
    )
    checks.require(
        np.array_equal(old["frame_idx"], np.arange(START, STOP)), "same1660 horizon required"
    )
    a, b = old["observed"], new["observed"]
    gains, losses = b & ~a, a & ~b
    changed = []
    for t, fish, node in np.argwhere(gains | losses):
        changed.append(
            dict(
                frame_idx=int(new["frame_idx"][t]),
                slot=int(fish),
                node=NODES[node],
                change="gain" if gains[t, fish, node] else "loss",
                directional_review=bool(old["identity_review"][t, fish]),
                bounded_review=bool(new["identity_review"][t, fish]),
                directional_sources=old["sources"][t, fish, :, node].tolist(),
                bounded_sources=new["sources"][t, fish, :, node].tolist(),
            )
        )
    return dict(
        gained_by_slot_node=gains.sum(0).tolist(),
        lost_by_slot_node=losses.sum(0).tolist(),
        gained_by_node=gains.sum((0, 1)).tolist(),
        lost_by_node=losses.sum((0, 1)).tolist(),
        shared_direct_by_node=(a & b).sum((0, 1)).tolist(),
        support_events_are_not_anatomical_or_ID_truth=True,
    ), changed


def node_reacquisitions(candidate):
    """Keep missing spans and elapsed time; never bridge missing nodes as instantaneous speed."""
    frames, held = candidate["frame_idx"], candidate["observed"]
    events, runs = [], []
    for fish in range(3):
        for node, name in enumerate(NODES):
            last, first_missing = None, None
            for t, native in enumerate(frames):
                native = int(native)
                if not held[t, fish, node]:
                    if first_missing is None:
                        first_missing = native
                    continue
                if first_missing is not None:
                    if max(first_missing, START) < min(native, STOP):
                        runs.append(
                            dict(
                                slot=fish,
                                node=name,
                                start=first_missing,
                                stop=native,
                                missing_frames_in_interval=min(native, STOP)
                                - max(first_missing, START),
                                terminal_open=False,
                            )
                        )
                    if START <= native < STOP:
                        distance = (
                            None
                            if last is None
                            else float(
                                np.linalg.norm(
                                    candidate["pose_xyz"][t, fish, node]
                                    - candidate["pose_xyz"][last, fish, node]
                                )
                            )
                        )
                        elapsed = None if last is None else native - int(frames[last])
                        events.append(
                            dict(
                                frame_idx=native,
                                slot=fish,
                                node=name,
                                missing_frames=native - first_missing,
                                previous_direct_frame=None if last is None else int(frames[last]),
                                direct_endpoint_distance_mm=distance,
                                direct_endpoint_elapsed_frames=elapsed,
                                direct_endpoint_average_mm_per_s=None
                                if last is None
                                else distance * 166 / elapsed,
                                identity_review=bool(candidate["identity_review"][t, fish]),
                                biological_identity_confirmed=False,
                            )
                        )
                    first_missing = None
                last = t
            if first_missing is not None and max(first_missing, START) < STOP:
                runs.append(
                    dict(
                        slot=fish,
                        node=name,
                        start=first_missing,
                        stop=STOP,
                        missing_frames_in_interval=STOP - max(first_missing, START),
                        terminal_open=True,
                    )
                )
    return dict(events=events, missing_runs=runs)


def load_inputs(branch, expected_sha, hashes):
    new, new_meta, fixed, migration, row = committed(branch, expected_sha, hashes)
    bind(OLD / "tracking.provenance.json", OLD_SEAL_SHA, hashes)
    old_seal = checks.verify(
        OLD, "tracking.provenance.json", "directional_one_minute_continuous", hashes
    )
    old_root = OLD / "checkpoints/through_068060"
    checks.require(
        hashes.get(str(old_root / "manifest.json")) == OLD_MANIFEST_SHA,
        "old same-horizon checkpoint changed",
    )
    old_fixed = checks.read(old_root / "manifest.json")["bindings"]
    checks.require(
        old_fixed["schema"] == "directional_one_minute_checkpoint_binding_v1"
        and old_fixed["native_interval"] == [START, comparison.STOP]
        and fixed["original_binding"] == old_fixed["original_binding"],
        "different original parent/input configuration",
    )
    for path, digest in old_fixed["experimental_code_sha256"].items():
        source = bind(path, digest, hashes)
        bind(OLD / "code_snapshot" / source.relative_to(REPO), digest, hashes)
    old, old_meta, _ = comparison.load_bound_checkpoint(old_root, old_fixed, hashes)
    comparison.validate_lifecycle(old_meta, STOP, 4)
    binding = fixed["original_binding"]
    input_member(old_seal, PARENT / "continuation.provenance.json", hashes)
    parent_seal = checks.verify(
        PARENT, "continuation.provenance.json", "three_body_long_continuation", hashes
    )
    checks.require(parent_seal["params"]["bindings"] == binding, "parent binding mismatch")
    parent, parent_meta, _ = comparison.load_bound_checkpoint(
        PARENT / "checkpoint", binding, hashes
    )
    comparison.validate_lifecycle(parent_meta, START, 3)
    verify_bounded_import(migration, parent, parent_meta, new_meta)
    old_import = bind(OLD / "state_import.json", old_fixed["state_import_sha256"], hashes)
    comparison.verify_import(checks.read(old_import), parent, parent_meta)
    checks.require(
        checks.read(old_import)["new_context"] == old_meta["context"], "old import context"
    )
    for arrays in (old, new):
        comparison.verify_history_prefix(parent, arrays)
    for path, digest in binding["code_sha256"].items():
        bind(path, digest, hashes)
    seed_info = binding["inputs"]["seed"]
    seed_path = bind(seed_info["registration_path"], seed_info["registration_sha256"], hashes)
    seed = np.asarray(checks.read(seed_path)["pose_xyz"], np.float64)
    shared = seed_info["SL_and_calibration_sha256"]
    checks.require(len(shared) == 7, "SL3/calibration4 required")
    for path, digest in shared.items():
        input_member(old_seal, path, hashes)
        checks.require(hashes[str(Path(path).resolve())] == digest, "shared input differs")
    paths = {
        v: next(Path(p) for p in shared if Path(p).name == EXPERIMENT + "__" + v + ".npz")
        for v in VIEWS
    }
    calibration = next(
        Path(p).parent for p in shared if Path(p).name == "calibration_manifest.json"
    )
    rig, nodes = load_rig(calibration), RegisteredNodes(paths, np.arange(START, STOP))
    decoded = {
        "directional": semantic_decode(DirectionalChestContinuousTracker, old, old_meta, rig, seed),
        "bounded": semantic_decode(bounded.BoundedMouthContinuousTracker, new, new_meta, rig, seed),
    }
    traces = {
        name: comparison.short.backtrace(arrays)
        for name, arrays in (("directional", old), ("bounded", new))
    }
    for trace in traces.values():
        checks.require(
            len(trace) == 21 and np.array_equal(trace["frame_idx"], np.arange(FIRST, STOP)),
            "all21 same68060 backtrace required",
        )
    return traces, decoded, nodes, rig, row, fixed


def main(args):
    branch, output = args.branch.resolve(strict=True), args.output.resolve()
    checks.require(output.is_relative_to(REPO / "work") and not output.exists(), "new work output")
    code = comparison.source_registry.local_code()
    hashes, outputs = dict(code), {}
    output.mkdir(parents=True)
    with provenance(STAGE, output, name="audit") as record:

        def emit(name, value=None):
            path = output / name
            if value is not None:
                path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
            digest = checks.sha(path)
            outputs[str(path)] = digest
            record.output(name, path, sha256=digest)

        for path, digest in code.items():
            target = output / "code_snapshot" / Path(path).relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(path).read_bytes())
            checks.require(checks.sha(target) == digest, "source snapshot changed")
            emit(str(target.relative_to(output)))
        traces, decoded, nodes, rig, row, fixed = load_inputs(
            branch, args.expected_manifest_sha256, hashes
        )
        parts, methods = {}, {}
        for name, trace in traces.items():
            parts[name] = {k: v[START - FIRST : STOP - FIRST].copy() for k, v in trace.items()}
            part = parts[name]
            np.savez_compressed(output / f"{name}_same68060_candidate.npz", **part)
            emit(f"{name}_same68060_candidate.npz")

            def save_support(_meta, arrays, label=name):
                np.savez_compressed(output / f"{label}_strict_support.npz", **arrays)
                emit(f"{label}_strict_support.npz")

            support = comparison.support_audit.verify_support_partitioned(
                part, nodes, rig, on_block=save_support
            )
            accepted = part["observed"] & ~part["identity_review"][:, :, None]
            methods[name] = dict(
                counts=comparison.short.stats(trace, START, STOP),
                direct_by_slot_node=part["observed"].sum(0).tolist(),
                accepted_by_slot_node=accepted.sum(0).tolist(),
                accepted_by_node=accepted.sum((0, 1)).tolist(),
                selected_kind_counts=dict(Counter(part["proposal_kind"].tolist())),
                all_native_source_geometry=support,
                review_scope="original inherited sticky per-slot review; no human correction",
            )
            emit(f"{name}_reacquisitions.json", node_reacquisitions(trace))
        delta, changes = compare_support(parts["directional"], parts["bounded"])
        emit("gain_loss.json", changes)
        emit("committed_checkpoint_record.json", row)
        result = dict(
            schema=STAGE,
            status="ok",
            native_interval=[START, STOP],
            native_frames=1660,
            horizon_stop=STOP,
            expected_manifest_sha256=args.expected_manifest_sha256,
            completed_checkpoint_interval=[START, STOP],
            complete_minute=False,
            parent_run_may_still_be_running=True,
            same_parent_and_horizon=True,
            source_snapshot_and_committed_checkpoint_verified=True,
            all39_arrays_semantically_decoded=True,
            all21_arrays_backtraced=True,
            initial_parent_history_prefix_preserved=True,
            logical_initialization_count=1,
            checkpoint_restore_count=4,
            restores_during_interval=0,
            external_pose_resets=0,
            independent_ID_truth=False,
            adopted=False,
            methods=methods,
            support_delta=delta,
            checkpoint_validation=decoded,
            CPU_validation_scope="copied context changes device only; no restore/fit/advance",
            shared_input_binding=fixed["original_binding"],
            comparison_limit=(
                "same anonymous slots; direct support is not anatomical or biological ID truth"
            ),
            interim_input_limit=(
                "original input/seed/calibration binding and loaded code are exact; "
                "new full-run input/output seal and raw/D2 replay are not yet audited here"
            ),
            old_full_run_output_scope=(
                "all completed outputs SHA-checked; only through68060 checkpoint arrays "
                "are deserialized, never the later final winner"
            ),
        )
        emit("summary.json", result)
        a, b = methods["directional"]["counts"], methods["bounded"]["counts"]
        (output / "RESULT.md").write_text(
            "# 最初の10秒・同68060終端の独立監査\n\n"
            f"[66400,68060) の1660時刻。口/胸部/尾の直接支持: "
            f"旧 {a['direct_node_support']} → bounded {b['direct_node_support']}。"
            f"元review下の胸部利用数: {a['accepted_chest_fish_frames']} → "
            f"{b['accepted_chest_fish_frames']}。\n\n"
            "共通66400親39配列・明示migration・初回commit SHA・全history再帰・"
            "同SL3/校正4による2mm/12px支持を検証した。"
            "可変progressは最初のcommit記録のみ固定し、将来のwinnerを読んでいない。\n\n"
            "これは最初の10秒の中間監査で、全1分の完走や生物IDの正解を認定しない。"
            "再観測は欠測期間を明示し、隔たる端点変位を瞬間速度と扱わない。"
            "旧原baselineとの三方式比較はこの監査に含めていない。\n",
            encoding="utf-8",
        )
        emit("RESULT.md")
        checks.require(
            checks.read(branch / "checkpoint_progress.json")[0] == row,
            "first commit changed while parent continued",
        )
        checks.require(code == comparison.source_registry.local_code(), "loaded source set changed")
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
        for path, digest in {**hashes, **outputs}.items():
            checks.require(checks.sha(path) == digest, "final bound input/source/output changed")
        record.param("all_source_input_snapshot_output_SHA_unchanged", True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(main(parser.parse_args()), indent=2))
