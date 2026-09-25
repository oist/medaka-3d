"""Completed same76360 four-way comparison; only new tail support is projected.

Old three support proofs are reused through the completed three-way comparison,
exact candidate/source bindings and direct counts. All four terminal39/21 are
validated; only the new tail six-checkpoint chain is traversed again. No fitting,
tracking, image decoding, new identity matching or evaluation threshold changes.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import compare_bounded_one_minute as old
import numpy as np
from tail_recovery_tracker import TailRecoveryContinuousTracker

from medaka3d.eval.generative import MouthAnchor
from medaka3d.provenance import provenance

checks, prior, interim = old.checks, old.prior, old.interim
REPO, WORK = old.REPO, old.WORK
FIRST, START, STOP, CHUNK = old.FIRST, old.START, old.STOP, old.CHUNK
PARENT = old.PARENT
STAGE = "tail_one_minute_comparison_v1"
METHODS = (*old.METHODS, "tail")
PREVIOUS = WORK / "bounded_one_minute_comparison_v1"
PREVIOUS_SHA = "3eb86ef4637b07946dfa88db545b5baecc2c046af7a4485d087f5d1c66f6d192"
TAIL_RUNNER = WORK / "run_tail_one_minute.py"
TAIL_RUNNER_SHA = "3c11909ab244cfdb1bae3589baa987a9102827111372db00c853d4666ca18ca8"
TAIL_CLASS_SHA = "b5a98085c6b1b03bd17e2f0b627ffc3eebdd569cf5add545146d9322761868a2"
ROOTS = dict(
    original=old.ORIGINAL,
    directional=old.DIRECTIONAL,
    bounded=WORK / "bounded_mouth_one_minute_66400-76360_v1",
)
PINS = {
    **old.PINS,
    Path(
        old.__file__
    ).resolve(): "5e1a4f570afadf204f96f3c6e83d55a9089f6fa3a14cf1b4c54cc84e4fe76cd0",
    TAIL_RUNNER: TAIL_RUNNER_SHA,
    WORK / "tail_recovery_tracker.py": TAIL_CLASS_SHA,
}
bind, input_binding = old.bind, old.input_binding


def successful(root, filename, stage, hashes, expected=None):
    record = old.successful(root, filename, stage, hashes, expected)
    if expected is not None:
        checks.require(
            hashes[str(Path(root).resolve() / filename)] == expected,
            "expected seal changed during verification",
        )
    return record


def validate_completion(s):
    checks.require(
        s.get("schema") == "tail_one_minute_v1"
        and s.get("status") == "ok"
        and s.get("one_minute_completed") is True
        and s.get("native_interval") == [START, STOP]
        and s.get("native_frames") == 9960
        and s.get("logical_initialization_count") == 1
        and s.get("restore_count") == 4
        and s.get("external_pose_resets") == 0
        and s.get("checkpoint_count") == 6
        and s.get("diagnostic_frames") == 572
        and s.get("periodic_checkpoints_are_not_reinitializations") is True,
        "completed tail minute/lifecycle required",
    )


def validate_prior_summary(s):
    checks.require(
        s.get("schema") == old.STAGE
        and s.get("status") == "ok"
        and s.get("native_interval") == [START, STOP]
        and s.get("native_frames") == 9960
        and s.get("horizon_stop") == STOP
        and all(
            s.get(k) is True
            for k in (
                "same_parent_and_horizon",
                "one_minute_completed",
                "all39_arrays_semantically_decoded",
                "all21_arrays_exact_backtrace",
                "original66400_parent_and_migration_verified",
                "bounded_all_six_checkpoint_prefixes_exact",
                "all_seven_C34_conditions_retained",
            )
        )
        and s.get("logical_initialization_count") == 1
        and s.get("checkpoint_restore_count") == 4
        and s.get("restores_during_one_minute") == s.get("external_pose_resets") == 0
        and s.get("requested_female_anchors") == 11
        and s.get("requested_female_intervals") == 10
        and all(
            set(s.get(k, {})) == set(old.METHODS)
            for k in ("methods", "candidate_paths", "candidate_sha256")
        ),
        "complete prior three-way scope required",
    )


def verify_tail_import(migration, parent, parent_meta, new_meta):
    checks.require(
        migration["schema"] == "tail_recovery_original_state_import_v1"
        and migration["status"] == "ok",
        "successful explicit tail migration required",
    )
    bounded_meta = copy.deepcopy(new_meta)
    bounded_meta["context"] = migration["bounded_context"]
    interim.verify_bounded_import(
        migration["bounded_validation"], parent, parent_meta, bounded_meta
    )
    checks.require(
        migration["arrays"] == old.array_signature(parent)
        and migration["original_metadata"] == parent_meta
        and migration["old_context"] == parent_meta["context"]
        and migration["new_context"] == new_meta["context"],
        "wrong original tail state origin",
    )
    expected = copy.deepcopy(parent_meta)
    expected["restore_count"] += 1
    expected["context"] = migration["new_context"]
    checks.require(
        migration["imported_metadata"] == expected, "tail migration changed other metadata"
    )
    context = copy.deepcopy(migration["new_context"])
    policy = context.pop("tail_recovery_policy")
    checks.require(
        policy["enabled"] is True
        and policy["target_node"] == 2
        and policy["allow_bounded"] is False
        and policy["clears_identity_review"] is False,
        "tail policy disabled/changed or clears review",
    )
    context["implementation"] = migration["bounded_context"]["implementation"]
    checks.require(
        context == migration["bounded_context"] == migration["bounded_validation"]["new_context"],
        "undeclared tail context change",
    )
    checks.require(
        migration["all_arrays_byte_preserved"] is True
        and migration["original_metadata_unchanged"] is True
        and migration["logical_initialize_count"] == 1
        and migration["new_begin_calls"] == 0
        and migration["added_restore_count"] == 1
        and migration["review_cleared_by_migration"] is False
        and migration["original_implementation_not_claimed_for_new_tracking"] is True,
        "tail migration preservation flags differ",
    )


def audit_tail(branch, binding, parent, parent_meta, hashes):
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
            current["schema"] == "tail_one_minute_checkpoint_binding_v1"
            and current["original_binding"] == binding
            and current["native_interval"] == [START, STOP]
            and current["state_import_sha256"] == hashes[str(branch / "state_import.json")]
            and current["experimental_code_sha256"].get(str(WORK / "run_tail_one_minute.py"))
            == TAIL_RUNNER_SHA,
            "tail checkpoint scientific/source/parent binding differs",
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
        verify_tail_import(migration, parent, parent_meta, meta)
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


def old_terminal(name, root, binding, parent, parent_meta, hashes):
    path = root / ("checkpoint" if name == "original" else "checkpoints/through_076360")
    fixed = checks.read(path / "manifest.json")["bindings"]
    migration = None
    if name == "original":
        checks.require(fixed == binding, "original checkpoint binding differs")
    else:
        checks.require(
            fixed["original_binding"] == binding and fixed["native_interval"] == [START, STOP],
            "old branch scientific origin differs",
        )
        migration = checks.read(root / "state_import.json")
        checks.require(
            fixed["state_import_sha256"] == hashes[str(root / "state_import.json")],
            "old migration SHA differs",
        )
        for path_name, digest in fixed["experimental_code_sha256"].items():
            source = bind(path_name, digest, hashes)
            checks.require(
                source.is_relative_to(REPO)
                and hashes.get(str(root / "code_snapshot" / source.relative_to(REPO))) == digest,
                "old source snapshot differs",
            )
    arrays, meta, _ = prior.load_bound_checkpoint(path, fixed, hashes)
    prior.validate_lifecycle(meta, STOP, 4)
    prior.verify_history_prefix(parent, arrays)
    if name == "directional":
        prior.verify_import(migration, parent, parent_meta)
        checks.require(meta["context"] == migration["new_context"], "directional context differs")
    elif name == "bounded":
        interim.verify_bounded_import(migration, parent, parent_meta, meta)
    return prior.load_candidate(root, arrays, hashes), arrays, meta


def load_inputs(branch, expected_sha, hashes):
    # Completion gates precede every candidate/checkpoint payload read.
    tail_seal = successful(
        branch, "tracking.provenance.json", "tail_one_minute_continuous", hashes, expected_sha
    )
    prior.require_bound(branch, ("summary.json",), hashes)
    validate_completion(checks.read(branch / "summary.json"))
    reused_seal = successful(
        PREVIOUS, "comparison.provenance.json", old.STAGE, hashes, PREVIOUS_SHA
    )
    prior.require_bound(PREVIOUS, ("summary.json",), hashes)
    saved = checks.read(PREVIOUS / "summary.json")
    validate_prior_summary(saved)
    producers = {}
    for name, root in ROOTS.items():
        filename, stage = (
            ("continuation.provenance.json", "three_body_long_continuation")
            if name == "original"
            else ("tracking.provenance.json", "directional_one_minute_continuous")
            if name == "directional"
            else ("tracking.provenance.json", "bounded_mouth_one_minute_continuous")
        )
        producers[name] = successful(root, filename, stage, hashes)
        for path in (root / filename, root / "candidate.npz"):
            input_binding(reused_seal, path, hashes[str(path)])
        checks.require(
            Path(saved["candidate_paths"][name]).resolve() == root / "candidate.npz"
            and saved["candidate_sha256"][name] == hashes[str(root / "candidate.npz")],
            "old candidate differs from reused support proof",
        )
    binding = producers["original"]["params"]["bindings"]
    checks.require(
        producers["original"]["params"]["through_frame"] == STOP
        and Path(producers["original"]["params"]["resume_phase"]).resolve() == PARENT,
        "original parent/horizon differs",
    )
    parent_seal = successful(
        PARENT, "continuation.provenance.json", "three_body_long_continuation", hashes
    )
    checks.require(parent_seal["params"]["bindings"] == binding, "parent binding differs")
    for seal in (reused_seal, tail_seal, producers["directional"], producers["bounded"]):
        input_binding(
            seal,
            PARENT / "continuation.provenance.json",
            hashes[str(PARENT / "continuation.provenance.json")],
        )
    for path, digest in binding["code_sha256"].items():
        bind(path, digest, hashes)
    parent, parent_meta, _ = prior.load_bound_checkpoint(PARENT / "checkpoint", binding, hashes)
    prior.validate_lifecycle(parent_meta, START, 3)
    traces, states, metas = {}, {}, {}
    for name, root in ROOTS.items():
        traces[name], states[name], metas[name] = old_terminal(
            name, root, binding, parent, parent_meta, hashes
        )
    traces["tail"], states["tail"], metas["tail"], audits, tail_summary = audit_tail(
        branch, binding, parent, parent_meta, hashes
    )
    checks.require(
        tail_summary["direct_node_support"]
        == traces["tail"]["observed"][START - FIRST :].sum((0, 1)).tolist()
        and tail_summary["identity_review_fish_frames"]
        == int(traces["tail"]["identity_review"][START - FIRST :].sum()),
        "tail reported counts differ",
    )
    shared = binding["inputs"]["seed"]["SL_and_calibration_sha256"]
    checks.require(
        len(shared) == 7 and saved["shared_SL_calibration_sha256"] == shared,
        "reused SL3/calibration4 differs",
    )
    for path, digest in shared.items():
        bind(path, digest, hashes)
        for seal in (reused_seal, tail_seal, *producers.values()):
            input_binding(seal, path, digest)
    paths = {
        v: next(Path(p) for p in shared if Path(p).name == prior.EXPERIMENT + "__" + v + ".npz")
        for v in prior.VIEWS
    }
    calibration = next(
        Path(p).parent for p in shared if Path(p).name == "calibration_manifest.json"
    )
    rig, nodes = prior.load_rig(calibration), prior.RegisteredNodes(paths, np.arange(START, STOP))
    info = binding["inputs"]["seed"]
    seed_path = bind(info["registration_path"], info["registration_sha256"], hashes)
    seed = np.asarray(checks.read(seed_path)["pose_xyz"], np.float64)
    classes = dict(
        original=old.AcceleratedContinuousPersistentNodeTracker,
        directional=old.DirectionalChestContinuousTracker,
        bounded=interim.bounded.BoundedMouthContinuousTracker,
        tail=TailRecoveryContinuousTracker,
    )
    decoded = {
        name: old.semantic_decode(classes[name], states[name], metas[name], rig, seed)
        for name in METHODS
    }
    return traces, decoded, nodes, rig, audits, saved, reused_seal, shared


def evaluation_inputs(traces, saved, reused_seal, shared, hashes):
    female_seal = successful(
        old.FEMALE,
        "evaluation.provenance.json",
        "long_female_endpoint_evaluation_160s",
        hashes,
        old.FEMALE_SHA,
    )
    input_binding(reused_seal, old.FEMALE / "evaluation.provenance.json", old.FEMALE_SHA)
    prior.require_bound(old.FEMALE, ("anchor_matches.json", "original_female_sex_gt.csv"), hashes)
    checks.require(
        hashes[str(old.FEMALE / "original_female_sex_gt.csv")] == prior.GT_SHA,
        "female version differs",
    )
    raw = [r for r in checks.read(old.FEMALE / "anchor_matches.json") if START <= r["frame"] < STOP]
    checks.require(
        tuple(r["frame"] for r in raw) == old.ANCHOR_FRAMES, "fixed11 female anchors differ"
    )
    anchors = [MouthAnchor(**r) for r in raw]
    for path, digest in shared.items():
        input_binding(female_seal, path, digest)
    classical_seal = successful(
        old.CLASSICAL,
        "comparison.provenance.json",
        "directional_minute_all7_classical_comparison_v1",
        hashes,
        old.CLASSICAL_SHA,
    )
    input_binding(reused_seal, old.CLASSICAL / "comparison.provenance.json", old.CLASSICAL_SHA)
    input_binding(classical_seal, old.FEMALE / "evaluation.provenance.json", old.FEMALE_SHA)
    prior.require_bound(old.CLASSICAL, ("summary.json",), hashes)
    classical = old.classical_rows(checks.read(old.CLASSICAL / "summary.json"), traces, saved)
    checks.require(classical == saved["classical_rows"], "reused all7 classical rows differ")
    for name in old.METHODS:
        checks.require(
            prior.female_scores(traces[name], anchors) == saved["methods"][name]["female"],
            "saved same11 female score differs",
        )
    return anchors, classical


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
            dest = output / "code_snapshot" / Path(path).relative_to(REPO)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(Path(path).read_bytes())
            checks.require(checks.sha(dest) == digest, "source snapshot changed")
            emit(str(dest.relative_to(output)))
        traces, decoded, nodes, rig, audits, saved, reused_seal, shared = load_inputs(
            branch, args.expected_seal_sha256, hashes
        )
        anchors, classical = evaluation_inputs(traces, saved, reused_seal, shared, hashes)
        methods = {}
        for name in METHODS:
            candidate = traces[name]
            part = {
                k: candidate[k][START - FIRST :]
                for k in ("frame_idx", "pose_xyz", "sources", "observed")
            }
            if name == "tail":

                def save_support(meta, arrays):
                    a, b = meta["frame_range"]
                    filename = f"tail_support_f{a:06d}-{b:06d}.npz"
                    np.savez_compressed(output / filename, **arrays)
                    emit(filename)

                geometry = prior.support_audit.verify_support_partitioned(
                    part, nodes, rig, on_block=save_support
                )
            else:
                geometry = old.reused_support(candidate, name, saved)
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
                support_recomputed_here=name == "tail",
                support_reused_comparison_sha256=PREVIOUS_SHA if name != "tail" else None,
                female=prior.female_scores(candidate, anchors),
            )
            metrics = old.node_metrics(candidate)
            methods[name]["node_metrics"] = metrics["by_slot_node"]
            emit(name + "_gaps_and_reacquisitions.json", metrics)
        deltas = {}
        for a, b in combinations(METHODS, 2):
            key = a + "_to_" + b
            deltas[key], changes = old.support_delta(traces[a], traces[b])
            emit(key + "_gain_loss.json", changes)
        roots = {**ROOTS, "tail": branch}
        candidates = {name: str(root / "candidate.npz") for name, root in roots.items()}
        result = dict(
            schema=STAGE,
            status="ok",
            native_interval=[START, STOP],
            native_frames=9960,
            horizon_stop=STOP,
            same_parent_and_horizon=True,
            one_minute_completed=True,
            candidate_paths=candidates,
            candidate_sha256={n: hashes[p] for n, p in candidates.items()},
            all39_arrays_semantically_decoded=True,
            all21_arrays_exact_backtrace=True,
            original66400_parent_and_migration_verified=True,
            tail_all_six_checkpoint_prefixes_exact=True,
            old_three_checkpoint_prefix_proofs_reused=True,
            logical_initialization_count=1,
            checkpoint_restore_count=4,
            restores_during_one_minute=0,
            external_pose_resets=0,
            owned_process_completion=dict(
                session=args.owned_session,
                exit_code=0,
                scope="ROOT-observed owned tool exit; not a new OS witness",
            ),
            methods=methods,
            support_delta=deltas,
            checkpoint_validation=decoded,
            shared_SL_calibration_sha256=shared,
            reused_three_way_comparison_sha256=PREVIOUS_SHA,
            reused_classical_comparison_sha256=old.CLASSICAL_SHA,
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
                "Fullminute raw-pixel scoring is a separate artifact.",
                "Fixed saved female matches reused; no new matching or dense male identity truth.",
                "Same numeric slots, no biological-ID permutation or review clearing.",
                "Direct steps require supported endpoints; latent boundary/gap scope is explicit.",
            ],
        )
        emit("checkpoint_audit.json", audits)
        emit("summary.json", result)
        for path, digest in hashes.items():
            checks.require(checks.sha(path) == digest, "bound input/source changed")
            record.input(path, path, sha256=digest)
        for path, digest in outputs.items():
            checks.require(checks.sha(path) == digest, "output changed")
        record.param("all_bound_files_unchanged", True)
    print(
        json.dumps({name: result["methods"][name]["full_minute"] for name in METHODS}), flush=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", type=Path, required=True)
    parser.add_argument("--expected-seal-sha256", required=True)
    parser.add_argument("--owned-session", type=int, required=True)
    parser.add_argument("--owned-exit-code", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
