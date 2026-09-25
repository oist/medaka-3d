"""Audit and compare completed original/directional traces at native 76360.

The six checkpoints must preserve every previously retained history row. Their
earlier recorded manifest digests and source snapshots are rechecked explicitly.
Sparse female annotations are evaluation inputs only; no male/dense ID verdict.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import audit_long_final as support_audit
import compare_directional_continuous as short
import fit_chest_recovery_candidates as source_registry
import numpy as np
import summarize_chest_candidate_replay as checks
from continuous_tracker import RECORD_FIELDS
from long_baseline_support import EXPERIMENT, RegisteredNodes
from tracker_checkpoint import read_checkpoint

from medaka3d.calib import load_rig
from medaka3d.eval.generative import match_mouth_anchors, score_female_identity
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS

REPO = Path(__file__).resolve().parents[2]
FIRST, START, STOP, CHUNK, FPS = 49800, 66400, 76360, 1660, 166
TARGETS = ((66656, 66706), (75047, 75163))
GT_SHA = "ab0a173ac737a63311aec834c5832fc19475d4f7d1892b8193f8465bba67519f"


def same(a, b):
    if a.dtype.kind == b.dtype.kind == "U":
        return a.shape == b.shape and np.array_equal(a, b)
    return checks.equal_bytes(a, b)


def verify_history_prefix(prior, current):
    """All retained alternatives persist, including unchosen old beam records."""
    for key in ("history_offsets", *("history_" + k for k in RECORD_FIELDS)):
        checks.require(
            same(prior[key], current[key][: len(prior[key])]),
            "retained history prefix changed: " + key,
        )


def validate_lifecycle(meta, stop, restores):
    checks.require(
        meta["previous_frame"] == stop - 1
        and meta["processed_frames"] == stop - FIRST
        and meta["initialize_count"] == 1
        and meta["restore_count"] == restores,
        "checkpoint horizon or initialization/restore count differs",
    )


def load_bound_checkpoint(path, binding, hashes):
    for name in ("manifest.json", "metadata.json", "arrays.npz"):
        checks.require(str(path / name) in hashes, "unsealed checkpoint component")
    arrays, meta, manifest = read_checkpoint(path, expected_bindings=binding)
    checks.require(
        len(arrays) == 39 and arrays["state_pose"].shape == (6, 3, 3, 3),
        "complete 39-array/six-beam checkpoint required",
    )
    return arrays, meta, manifest


def load_candidate(root, checkpoint, hashes):
    path = root / "candidate.npz"
    checks.require(str(path) in hashes, "unsealed candidate")
    traced = short.backtrace(checkpoint)
    with np.load(path, allow_pickle=False) as data:
        checks.require(
            set(data.files) == set(traced) and len(data.files) == 21, "candidate schema differs"
        )
        for key in traced:
            checks.require(same(data[key], traced[key]), "independent backtrace differs: " + key)
    checks.require(
        np.array_equal(traced["frame_idx"], np.arange(FIRST, STOP)),
        "same complete native horizon required",
    )
    return traced


def require_bound(root, names, hashes):
    for name in names:
        checks.require(str(root / name) in hashes, "required artifact file is unsealed: " + name)


def verify_import(migration, parent, metadata):
    signatures = {
        key: dict(
            dtype=value.dtype.str,
            shape=list(value.shape),
            sha256=hashlib.sha256(value.tobytes()).hexdigest(),
        )
        for key, value in sorted(parent.items())
    }
    checks.require(
        migration["schema"] == "directional_chest_original_state_import_v1"
        and migration["status"] == "ok"
        and migration["arrays"] == signatures
        and migration["original_metadata"] == metadata
        and migration["old_context"] == metadata["context"],
        "imported original arrays/metadata differ from actual parent",
    )
    expected = copy.deepcopy(metadata)
    expected["context"] = migration["new_context"]
    expected["restore_count"] += 1
    checks.require(
        migration["imported_metadata"] == expected,
        "imported metadata changed beyond declared context/one restore",
    )
    old_context = copy.deepcopy(migration["new_context"])
    old_context.pop("experimental_chest_policy")
    old_context["implementation"] = metadata["context"]["implementation"]
    checks.require(
        old_context == metadata["context"], "import changed original calibration/configuration"
    )


def audit_branch(branch, old_binding, parent, parent_metadata, hashes):
    require_bound(
        branch,
        ("summary.json", "checkpoint_progress.json", "final_checkpoint.json", "state_import.json"),
        hashes,
    )
    migration = checks.read(branch / "state_import.json")
    verify_import(migration, parent, parent_metadata)
    summary = checks.read(branch / "summary.json")
    checks.require(
        summary["status"] == "ok"
        and summary["one_minute_completed"]
        and summary["native_interval"] == [START, STOP]
        and summary["native_frames"] == 9960
        and summary["external_pose_resets"] == 0
        and summary["checkpoint_count"] == 6
        and summary["diagnostic_frames"] == 166,
        "one-minute branch incomplete",
    )
    rows = checks.read(branch / "checkpoint_progress.json")
    checks.require(len(rows) == 6, "six checkpoints required")
    previous, audits, fixed_binding = parent, [], None
    for index, stop in enumerate(range(START + CHUNK, STOP + CHUNK, CHUNK)):
        path = branch / "checkpoints" / f"through_{stop:06d}"
        row = rows[index]
        checks.require(
            Path(row["path"]).resolve() == path
            and row["manifest_sha256"] == checks.sha(path / "manifest.json"),
            "earlier checkpoint manifest digest/path changed",
        )
        binding = checks.read(path / "manifest.json")["bindings"]
        checks.require(
            binding["schema"] == "directional_one_minute_checkpoint_binding_v1"
            and binding["original_binding"] == old_binding
            and binding["native_interval"] == [START, STOP]
            and binding["state_import_sha256"] == hashes[str(branch / "state_import.json")],
            "experimental state/parent binding differs",
        )
        if fixed_binding is None:
            fixed_binding = binding
            for source, digest in binding["experimental_code_sha256"].items():
                source = Path(source).resolve()
                checks.require(source.is_relative_to(REPO), "source outside repository")
                snapshot = branch / "code_snapshot" / source.relative_to(REPO)
                checks.require(
                    str(snapshot) in hashes
                    and hashes[str(snapshot)] == digest
                    and checks.sha(source) == digest,
                    "initial code snapshot differs from recorded source",
                )
                hashes[str(source)] = digest
        checks.require(binding == fixed_binding, "checkpoint context changed between chunks")
        arrays, meta, _ = load_bound_checkpoint(path, binding, hashes)
        validate_lifecycle(meta, stop, 4)
        checks.require(
            meta["context"] == migration["new_context"], "checkpoint context differs from import"
        )
        checks.require(
            all(
                row[k] == meta[k]
                for k in ("previous_frame", "processed_frames", "initialize_count", "restore_count")
            )
            and row["arrays"] == 39,
            "checkpoint progress metadata differs",
        )
        verify_history_prefix(previous, arrays)
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
        checks.read(branch / "final_checkpoint.json") == rows[-1],
        "final checkpoint pointer differs",
    )
    return load_candidate(branch, previous, hashes), audits


def gap_metrics(candidate, start=START, stop=STOP):
    """Do not equate a correction after missing data with one-frame fish speed."""
    frames, pose = candidate["frame_idx"], candidate["pose_xyz"][:, :, 1]
    observed = candidate["observed"][:, :, 1]
    checks.require(np.all(np.diff(frames) == 1), "native gaps in candidate rows")
    events, runs = [], []
    for fish in range(3):
        last_direct, run_start = None, None
        for t, native in enumerate(frames):
            native = int(native)
            if not observed[t, fish]:
                if run_start is None:
                    run_start = t
                continue
            if run_start is not None:
                first_missing = int(frames[run_start])
                if max(first_missing, start) < min(native, stop):
                    runs.append(
                        dict(
                            slot=fish,
                            first_missing=first_missing,
                            missing_stop_exclusive=native,
                            closed_by_direct_observation=True,
                            missing_frames_in_evaluation=min(native, stop)
                            - max(first_missing, start),
                        )
                    )
                if start <= native < stop:
                    dt = None if last_direct is None else native - int(frames[last_direct])
                    distance = (
                        None
                        if dt is None
                        else float(np.linalg.norm(pose[t, fish] - pose[last_direct, fish]))
                    )
                    events.append(
                        dict(
                            frame_idx=native,
                            slot=fish,
                            missing_frames=native - first_missing,
                            last_direct_frame=None if dt is None else int(frames[last_direct]),
                            previous_latent_to_reacquired_step_mm=float(
                                np.linalg.norm(pose[t, fish] - pose[t - 1, fish])
                            )
                            if t
                            else None,
                            direct_endpoint_distance_mm=distance,
                            direct_endpoint_elapsed_frames=dt,
                            direct_endpoint_average_mm_per_s=None
                            if dt is None
                            else distance * FPS / dt,
                            reacquired_chest_xyz=pose[t, fish].tolist(),
                            identity_review=bool(candidate["identity_review"][t, fish]),
                            biological_identity_confirmed=False,
                        )
                    )
                run_start = None
            last_direct = t
        if run_start is not None:
            first_missing, terminal = int(frames[run_start]), int(frames[-1]) + 1
            if max(first_missing, start) < min(terminal, stop):
                runs.append(
                    dict(
                        slot=fish,
                        first_missing=first_missing,
                        missing_stop_exclusive=terminal,
                        closed_by_direct_observation=False,
                        missing_frames_in_evaluation=min(terminal, stop)
                        - max(first_missing, start),
                    )
                )
    keep = (frames >= start) & (frames < stop)
    pair = observed[1:] & observed[:-1] & keep[1:, None]
    steps = np.linalg.norm(np.diff(pose, axis=0), axis=-1)[pair]
    return dict(
        missing_runs=runs,
        reacquisitions=events,
        directly_supported_adjacent_steps=int(pair.sum()),
        directly_supported_adjacent_steps_gt10mm=int((steps > 10).sum()),
        maximum_directly_supported_adjacent_step_mm=float(steps.max()) if steps.size else None,
        sum_missing_chest_fish_frames=int((~observed[keep]).sum()),
        adjacent_step_scope=(
            "Ending frame inside interval; preceding native frame may be just before start."
        ),
    )


def female_scores(candidate, anchors):
    keep = (candidate["frame_idx"] >= START) & (candidate["frame_idx"] < STOP)
    values = (
        candidate["frame_idx"][keep],
        candidate["sources"][keep, :, :, 0],
        candidate["observed"][keep, :, 0],
        anchors,
    )
    return dict(
        raw=score_female_identity(*values),
        with_original_review=score_female_identity(
            *values, review=candidate["identity_review"][keep]
        ),
    )


def evaluate(args):
    original, branch, female, output = (
        getattr(args, k).resolve() for k in ("original", "branch", "female", "output")
    )
    checks.require(
        output.is_relative_to(REPO / "work") and not output.exists(),
        "new workspace output required",
    )
    hashes = source_registry.local_code()
    for module in (
        Path(__file__),
        Path(short.__file__),
        Path(checks.__file__),
        REPO / "src/medaka3d/eval/generative.py",
    ):
        hashes[str(module.resolve())] = checks.sha(module)
    old_seal = checks.verify(
        original, "continuation.provenance.json", "three_body_long_continuation", hashes
    )
    branch_seal = checks.verify(
        branch, "tracking.provenance.json", "directional_one_minute_continuous", hashes
    )
    binding = old_seal["params"]["bindings"]
    checks.require(old_seal["params"]["through_frame"] == STOP, "wrong reference horizon")
    for path, digest in binding["code_sha256"].items():
        checks.require(checks.sha(path) == digest, "original decoder source changed")
        hashes[path] = digest
    old_state, old_meta, _ = load_bound_checkpoint(original / "checkpoint", binding, hashes)
    validate_lifecycle(old_meta, STOP, 4)
    old = load_candidate(original, old_state, hashes)
    parent_root = Path(old_seal["params"]["resume_phase"]).resolve()
    parent_seal = checks.verify(
        parent_root, "continuation.provenance.json", "three_body_long_continuation", hashes
    )
    checks.require(
        parent_seal["params"]["bindings"] == binding
        and branch_seal["inputs"][str(parent_root / "continuation.provenance.json")]["sha256"]
        == hashes[str(parent_root / "continuation.provenance.json")],
        "different branch parent",
    )
    parent, meta, _ = load_bound_checkpoint(parent_root / "checkpoint", binding, hashes)
    validate_lifecycle(meta, START, 3)
    verify_history_prefix(parent, old_state)
    new, audits = audit_branch(branch, binding, parent, meta, hashes)
    female_seal = checks.verify(
        female, "evaluation.provenance.json", "long_female_endpoint_evaluation_160s", hashes
    )
    require_bound(female, ("anchor_matches.json", "original_female_sex_gt.csv"), hashes)
    checks.require(
        female_seal["params"]["all_input_source_output_sha_unchanged"],
        "female evaluation guard missing",
    )
    # Reuse independently audited annotation-to-detection matches, binding their SL version.
    for path, digest in binding["inputs"]["seed"]["SL_and_calibration_sha256"].items():
        checks.require(
            female_seal["inputs"][path]["sha256"] == digest and checks.sha(path) == digest,
            "annotation matching used different SL/calibration",
        )
        hashes[path] = digest
    shared = binding["inputs"]["seed"]["SL_and_calibration_sha256"]
    paths = {
        v: next(Path(p) for p in shared if Path(p).name == EXPERIMENT + "__" + v + ".npz")
        for v in VIEWS
    }
    calibration = [Path(p) for p in shared if Path(p).name == "calibration_manifest.json"]
    checks.require(
        len(shared) == 7 and len(calibration) == 1, "three SL/four calibration inputs required"
    )
    nodes = RegisteredNodes(paths, np.arange(START, STOP))
    rig = load_rig(calibration[0].parent)
    gt_path = female / "original_female_sex_gt.csv"
    checks.require(hashes[str(gt_path)] == GT_SHA, "female annotation version differs")
    with gt_path.open(encoding="utf-8-sig", newline="") as stream:
        anchors = match_mouth_anchors(
            csv.DictReader(stream), lambda v, f: nodes.at(v, f)[:, 0], START, STOP
        )
    saved = [
        row for row in checks.read(female / "anchor_matches.json") if START <= row["frame"] < STOP
    ]
    checks.require(
        json.loads(json.dumps([asdict(a) for a in anchors])) == saved,
        "fresh female annotation-to-SL matching differs",
    )
    checks.require(len(anchors) >= 2, "no evaluable female intervals")
    result = dict(
        schema="directional_one_minute_comparison_v1",
        status="ok",
        native_interval=[START, STOP],
        native_frames=9960,
        same_parent_and_horizon=True,
        all21_original_and_directional_arrays_exact_backtrace=True,
        original_history_and_all_six_checkpoint_prefixes_exact=True,
        checkpoint_snapshots_and_earlier_manifest_digests_verified=True,
        logical_initialization_count=1,
        restores_during_one_minute=0,
        external_pose_resets=0,
        all_three_ID_truth=False,
        adopted=False,
        female_scope="Sparse pre-existing female endpoints only; no dense/male ID truth.",
        latent_step_scope=(
            "short.stats chest_steps/max_chest_step include missing latent poses "
            "and are not measured fish speed."
        ),
        methods={},
    )
    for name, candidate in (("original", old), ("directional", new)):
        part = {
            key: candidate[key][START - FIRST : STOP - FIRST]
            for key in ("frame_idx", "pose_xyz", "sources", "observed")
        }
        geometry = support_audit.verify_support_partitioned(part, nodes, rig)
        result["methods"][name] = dict(
            full_minute=short.stats(candidate, START, STOP),
            diagnostic_intervals={f"{a}:{b}": short.stats(candidate, a, b) for a, b in TARGETS},
            gap_metrics=gap_metrics(candidate),
            female=female_scores(candidate, anchors),
            all_native_source_geometry=geometry,
        )
    output.mkdir(parents=True)
    with provenance("directional_one_minute_comparison", output, name="comparison") as record:
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
        for name, value in (("summary.json", result), ("checkpoint_audit.json", audits)):
            (output / name).write_text(
                json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
            )
        rows = []
        for t in range(START - FIRST, STOP - FIRST):
            row = dict(frame_idx=int(new["frame_idx"][t]))
            for name, candidate in (("original", old), ("directional", new)):
                for node, label in enumerate(("mouth", "chest", "tail")):
                    row[f"{name}_{label}_supported"] = int(candidate["observed"][t, :, node].sum())
                row[name + "_review_fish"] = int(candidate["identity_review"][t].sum())
                row[name + "_mask_error"] = float(candidate["loss_parts"][t, 4])
            rows.append(row)
        with (output / "frames.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        for path in output.iterdir():
            if path.is_file():
                record.output(path.name, path, sha256=checks.sha(path))
        for path, digest in hashes.items():
            checks.require(checks.sha(path) == digest, "evaluation input changed")
        record.param("all_bound_files_unchanged", True)
    print(
        json.dumps(
            {
                k: {j: v for j, v in value.items() if j != "gap_metrics"}
                for k, value in result["methods"].items()
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "branch", "female", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    evaluate(parser.parse_args())
