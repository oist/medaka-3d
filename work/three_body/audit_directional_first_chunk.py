"""Read an already committed first checkpoint while the minute run continues.

Compare the completed 306-frame experiment with the same retained beam history,
then compare original/directional first-ten-second winners at the same horizon.
This audit never marks the parent minute run complete and never alters it.
"""

import argparse
import json
from pathlib import Path

import compare_directional_one_minute as comparison
import numpy as np

from medaka3d.provenance import provenance

checks, REPO = comparison.checks, comparison.REPO
FIRST, START, STOP, SHORT_STOP = comparison.FIRST, comparison.START, 68060, 66706


def prefix_trace(arrays, stop):
    n = stop - FIRST
    offsets = arrays["history_offsets"][: n + 1]
    checks.require(len(offsets) == n + 1, "checkpoint prefix unavailable")
    selected = {"history_offsets": offsets}
    selected.update(
        {
            "history_" + key: arrays["history_" + key][: offsets[-1]]
            for key in comparison.RECORD_FIELDS
        }
    )
    return comparison.short.backtrace(selected)


def main(args):
    branch, short_root, original, output = (
        getattr(args, k).resolve() for k in ("branch", "short", "original", "output")
    )
    checks.require(
        output.is_relative_to(REPO / "work") and not output.exists(), "new output required"
    )
    hashes = comparison.source_registry.local_code()
    progress = checks.read(branch / "checkpoint_progress.json")
    checks.require(len(progress) >= 1, "first checkpoint not committed")
    row = progress[0]
    checkpoint = branch / "checkpoints/through_068060"
    checks.require(
        Path(row["path"]).resolve() == checkpoint
        and row["manifest_sha256"] == checks.sha(checkpoint / "manifest.json"),
        "first committed checkpoint digest/path differs",
    )
    # The progress file will be updated by its owner. Bind an immutable copy of
    # the first record, and its checkpoint, not that mutable whole file.
    for name in ("manifest.json", "metadata.json", "arrays.npz"):
        hashes[str(checkpoint / name)] = checks.sha(checkpoint / name)
    fixed = checks.read(checkpoint / "manifest.json")["bindings"]
    checks.require(
        fixed["schema"] == "directional_one_minute_checkpoint_binding_v1"
        and fixed["native_interval"] == [START, comparison.STOP],
        "wrong branch binding",
    )
    for path, digest in fixed["experimental_code_sha256"].items():
        source = Path(path).resolve()
        target = branch / "code_snapshot" / source.relative_to(REPO)
        checks.require(
            checks.sha(source) == digest and checks.sha(target) == digest,
            "snapshot or live source changed",
        )
        hashes[str(source)] = hashes[str(target)] = digest
    migration = branch / "state_import.json"
    checks.require(checks.sha(migration) == fixed["state_import_sha256"], "state import changed")
    hashes[str(migration)] = fixed["state_import_sha256"]
    state, metadata, _ = comparison.load_bound_checkpoint(checkpoint, fixed, hashes)
    comparison.validate_lifecycle(metadata, STOP, 4)
    checks.verify(
        short_root, "replay.provenance.json", "directional_chest_continuous_replay", hashes
    )
    short_binding = checks.read(short_root / "checkpoint/manifest.json")["bindings"]
    short_state, short_meta, _ = comparison.load_bound_checkpoint(
        short_root / "checkpoint", short_binding, hashes
    )
    comparison.validate_lifecycle(short_meta, SHORT_STOP, 4)
    audit_rows = []
    for name in ("history_offsets", *("history_" + k for k in comparison.RECORD_FIELDS)):
        prior = short_state[name]
        current = state[name][: len(prior)]
        exact = comparison.same(prior, current)
        floating = prior.dtype.kind == current.dtype.kind == "f"
        close = (
            bool(np.allclose(prior, current, atol=1e-10, rtol=0, equal_nan=True))
            if floating
            else exact
        )
        finite = np.isfinite(prior) & np.isfinite(current) if floating else None
        maximum = (
            float(np.max(np.abs(prior[finite] - current[finite])))
            if floating and finite.any()
            else None
        )
        audit_rows.append(
            dict(
                array=name,
                exact=exact,
                within_fixed_1e_10=close,
                maximum_absolute_difference=maximum,
            )
        )
    old_seal = checks.verify(
        original, "continuation.provenance.json", "three_body_long_continuation", hashes
    )
    binding = old_seal["params"]["bindings"]
    checks.require(fixed["original_binding"] == binding, "different original inputs")
    old_state, old_meta, _ = comparison.load_bound_checkpoint(
        original / "checkpoint", binding, hashes
    )
    comparison.validate_lifecycle(old_meta, comparison.STOP, 4)
    parent_root = Path(old_seal["params"]["resume_phase"]).resolve()
    checks.verify(
        parent_root, "continuation.provenance.json", "three_body_long_continuation", hashes
    )
    parent, parent_meta, _ = comparison.load_bound_checkpoint(
        parent_root / "checkpoint", binding, hashes
    )
    comparison.validate_lifecycle(parent_meta, START, 3)
    imported = checks.read(migration)
    comparison.verify_import(imported, parent, parent_meta)
    checks.require(
        imported["new_context"] == metadata["context"], "first checkpoint context changed"
    )
    comparison.verify_history_prefix(parent, old_state)
    comparison.verify_history_prefix(parent, state)
    traces = dict(original=prefix_trace(old_state, STOP), directional=prefix_trace(state, STOP))
    result = dict(
        schema="directional_first_chunk_interim_audit_v1",
        status="ok",
        completed_checkpoint_interval=[START, STOP],
        complete_minute=False,
        parent_run_may_still_be_running=True,
        checkpoint_restore_count=4,
        logical_initialization_count=1,
        external_pose_resets=0,
        short_retained_history_all_arrays_exact=all(r["exact"] for r in audit_rows),
        short_retained_history_within_fixed_tolerance=all(
            r["within_fixed_1e_10"] for r in audit_rows
        ),
        compared_history_rows=int(short_state["history_offsets"][-1]),
        short_new_native_frames=306,
        source_snapshot_and_committed_checkpoint_verified=True,
        independent_ID_truth=False,
        methods={},
    )
    for name, candidate in traces.items():
        result["methods"][name] = dict(
            ten_seconds=comparison.short.stats(candidate, START, STOP),
            target50=comparison.short.stats(candidate, 66656, 66706),
            gap_metrics=comparison.gap_metrics(candidate, START, STOP),
        )
    output.mkdir(parents=True)
    with provenance("directional_first_chunk_interim_audit", output, name="audit") as record:
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
        for name, value in (
            ("summary.json", result),
            ("short_history_comparison.json", audit_rows),
            ("committed_checkpoint_record.json", row),
        ):
            (output / name).write_text(
                json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
            )
        for path in output.iterdir():
            if path.is_file():
                record.output(path.name, path, sha256=checks.sha(path))
        for path, digest in hashes.items():
            checks.require(checks.sha(path) == digest, "bound checkpoint/source changed")
    print(json.dumps({k: v for k, v in result.items() if k != "methods"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("branch", "short", "original", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    main(parser.parse_args())
