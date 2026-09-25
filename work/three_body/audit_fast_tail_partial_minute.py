"""Qualify the committed minute of an interrupted run, without claiming two minutes.

The missing process handle and missing whole-run seal do not certify completion.
The saved final checkpoint is independently decoded and compared to the existing
successful Tail minute, including all39 fields and the original21 backtrace.
This audit performs no tracking, fit, GPU initialization or source image decode.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import compare_temporal_point64_v2 as proof
import numpy as np
import run_fast_tail_two_minutes as prior

WORK, REPO, require = prior.WORK, prior.REPO, proof.require
PARENT = WORK / "fast_tail_two_minute_continuous_v1"
CHECKPOINT = PARENT / "checkpoints/through_076360"
CHECKPOINT_SHA = "cef133c3e6192e373df095a3ec8b3b1cd546363a3aed8a6bf8ead670ff529cf6"
PRIOR_SHA = "8efbba7abb6cdedd0b077b8dc953a7a1d10c268b8aab7844b17f25cc337862af"
STAGE = "fast_tail_interrupted_minute_checkpoint_audit_v1"


def load_and_verify(evidence):
    evidence.add(Path(prior.__file__), PRIOR_SHA)
    evidence.add(CHECKPOINT / "manifest.json", CHECKPOINT_SHA)
    manifest = proof.read_json(CHECKPOINT / "manifest.json")
    binding = manifest["bindings"]
    require(
        binding["schema"] == "fast_tail_two_minute_checkpoint_binding_v1"
        and binding["native_interval"] == [66400, 86320],
        "fixed interrupted parent required",
    )
    for name in ("metadata", "payload"):
        evidence.add(CHECKPOINT / manifest[name]["file"], manifest[name]["sha256"])
    arrays, metadata, _ = prior.read_checkpoint(CHECKPOINT, expected_bindings=binding)
    require(
        len(arrays) == 39
        and metadata["previous_frame"] == 76359
        and metadata["processed_frames"] == 26560
        and metadata["initialize_count"] == 1
        and metadata["restore_count"] == 4,
        "complete one-minute state required",
    )
    for path, digest in binding["experimental_code_sha256"].items():
        evidence.add(path, digest)
        evidence.add(PARENT / "code_snapshot" / Path(path).relative_to(REPO), digest)
    migration_path = PARENT / "state_import.json"
    evidence.add(migration_path, binding["state_import_sha256"])
    migration = proof.read_json(migration_path)
    controls = prior.metadata_inputs(evidence, prior.local_code())
    expected = prior.load_control(76360, controls, evidence)
    comparison = prior.compare_checkpoint(arrays, metadata, expected, migration)
    require(comparison["exact"], "all39 state/metadata differs from successful Tail minute")
    traced = prior.original.phase.COMMON.traced_candidate(arrays)
    old_path = prior.consumed(controls["tail"], "candidate.npz", evidence)
    with np.load(old_path, allow_pickle=False) as z:
        old_candidate = {k: z[k].copy() for k in z.files}
    candidate_proof = prior.compare_candidate(traced, old_candidate, 76360)
    require(candidate_proof["exact"], "saved history backtrace differs from qualified minute")
    saved_path = PARENT / "candidate_through_076360.npz"
    evidence.add(saved_path)
    with np.load(saved_path, allow_pickle=False) as z:
        saved = {k: z[k].copy() for k in z.files}
    require(
        prior.compare_candidate(traced, saved, 76360)["exact"], "partial saved candidate differs"
    )
    evidence.add(PARENT / "history_comparison.json")
    rows = proof.read_json(PARENT / "history_comparison.json")
    require(
        [r["frame_idx"] for r in rows] == list(range(66400, 76360)),
        "committed history does not cover exactly the first9960 native frames",
    )
    require(
        all(r["control"] is not None and r["control"]["exact"] for r in rows),
        "a per-frame original ordered-beam comparison failed",
    )
    return (
        arrays,
        metadata,
        binding,
        dict(
            schema=STAGE,
            status="ok",
            committed_interval=[66400, 76360],
            intended_but_incomplete_interval=[66400, 86320],
            original_run_completed=False,
            original_process_exit_code_known=False,
            checkpoint_manifest_sha256=CHECKPOINT_SHA,
            checkpoint_all39_comparison=comparison,
            candidate_all21_comparison=candidate_proof,
            ordered_beam_comparison_frames=len(rows),
            initializes=1,
            restore_count=4,
            source_code_unchanged=True,
            GPU_initialized=False,
            new_tracking_frames=0,
            eligible_for_exact_state_restore=True,
            physical_ID_truth=False,
        ),
    )


def run(output):
    output = output.resolve()
    require(output.is_relative_to(WORK) and not output.exists(), "new audit output required")
    evidence, saved = proof.Evidence(), proof.Output(output)
    started = proof.utc()
    try:
        code = prior.local_code()
        for path, digest in code.items():
            evidence.add(path, digest)
        _, _, _, summary = load_and_verify(evidence)
        require(not prior.torch.cuda.is_initialized(), "audit initialized CUDA")
        saved.write("summary.json", summary)
        require(prior.local_code() == code, "audit loaded source changed")
        evidence.final_guard()
        saved.final_guard()
        seal = dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=saved.records,
        )
    except BaseException as error:
        seal = dict(
            stage=STAGE,
            status="failed",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=saved.records,
            error=repr(error),
        )
        with (output / "audit.provenance.json").open("xb") as stream:
            stream.write(proof.json_bytes(seal))
        raise
    with (output / "audit.provenance.json").open("xb") as stream:
        stream.write(proof.json_bytes(seal))
    print(
        proof.json_bytes(
            {
                k: summary[k]
                for k in (
                    "status",
                    "committed_interval",
                    "ordered_beam_comparison_frames",
                    "eligible_for_exact_state_restore",
                    "physical_ID_truth",
                )
            }
        ).decode(),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
