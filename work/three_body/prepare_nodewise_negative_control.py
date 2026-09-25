"""Extract the Sparse checkpoint backtrace ending at native 91300, without tracking."""

from __future__ import annotations

import argparse
from pathlib import Path

import compare_instance_rescore_10s as common
import numpy as np

proof, require = common.proof, common.require
WORK = common.WORK
PARENT = WORK / "sparse_instance_five_minute_continuation_v1"
PARENT_SHA = "596b23afe5fb572a66a72b96b0b1f5ba9b3971bfef4dd43cb8ee6b32459e32ad"
FIRST, START, STOP = 49800, 89640, 91300
STAGE = "nodewise_negative_same_horizon_control_v1"
CHECKPOINT_RELATIVE = "checkpoints/through_091300"
CHECKPOINT_SHA = "d1fb9abca83edd772841ac90740454d88370b121f6efc0524bb9187925437bf4"


def validate_control_metadata(metadata):
    require(
        metadata["schema"] == "persistent_tracker_full_state_v1"
        and metadata["context"]["implementation"] == "SparseInstanceRescoreContinuousTracker"
        and [
            metadata[k]
            for k in ("previous_frame", "processed_frames", "initialize_count", "restore_count")
        ]
        == [STOP - 1, STOP - FIRST, 1, 6],
        "control lifecycle or implementation differs",
    )


def run(output):
    output = Path(output).resolve()
    require(output.is_relative_to(WORK) and not output.exists(), "fresh output required")
    out, evidence, started = proof.Output(output), proof.Evidence(), proof.utc()
    try:
        evidence.add(PARENT / "tracking.provenance.json", PARENT_SHA)
        seal = proof.read_json(PARENT / "tracking.provenance.json")
        require(seal["status"] == "ok" and seal["stage"] == PARENT.name, "successful parent")
        root = PARENT / CHECKPOINT_RELATIVE
        for name in ("manifest.json", "metadata.json", "arrays.npz"):
            common.bound_member(PARENT, seal, CHECKPOINT_RELATIVE + "/" + name, evidence)
        require(proof.sha(root / "manifest.json") == CHECKPOINT_SHA, "wrong control checkpoint")
        manifest = proof.read_json(root / "manifest.json")
        arrays, metadata, verified = common.codec.read_checkpoint(
            root, expected_bindings=manifest["bindings"]
        )
        require(verified == manifest and len(arrays) == 39, "registered full checkpoint")
        validate_control_metadata(metadata)
        candidate = common.support.COMMON.traced_candidate(arrays)
        require(
            np.array_equal(candidate["frame_idx"], np.arange(FIRST, STOP)),
            "wrong terminal backtrace extent",
        )
        common.metrics.validate_candidate(candidate, FIRST, STOP)
        out.write("candidate.npz", candidate)
        out.write("checkpoint_metadata.json", metadata)
        code = {
            str(Path(__file__).resolve()): evidence.add(Path(__file__).resolve()),
            str(Path(common.__file__).resolve()): evidence.add(Path(common.__file__).resolve()),
            str(Path(common.support.COMMON.__file__).resolve()): evidence.add(
                Path(common.support.COMMON.__file__).resolve()
            ),
            str(Path(common.codec.__file__).resolve()): evidence.add(
                Path(common.codec.__file__).resolve()
            ),
        }
        out.write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                retained_interval=[FIRST, STOP],
                comparison_interval=[START, STOP],
                terminal_horizon=STOP,
                parent_tracking_sha256=PARENT_SHA,
                control_checkpoint_manifest_sha256=proof.sha(root / "manifest.json"),
                candidate_arrays=common.signature(candidate),
                source_code=code,
                original_binding=manifest["bindings"]["original_binding"],
                checkpoint_payloads_decoded=1,
                original_candidate_116200_used=False,
                tracking_performed=False,
                physical_ID_truth=False,
            ),
        )
        evidence.final_guard()
        out.final_guard()
        result = dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=out.records,
        )
    except BaseException as exc:
        result = dict(
            stage=STAGE,
            status="failed",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=out.records,
            error=repr(exc),
        )
        (output / "control.provenance.json").write_bytes(proof.json_bytes(result))
        raise
    (output / "control.provenance.json").write_bytes(proof.json_bytes(result))
    print("Completed control backtrace through " + str(STOP), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=WORK / STAGE)
    run(parser.parse_args().output)
