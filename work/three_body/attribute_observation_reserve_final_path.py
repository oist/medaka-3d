"""Attribute reserved admissions to the completed76360 parent backtrace.

Only four checkpoint members are decompressed, once each. Costs are a storage
QA check; final state zero is the frozen finish rule, not a new optimization.
No candidate poses, diagnostic batches, masks, SLEAP, video or GPU are read.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import prepare_top_background_5to6_minutes as proof

REPO, WORK, require = proof.REPO, proof.WORK, proof.require
sys.path.insert(0, str(REPO / "labs/20260909_three_body"))
import tracker_checkpoint as codec  # noqa: E402

STAGE = "observation_reserve_final_path_attribution_v1"
TRACKING_STAGE = "observation_reserve_ten_second_continuation_v1"
TRACKING = WORK / TRACKING_STAGE
FIRST, START, STOP = 49800, 74700, 76360
CHECKPOINT = "checkpoints/through_076360"
PARENT_SHA = "7c9672b6fde9f96ee5d23efe7dbbd6e7f88d48299d81e86e00a87fcca5c5a076"
FIELDS = ("history_offsets", "history_parent", "history_frame_idx", "state_cost")
FAMILIES = ["instance", "body_row", "observation_reserve"]
PINS = {
    "work/three_body/run_observation_reserve_continuation.py": (
        "f2ad6b4039da5d14244cb2f0052003d3003903146a9ae573f362fc14aee437bc"
    ),
    "work/three_body/observation_reserve_tracker.py": (
        "7b03698dacda5d18be87ff99c0bccd47474a94db08ae89aff0231f4dcb50c74a"
    ),
    "work/three_body/observation_reserve_diagnostic_batch.py": (
        "f04cd7ef660b71f77672022a1889600555d80793de74b2225804c25c989ea60d"
    ),
    "work/three_body/observation_reserve_selection.py": (
        "b4b69b67e488db00fdff49b278a0c39f3966cc24e55749e2b1ca8251d1196853"
    ),
    "labs/20260909_three_body/continuous_tracker.py": (
        "91a811cab27a8a26e605aa00159fdb2a2b450bbcca1a2505e49543f58ef7c7e2"
    ),
    "labs/20260909_three_body/tracker_checkpoint.py": (
        "39272a664d3f2056d4d876b75e9241ae1a775d19f7352a0423a3e1c76125a32b"
    ),
}
PROOF_SHA = "3c4bd35a83c192cc61082ef783fe5d3859438943d2ea23f62b4b25b6e8b6175c"


def member(seal, name):
    ref = seal["outputs"][name]
    require(Path(ref["path"]).resolve() == TRACKING / name, "producer output path differs")
    return ref


def load_headers(tracking_sha, evidence):
    require(
        isinstance(tracking_sha, str)
        and len(tracking_sha) == 64
        and all(c in "0123456789abcdef" for c in tracking_sha),
        "explicit successful tracking SHA required",
    )
    seal = evidence.read(TRACKING / "tracking.provenance.json", tracking_sha)
    require(
        seal["stage"] == TRACKING_STAGE and seal["status"] == "ok", "successful producer required"
    )

    def document(name):
        ref = member(seal, name)
        data = evidence.capture(ref["path"], ref["sha256"])
        require(len(data) == ref["size_bytes"], "small producer output size differs")
        return json.loads(data)

    plan, summary, index = (
        document(name + ".json") for name in ("plan", "summary", "diagnostic_index")
    )
    manifest = document(CHECKPOINT + "/manifest.json")
    metadata = document(CHECKPOINT + "/metadata.json")
    payload = member(seal, CHECKPOINT + "/arrays.npz")
    for doc in (plan, summary):
        require(
            doc["schema"] == TRACKING_STAGE
            and doc["process_interval"] == doc["scientific_interval"] == [START, STOP]
            and doc["retained_interval"] == [FIRST, STOP]
            and doc["parent_checkpoint_sha256"] == PARENT_SHA
            and doc["initialize_count"] == 1
            and doc["restore_count"] == 5
            and doc["new_begin_calls"] == doc["external_pose_resets"] == 0,
            "completed same-parent76360 scope/lifecycle differs",
        )
    require(
        summary["status"] == "ok"
        and summary["finish_calls"] == summary["checkpoint_count"] == 1
        and summary["native_frames"]
        == summary["process_frames"]
        == summary["diagnostic_frames"]
        == STOP - START
        and all(
            summary[key] is True
            for key in (
                "all39_roundtrip",
                "completed_checkpoint_decoder_accepted",
                "prior_history_prefix_unchanged",
                "finish_matches_checkpoint_backtrace",
            )
        ),
        "completed producer backtrace qualification required",
    )
    require(
        manifest["schema"] == codec.SCHEMA
        and manifest["status"] == "ok"
        and manifest["bindings"] == plan["checkpoint_binding"]
        and manifest["bindings"]["source_sha256"]
        == plan["source_sha256"]
        == summary["source_sha256"]
        and manifest["bindings"]["tracker_context"]
        == plan["tracker_context"]
        == summary["tracker_context"]
        == metadata["context"]
        and summary["checkpoint_manifest_sha256"]
        == member(seal, CHECKPOINT + "/manifest.json")["sha256"]
        and Path(summary["checkpoint"]["path"]).resolve() == TRACKING / CHECKPOINT,
        "final manifest, source and context binding differ",
    )
    for path, digest in PINS.items():
        absolute = str((REPO / path).resolve())
        require(
            plan["source_sha256"].get(absolute) == seal["inputs"].get(absolute) == digest,
            "executed producer source differs: " + path,
        )
    for field, name in (("metadata", "metadata.json"), ("payload", "arrays.npz")):
        ref = member(seal, CHECKPOINT + "/" + name)
        require(
            manifest[field] == dict(file=name, sha256=ref["sha256"], bytes=ref["size_bytes"]),
            "checkpoint member edge differs",
        )
    require(
        metadata["schema"] == "persistent_tracker_full_state_v1"
        and metadata["previous_frame"] == STOP - 1
        and metadata["processed_frames"] == STOP - FIRST
        and metadata["initialize_count"] == 1
        and metadata["restore_count"] == 5
        and metadata["context"]["observation_reserve_policy"]["effective_from_native"] == START
        and len(manifest["arrays"]) == 39,
        "final checkpoint metadata extent differs",
    )
    require(
        index["process_interval"] == [START, STOP]
        and index["batch_size"] == 100
        and index["diagnostic_families"] == FAMILIES
        and [row["frame_idx"] for row in index["rows"]] == list(range(START, STOP)),
        "complete three-family1660 index required",
    )
    next_frame = START
    for batch in index["batches"]:
        stop = min(next_frame + 100, STOP)
        require(
            batch["first_frame"] == next_frame
            and batch["stop_frame_exclusive"] == stop
            and batch["frame_count"] == stop - next_frame
            and batch["full_codec_roundtrip_verified"] is True,
            "diagnostic batch coverage differs",
        )
        for key, suffix in (("archive", ".npz"), ("metadata", ".json")):
            name = f"diagnostic_batches/f{next_frame:06d}-{stop:06d}{suffix}"
            require(batch[key] == member(seal, name), "sealed diagnostic index edge differs")
        next_frame = stop
    require(
        next_frame == STOP and len(index["batches"]) == summary["diagnostic_batch_count"],
        "batch tail missing",
    )
    return dict(manifest=manifest, metadata=metadata, payload=payload, index=index, summary=summary)


def load_four_fields(manifest, payload):
    """Verify selected logical arrays by manifest descriptor; never call read_checkpoint."""
    path = Path(payload["path"])
    before = proof.stamp(path)
    require(before[2] == payload["size_bytes"], "checkpoint archive size differs")
    descriptions = manifest["arrays"]
    require(
        len(descriptions) == 39 and all(key in descriptions for key in FIELDS),
        "complete manifest inventory required",
    )
    expected_members = []
    for i, (_name, ref) in enumerate(sorted(descriptions.items())):
        require(ref["storage_key"] == f"array_{i:06d}", "checkpoint storage key differs")
        expected_members.append(ref["storage_key"] + ".npy")
    arrays = {}
    with zipfile.ZipFile(path) as archive:
        require(sorted(archive.namelist()) == expected_members, "checkpoint ZIP membership differs")
        for key in FIELDS:
            ref = descriptions[key]
            # One decompression per selected member. All other35 remain unread.
            with archive.open(ref["storage_key"] + ".npy") as stream:
                value = np.lib.format.read_array(stream, allow_pickle=False)
                require(stream.read(1) == b"", "unexpected trailing selected NPY bytes")
            require(
                codec._array_record(value, ref["storage_key"]) == ref,
                "selected array signature differs: " + key,
            )
            arrays[key] = value
    require(before == proof.stamp(path), "checkpoint changed during selected reads")
    return arrays, dict(
        archive_reference=payload,
        archive_stat=before,
        logical_fields=list(FIELDS),
        selected_fields={key: descriptions[key] for key in FIELDS},
        decompressed_members=4,
        unselected_members_not_decoded=35,
        zip_member_inventory_verified=True,
        current_container_size_verified=True,
        current_full_archive_sha_recomputed=False,
        validation=(
            "sealed manifest descriptors plus each selected array full logical "
            "byte signature and archive stat"
        ),
    )


def attribute(arrays, metadata, index):
    offsets, parents, native, terminal_cost = (arrays[key] for key in FIELDS)
    count = metadata["processed_frames"]
    first = metadata["previous_frame"] - count + 1
    require(first == FIRST and count == STOP - FIRST, "native history origin differs")
    require(
        offsets.dtype == np.int64
        and offsets.shape == (count + 1,)
        and offsets[0] == 0
        and np.all((np.diff(offsets) >= 1) & (np.diff(offsets) <= 6))
        and parents.dtype == native.dtype == np.int64
        and parents.shape == native.shape == (int(offsets[-1]),)
        and np.array_equal(native, np.repeat(np.arange(FIRST, STOP), np.diff(offsets))),
        "history offset/native arrays differ",
    )
    sizes = np.diff(offsets)
    require(
        terminal_cost.dtype == np.float64
        and terminal_cost.shape == (int(sizes[-1]),)
        and np.isfinite(terminal_cost).all()
        and np.all(np.diff(terminal_cost) >= 0),
        "terminal costs must match the stored sorted beam",
    )
    rows = index["rows"]
    require(
        [row["frame_idx"] for row in rows] == list(range(START, STOP)),
        "native index coverage differs",
    )
    changes, reasons = [], {}
    for frame, row in zip(range(START, STOP), rows, strict=True):
        t = frame - FIRST
        block = parents[offsets[t] : offsets[t + 1]]
        require(
            np.all((block >= 0) & (block < sizes[t - 1])), "history parent outside preceding beam"
        )
        reserve, candidate_count = row["observation_reserve"], row["candidate_count"]
        baseline, selected = reserve["baseline_indices"], reserve["selected_indices"]
        require(
            len(baseline) == len(selected) == sizes[t]
            and len(set(baseline)) == len(baseline)
            and len(set(selected)) == len(selected)
            and all(type(i) is int and 0 <= i < candidate_count for i in baseline + selected)
            and selected[:-1] == baseline[:-1]
            and selected[0] == baseline[0] == reserve["baseline_best_index"]
            and type(reserve["changed"]) is bool
            and reserve["changed"] == (baseline != selected)
            and reserve["reservation_count"] == int(reserve["changed"])
            and reserve["support_evaluation_count"] == candidate_count,
            "saved selection/index disagrees with history beam",
        )
        replacement = reserve["replacement"]
        if reserve["changed"]:
            require(
                len(selected) == 6
                and replacement["admitted"] == selected[-1]
                and replacement["removed"] == baseline[-1]
                and np.isfinite(replacement["extra_path_cost"])
                and replacement["extra_path_cost"] >= 0,
                "reservation replacement disagrees with last slot",
            )
            changes.append(frame)
        else:
            require(replacement is None, "unchanged beam has a replacement")
        reasons[reserve["reason"]] = reasons.get(reserve["reason"], 0) + 1
    slots, flat_rows, candidate_indices, adopted, adopted_details = [], [], [], [], []
    slot = 0  # Exact frozen finish semantics, including ties; costs are QA only.
    for frame in range(STOP - 1, START - 1, -1):
        t, row = frame - FIRST, rows[frame - START]
        require(0 <= slot < sizes[t], "backtrace slot outside current beam")
        flat = int(offsets[t]) + slot
        reserve = row["observation_reserve"]
        candidate = reserve["selected_indices"][slot]
        hit = reserve["changed"] and candidate == reserve["replacement"]["admitted"]
        if hit:
            require(slot == 5, "reserved final ancestor must occupy slot five")
            adopted_details.append(
                dict(
                    frame_idx=frame,
                    beam_slot=slot,
                    candidate_index=candidate,
                    **reserve["replacement"],
                )
            )
        slots.append(slot)
        flat_rows.append(flat)
        candidate_indices.append(candidate)
        adopted.append(hit)
        slot = int(parents[flat])
    for value in (slots, flat_rows, candidate_indices, adopted, adopted_details):
        value.reverse()
    return dict(
        frame_idx=list(range(START, STOP)),
        final_beam_slot=slots,
        final_history_flat_row=flat_rows,
        final_fitted_candidate_index=candidate_indices,
        final_path_uses_reserved_admission=adopted,
        reserved_final_path_frames=adopted_details,
        current_beam_changed_native_frames=changes,
        current_beam_reasons=reasons,
        inherited_parent_slot_at_74699=slot,
        terminal_state_zero_cost=float(terminal_cost[0]),
    )


def run(tracking_sha, output):
    output = Path(output).resolve()
    require(
        output.is_relative_to(WORK) and output != WORK and not output.exists(),
        "fresh workspace output required",
    )
    output.mkdir(parents=True)
    evidence, outputs, started = proof.Evidence(), {}, proof.utc()

    def write(name, value):
        data = (
            value
            if isinstance(value, bytes)
            else (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
        )
        path = output / name
        with path.open("xb") as stream:
            stream.write(data)
        outputs[name] = dict(path=str(path), sha256=proof.sha(data), size_bytes=len(data))

    try:
        for path, digest in PINS.items():
            evidence.capture(REPO / path, digest)
        evidence.capture(Path(proof.__file__), PROOF_SHA)
        write("source_snapshot.py", evidence.capture(Path(__file__)))
        context = load_headers(tracking_sha, evidence)
        write(
            "plan.json",
            dict(
                schema=STAGE,
                tracking_sha256=tracking_sha,
                source_pins=PINS,
                process_interval=[START, STOP],
                retained_interval=[FIRST, STOP],
                checkpoint_reference=context["payload"],
                logical_fields=list(FIELDS),
                terminal_rule="frozen finish uses state zero; no new argmin/tie resolution",
            ),
        )
        arrays, array_evidence = load_four_fields(context["manifest"], context["payload"])
        result = attribute(arrays, context["metadata"], context["index"])
        reported = context["summary"]["observation_reserve"]
        require(
            reported["changed_frames"] == len(result["current_beam_changed_native_frames"])
            and reported["reasons"] == result["current_beam_reasons"]
            and reported["current_beam_selection_only"] is True
            and reported["final_path_reserved_ancestor_count_not_computed"] is True,
            "producer current-beam summary differs",
        )
        write("attribution.json", dict(schema=STAGE + "_attribution", **result))
        target = result["frame_idx"].index(75053)
        summary = dict(
            schema=STAGE,
            status="ok",
            process_interval=[START, STOP],
            native_frames=STOP - START,
            current_beam_reservation_count=len(result["current_beam_changed_native_frames"]),
            final_path_reserved_admission_count=len(result["reserved_final_path_frames"]),
            final_path_reserved_native_frames=[
                row["frame_idx"] for row in result["reserved_final_path_frames"]
            ],
            native75053=dict(
                final_beam_slot=result["final_beam_slot"][target],
                final_fitted_candidate_index=result["final_fitted_candidate_index"][target],
                uses_reserved_admission=result["final_path_uses_reserved_admission"][target],
                current_beam_changed=75053 in result["current_beam_changed_native_frames"],
            ),
            checkpoint_array_evidence=array_evidence,
            current_beam_reservation_is_not_final_path_adoption=True,
            zero_final_admissions_does_not_prove_no_effect=True,
            limitation=(
                "Removing other sixth-slot candidates can alter later search even "
                "when no reserved admission is on the final path."
            ),
            geometry_support_quality_ID_or_global_optimum_claimed=False,
            original_candidate_pose_diagnostic_batch_SL_mask_video_GPU_reads=0,
            full39_decoder_calls=0,
        )
        write("summary.json", summary)
        evidence.final_guard()
        require(
            proof.stamp(context["payload"]["path"]) == array_evidence["archive_stat"],
            "checkpoint changed after attribution",
        )
        seal = dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.records,
            selected_checkpoint_evidence=array_evidence,
            outputs=dict(outputs),
        )
    except BaseException as error:
        write(
            "attribution.provenance.json",
            dict(
                stage=STAGE,
                status="failed",
                started_at=started,
                finished_at=proof.utc(),
                inputs=evidence.records,
                outputs=dict(outputs),
                error_type=type(error).__name__,
                error=str(error),
            ),
        )
        raise
    write("attribution.provenance.json", seal)
    print(
        json.dumps({k: v for k, v in summary.items() if k != "checkpoint_array_evidence"}, indent=2)
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-sha", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.tracking_sha, args.output)
