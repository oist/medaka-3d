"""CPU risk inventory of a sealed long backtrace; no raw decoding or ID repair.

Checks producer outputs and registered metadata, not the independent geometry /
history audit. Always use the final backtrace again after the whole run finishes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(REPO / "src"))

from review_events import extract_review_events  # noqa: E402
from review_plan import build_review_plan  # noqa: E402
from tracklet_ledger import build_tracklet_ledger  # noqa: E402

from medaka3d.provenance import provenance  # noqa: E402

FIRST, STOP, CHUNK = 49800, 348600, 1660
PROTOCOL = LAB / "long_tracking_protocol.json"
PROTOCOL_SHA = "fda72ac8304be4e46032853a13f895ddb9bf25f2436096b65e27e2418a619c85"
CANDIDATE_KEYS = {
    "frame_idx", "pose_xyz", "observed", "sources", "identity_review",
    "mask_supported", "body_supported", "missing", "costs", "path_costs",
    "loss_parts", "score_margin", "proposal_kind", "hypotheses",
    "proposal_flow_nodes", "proposal_flow_parent", "node_missing",
    "direct_pectoral_age", "xyz", "accepted_xyz", "body_xyz",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_metadata(seal, summary, metadata, consumed, protocol, frames):
    """Refuse partial/mismatched outputs before extracting model-selected risks."""
    require(seal.get("status") == "ok"
            and seal.get("stage") == "three_body_long_continuation",
            "successful long continuation required")
    params = seal["params"]
    require(params.get("all_input_source_snapshot_output_sha_unchanged") is True,
            "producer final guard missing")
    stop = params["through_frame"]
    require(type(stop) is int and FIRST < stop <= STOP and (stop-FIRST) % CHUNK == 0,
            "registered native prefix endpoint required")
    count = stop-FIRST
    for key, expected in {
        "schema": "long_continuation_phase_v1", "technical_completion": True,
        "full_sequence_complete": stop == STOP,
        "prefix_backtrace_provisional": stop != STOP,
        "registered_interval": [FIRST, STOP], "processed_interval": [FIRST, stop],
        "processed_total": count, "consumed_chunk_count": count//CHUNK,
        "logical_initialization_count": 1, "external_pose_reset_count": 0,
        "temporal_subsampling": False, "human_annotation_input": False,
        "post_initialization_baseline_coordinates_input": False, "active_beam_count": 6,
    }.items():
        require(type(summary.get(key)) is type(expected) and summary[key] == expected,
                "producer summary differs: " + key)
    require(frames.dtype == np.int64 and frames.shape == (count,)
            and np.array_equal(frames, np.arange(FIRST, stop)), "native frame sequence differs")
    require(protocol["source_frame_range"] == [FIRST, STOP] and protocol["fps"] == 166,
            "registered range/fps differs")
    config = protocol["tracker_config"]
    require(config["review_gap_frames"] == 4 and config["ambiguity_margin"] == 1.0,
            "registered risk policy differs")
    require(metadata.get("schema") == "persistent_tracker_full_state_v1"
            and metadata["context"]["tracker_config"] == config
            and metadata["initialize_count"] == 1
            and metadata["processed_frames"] == count and metadata["previous_frame"] == stop-1
            and type(metadata["restore_count"]) is int and metadata["restore_count"] >= 0
            and type(summary["checkpoint_restore_count"]) is int
            and metadata["restore_count"] == summary["checkpoint_restore_count"],
            "checkpoint metadata differs")
    require(len(consumed) == count//CHUNK, "consumed chunk count differs")
    for index, item in enumerate(consumed):
        require(item["index"] == index and item["start"] == FIRST+CHUNK*index
                and item["stop"] == FIRST+CHUNK*(index+1), "consumed native chunk order differs")
    return config


def load_run(root, bindings):
    root = Path(root).resolve(strict=True)
    require(root.is_relative_to(REPO), "run outside workspace")

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = sha(path)
        require(expected is None or expected == digest, "SHA differs: " + str(path))
        require(str(path) not in bindings or bindings[str(path)] == digest, "input changed")
        bindings[str(path)] = digest
        return path

    seal = read_json(bind(root / "continuation.provenance.json"))
    require(seal.get("status") == "ok"
            and seal.get("stage") == "three_body_long_continuation", "successful long run required")
    bound_paths = set()
    for item in seal["outputs"].values():
        path = Path(item["path"]).resolve(strict=True)
        require(path.is_relative_to(root) and path.stat().st_size == item["size"],
                "output escaped run or changed size")
        bind(path, item["sha256"])
        bound_paths.add(path)
    for role, filename in (("candidate", "candidate.npz"), ("summary", "summary.json"),
                           ("consumed_chunks", "consumed_chunks.json")):
        require(Path(seal["outputs"][role]["path"]).resolve() == root / filename,
                "output role differs: " + role)
    meta_path = root / "checkpoint/metadata.json"
    require(meta_path in bound_paths, "checkpoint metadata is not sealed")
    binding = seal["params"]["bindings"]
    require(binding["tracking_protocol_sha256"] == PROTOCOL_SHA, "unregistered tracker protocol")
    protocol = read_json(bind(PROTOCOL, PROTOCOL_SHA))
    for source, digest in binding["code_sha256"].items():
        path = Path(source).resolve(strict=True)
        require(path.is_relative_to(REPO), "producer source outside workspace")
        bind(path, digest)
        snapshot = root / "code_snapshot" / path.relative_to(REPO)
        require(snapshot in bound_paths and sha(snapshot) == digest, "producer snapshot differs")
    summary, metadata = read_json(root / "summary.json"), read_json(meta_path)
    consumed = read_json(root / "consumed_chunks.json")
    with np.load(root / "candidate.npz", allow_pickle=False) as saved:
        require(set(saved.files) == CANDIDATE_KEYS, "candidate schema differs")
        data = {key: saved[key] for key in saved.files}
    config = validate_metadata(seal, summary, metadata, consumed, protocol, data["frame_idx"])
    for array in data.values():
        array.flags.writeable = False
    return data, summary, config


def make_reports(data, producer_summary, config):
    inventory = extract_review_events(data, review_gap_frames=config["review_gap_frames"])
    ledger = build_tracklet_ledger(data, tracker_config=config)
    proposal = build_review_plan(data)
    require(inventory["counts"] == proposal["risk_counts"] == ledger["frozen_inventory_counts"],
            "pure report risk counts differ")
    observed, review = data["observed"], data["identity_review"]
    direct, accepted = observed[:, :, 1], observed[:, :, 1] & ~review
    for key, actual in {
        "direct_pectoral_count": int(direct.sum()),
        "accepted_pectoral_count": int(accepted.sum()),
        "identity_review_count": int(review.sum()),
        "per_slot_direct_pectoral": direct.sum(0).tolist(),
        "per_slot_accepted_pectoral": accepted.sum(0).tolist(),
    }.items():
        require(producer_summary[key] == actual, "candidate/producer count differs: " + key)
    scenes = proposal["scenes"]
    summary = {
        "schema": "long_backtrace_review_inventory_v1",
        "processed_interval": producer_summary["processed_interval"],
        "native_frames": len(data["frame_idx"]),
        "source_seconds": len(data["frame_idx"])/166,
        "full_sequence_complete": producer_summary["full_sequence_complete"],
        "prefix_backtrace_provisional": producer_summary["prefix_backtrace_provisional"],
        "risk_counts": inventory["counts"], "ledger_counts": ledger["counts"],
        "support": inventory["support"], "margin": inventory["margin"],
        "proposed_scene_count": proposal["proposed_scene_count"],
        "incomplete_scene_count": proposal["incomplete_scene_count"],
        "missing_probes": sum(p is None for s in scenes for p in s["proposed_anchor_frames"]),
        "scenes_with_unresolved_entry": sum(
            s["entry_contains_unresolved_machine_id"] is True for s in scenes),
        "context_playback_seconds_at_20fps_includes_overlaps":
            inventory["counts"]["scene_context_frames"]/20,
        "human_confirmed_links": 0, "identities_repaired": False,
        "all_three_identity_accuracy_measured": False,
        "verification_scope": (
            "producer output hashes, config/metadata and saved-array risk extraction"),
        "limitations": [
            "This inventory does not repeat the independent history/geometry/raw-image audit.",
            "Consumed chunk order is checked; raw videos and packed detections are not reopened.",
            "All risks are retained, including missing probes and unresolved entries.",
            "No scenes is not proof of no swaps; unflagged errors need independent review.",
            "Scene duration is playback length, not measured human time or correction burden.",
            "A provisional prefix can change; rerun this inventory on the final full backtrace.",
            "Initial anonymous slots derive from the optimistic C34 seed, not biological truth.",
        ],
    }
    return {"inventory": inventory, "ledger": ledger,
            "review_proposal": proposal, "summary": summary}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    require(output.is_relative_to(REPO) and not output.exists(), "new workspace output required")
    bindings = {}
    code = [Path(__file__).resolve(), LAB / "review_events.py", LAB / "review_plan.py",
            LAB / "tracklet_ledger.py", REPO / "src/medaka3d/provenance.py"]
    for path in code:
        bindings[str(path)] = sha(path)
    data, producer_summary, config = load_run(args.run, bindings)
    reports = make_reports(data, producer_summary, config)
    reports["summary"]["candidate_sha256"] = bindings[str(args.run.resolve() / "candidate.npz")]
    reports["summary"]["producer_seal_sha256"] = bindings[
        str(args.run.resolve() / "continuation.provenance.json")]
    with provenance("long_saved_backtrace_review_inventory", output, name="inventory") as record:
        for path, digest in bindings.items():
            record.input(path, path, sha256=digest)
        for path in code:
            target = output / "source_snapshot" / path.relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            require(sha(target) == bindings[str(path)], "inventory source snapshot differs")
            record.output(str(target), target, sha256=sha(target))
        for name, value in reports.items():
            target = output / (name + ".json")
            target.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
            record.output(name, target, sha256=sha(target))
        for path, digest in bindings.items():
            require(sha(path) == digest, "input/source changed while saving: " + path)
        for item in record.outputs.values():
            require(sha(item["path"]) == item["sha256"], "saved output changed")
        record.params["all_bound_input_source_output_sha_unchanged"] = True
    print(json.dumps(reports["summary"], allow_nan=False))
    return reports


if __name__ == "__main__":
    main()
