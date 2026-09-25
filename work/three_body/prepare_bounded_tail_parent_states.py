"""Fixed301 pre-frame state evidence from a sealed bounded68060 checkpoint.

No raw/SL/flow reader, proposal generation, fit, advance, restore or GPU calls.
The whole1660 retained suffix is traversed once to recover exact velocities.
Only {66524} union [66620,66920) previous parents and current retained rows are
exported. Retained rows do not reveal discarded candidates or pre-fit plans.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "labs/20260909_three_body"), str(REPO / "src")]

import audit_bounded_first_chunk as interim
import numpy as np
import recompute_directional_mouth_generation as old
from continuous_tracker import RECORD_FIELDS, STATE_FIELDS
from directional_chest_tracker import array_signature

from medaka3d.provenance import provenance

checks, comparison = interim.checks, interim.comparison
FIRST, START, HORIZON = 49800, 66400, 68060
FRAMES = np.r_[np.array([66524], np.int64), np.arange(66620, 66920, dtype=np.int64)]
STAGE = "bounded_tail_parent_states_301_v1"
WORK = REPO / "work/three_body"
AUDIT = WORK / "bounded_mouth_first_chunk_interim_audit_v1"
BRANCH = WORK / "bounded_mouth_one_minute_66400-76360_v1"
AUDIT_SHA = "08a7440cb5c851c4a97d98d1d67f665f6762a89c524c2ba87bfa7c0245656f04"
MANIFEST_SHA = "603490402bbaa6ea99a1694ae3e81ade77ba7b4dfa2f1ff7dec5b16605ba9048"
PINS = {
    Path(
        interim.__file__
    ).resolve(): "7be951a77f39e8f2dea5e6cf2aeae1a10663036ede0e131a1462beb67c45759e",
    Path(
        old.__file__
    ).resolve(): "a432b1c8ac4703040dc27decd0eb7880de4de26e0ac095a1e85640a85ae3abe3",
    REPO / "labs/20260909_three_body/continuous_tracker.py": (
        "91a811cab27a8a26e605aa00159fdb2a2b450bbcca1a2505e49543f58ef7c7e2"
    ),
}


def reconstruct_selected(parent, state, metadata, frames):
    """Frozen original recurrence; only capture membership/retained fields differ."""
    frames = np.asarray(frames)
    checks.require(
        frames.dtype == np.int64
        and frames.ndim == 1
        and len(frames) > 0
        and np.all(np.diff(frames) > 0)
        and frames[0] >= START
        and frames[-1] < HORIZON,
        "ordered unique int64 native selection inside suffix required",
    )
    wanted = set(frames.tolist())
    comparison.verify_history_prefix(parent, state)
    previous = {key: parent["state_" + key].copy() for key in STATE_FIELDS}
    offset = state["history_offsets"]
    captured, retained, record_rows = [], [], []
    for t in range(START - FIRST, HORIZON - FIRST):
        frame = FIRST + t
        rows = np.arange(offset[t], offset[t + 1])
        checks.require(
            len(rows) == 6 and np.all(state["history_frame_idx"][rows] == frame),
            "six native records required",
        )
        if frame in wanted:
            captured.append({key: value.copy() for key, value in previous.items()})
            retained.append({key: state["history_" + key][rows].copy() for key in RECORD_FIELDS})
            record_rows.append(rows.copy())
        pi = state["history_parent"][rows]
        checks.require(np.all((pi >= 0) & (pi < 6)), "invalid history parent")
        pose, source, held = (
            state["history_" + key][rows] for key in ("pose_xyz", "sources", "observed")
        )
        body = held[..., 1] | (
            state["history_mask_supported"][rows] & (source[..., :2] >= 0).any((2, 3))
        )
        missing = np.where(body, 0, previous["missing"][pi] + 1)
        node_age = np.where(held, 0, previous["node_missing"][pi] + 1)
        chest_age = np.where(held[..., 1], previous["direct_pectoral_age"][pi] + 1, 0)
        cost = previous["cost"][pi] + state["history_costs"][rows]
        for key, value in (
            ("body_supported", body),
            ("missing", missing),
            ("node_missing", node_age),
            ("direct_pectoral_age", chest_age),
            ("path_costs", cost),
        ):
            checks.require(
                comparison.same(value, state["history_" + key][rows]),
                "parent recurrence differs: " + key,
            )
        review = state["history_identity_review"][rows]
        checks.require(not np.any(previous["review"][pi] & ~review), "sticky review changed")
        displacement = pose[:, :, 1] - previous["pose"][pi, :, 1]
        velocity = 0.5 * previous["velocity"][pi] + 0.5 * displacement
        speed = np.linalg.norm(velocity, axis=-1, keepdims=True)
        velocity *= np.minimum(1.0, 2.0 / np.maximum(speed, 1e-12))
        velocity = np.where(body[..., None], velocity, 0.8 * previous["velocity"][pi])
        previous = dict(
            pose=pose,
            velocity=velocity,
            missing=missing,
            review=review,
            cost=cost,
            node_missing=node_age,
            direct_pectoral_age=chest_age,
        )
    for key in STATE_FIELDS:
        checks.require(
            comparison.same(previous[key], state["state_" + key]), "terminal state differs: " + key
        )
    checks.require(
        len(captured) == len(frames) and metadata["previous_frame"] == HORIZON - 1,
        "wrong reconstruction scope",
    )
    parents = {key: np.stack([item[key] for item in captured]) for key in STATE_FIELDS}
    records = {key: np.stack([item[key] for item in retained]) for key in RECORD_FIELDS}
    checks.require(
        np.array_equal(records["frame_idx"], np.repeat(frames[:, None], 6, axis=1)),
        "captured native order differs",
    )
    return parents, records, np.stack(record_rows)


def load_inputs(hashes):
    interim.bind(AUDIT / "audit.provenance.json", AUDIT_SHA, hashes)
    audit = checks.verify(AUDIT, "audit.provenance.json", interim.STAGE, hashes)
    for source, digest in list(hashes.items()):
        if source in audit["inputs"]:
            checks.require(audit["inputs"][source]["sha256"] == digest, "audited source changed")
    comparison.require_bound(AUDIT, ("summary.json", "committed_checkpoint_record.json"), hashes)
    summary = checks.read(AUDIT / "summary.json")
    checks.require(
        summary["status"] == "ok"
        and summary["native_interval"] == [START, HORIZON]
        and summary["horizon_stop"] == HORIZON
        and summary["all39_arrays_semantically_decoded"] is True
        and summary["all21_arrays_backtraced"] is True
        and summary["initial_parent_history_prefix_preserved"] is True
        and summary["expected_manifest_sha256"] == MANIFEST_SHA,
        "required successful interim state audit differs",
    )
    checkpoint = BRANCH / "checkpoints/through_068060"
    pointer = checks.read(AUDIT / "committed_checkpoint_record.json")
    checks.require(
        Path(pointer["path"]).resolve() == checkpoint
        and pointer["manifest_sha256"] == MANIFEST_SHA,
        "committed pointer differs",
    )
    for name in ("manifest.json", "metadata.json", "arrays.npz"):
        interim.input_member(audit, checkpoint / name, hashes)
        interim.input_member(audit, interim.PARENT / "checkpoint" / name, hashes)
    checks.require(
        hashes[str(checkpoint / "manifest.json")] == MANIFEST_SHA, "new manifest differs"
    )
    binding = checks.read(checkpoint / "manifest.json")["bindings"]
    checks.require(
        binding["schema"] == "bounded_mouth_one_minute_checkpoint_binding_v1"
        and binding["original_binding"] == summary["shared_input_binding"]
        and binding["native_interval"] == [START, 76360],
        "checkpoint binding differs",
    )
    for path, digest in binding["original_binding"]["code_sha256"].items():
        interim.bind(path, digest, hashes)
    for path, digest in binding["experimental_code_sha256"].items():
        source = interim.bind(path, digest, hashes)
        checks.require(source.is_relative_to(REPO), "source outside repository")
        snapshot = BRANCH / "code_snapshot" / source.relative_to(REPO)
        interim.input_member(audit, snapshot, hashes)
        checks.require(hashes[str(snapshot)] == digest, "experimental code snapshot differs")
    parent, parent_meta, _ = comparison.load_bound_checkpoint(
        interim.PARENT / "checkpoint", binding["original_binding"], hashes
    )
    state, metadata, _ = comparison.load_bound_checkpoint(checkpoint, binding, hashes)
    comparison.validate_lifecycle(parent_meta, START, 3)
    comparison.validate_lifecycle(metadata, HORIZON, 4)
    comparison.verify_history_prefix(parent, state)
    migration = interim.input_member(audit, BRANCH / "state_import.json", hashes)
    checks.require(
        hashes[str(migration)] == binding["state_import_sha256"], "migration SHA differs"
    )
    interim.verify_bounded_import(checks.read(migration), parent, parent_meta, metadata)
    return parent, state, metadata, binding


def run(output):
    output = Path(output).resolve()
    checks.require(output.is_relative_to(WORK) and not output.exists(), "new work output required")
    code = comparison.source_registry.local_code()
    hashes, outputs = dict(code), {}
    for path, digest in PINS.items():
        interim.bind(path, digest, hashes)
    output.mkdir(parents=True)
    with provenance(STAGE, output, name="preparation") as record:

        def emit(name, value=None):
            path = output / name
            if value is not None:
                path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
            outputs[str(path)] = checks.sha(path)
            record.output(name, path, sha256=outputs[str(path)])

        for path, digest in code.items():
            snapshot = output / "code_snapshot" / Path(path).relative_to(REPO)
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(Path(path).read_bytes())
            checks.require(checks.sha(snapshot) == digest, "source copy changed")
            emit(str(snapshot.relative_to(output)))
        emit(
            "plan.json",
            dict(
                schema=STAGE,
                frames=FRAMES.tolist(),
                previous_frames=(FRAMES - 1).tolist(),
                horizon_stop=HORIZON,
                original_parent_stop=START,
                audit_sha256=AUDIT_SHA,
                committed_manifest_sha256=MANIFEST_SHA,
                parent_fields=list(STATE_FIELDS),
                retained_fields=list(RECORD_FIELDS),
                raw_decode=False,
                SL_read=False,
                flow_compute=False,
                proposal_generation=False,
                fit=False,
                GPU=False,
            ),
        )
        parent, state, metadata, binding = load_inputs(hashes)
        before = dict(parent=array_signature(parent), checkpoint=array_signature(state))
        parents, records, rows = reconstruct_selected(parent, state, metadata, FRAMES)
        checks.require(
            before == dict(parent=array_signature(parent), checkpoint=array_signature(state)),
            "input state arrays mutated",
        )
        np.savez_compressed(
            output / "previous_all6_exact_states.npz",
            frame_idx=FRAMES,
            previous_frame_idx=FRAMES - 1,
            **parents,
        )
        emit("previous_all6_exact_states.npz")
        np.savez_compressed(
            output / "retained_all6_records.npz",
            selected_native_frames=FRAMES,
            history_row_indices=rows,
            **records,
        )
        emit("retained_all6_records.npz")
        emit("checkpoint_signatures.json", before)
        emit("checkpoint_metadata.json", metadata)
        emit("checkpoint_binding.json", binding)
        result = dict(
            schema=STAGE,
            status="ok",
            selected_native_frames=301,
            frames=FRAMES.tolist(),
            previous_frames=(FRAMES - 1).tolist(),
            retained_rows=301 * 6,
            active_parent_states=301 * 6,
            parent_fields=list(STATE_FIELDS),
            retained_fields=list(RECORD_FIELDS),
            reconstructed_suffix=[START, HORIZON],
            reconstructed_native_frames=HORIZON - START,
            reconstructed_retained_rows=6 * (HORIZON - START),
            same_original66400_parent_history_exact=True,
            parent_and_terminal39_codec_bound=True,
            terminal_state7_including_velocity_age_byte_exact=True,
            input_arrays_unchanged=True,
            reused_all39_semantic_audit_sha256=AUDIT_SHA,
            checkpoint_manifest_sha256=MANIFEST_SHA,
            outputs=dict(
                previous_states=dict(
                    path="previous_all6_exact_states.npz",
                    sha256=outputs[str(output / "previous_all6_exact_states.npz")],
                    arrays=array_signature(parents),
                ),
                retained_records=dict(
                    path="retained_all6_records.npz",
                    sha256=outputs[str(output / "retained_all6_records.npz")],
                    arrays=array_signature(records),
                ),
            ),
            recurrence_source_sha256=PINS[Path(old.__file__).resolve()],
            limits=[
                "Each saved parent belongs to native frame minus one; "
                "no zero velocity substitution.",
                "All six retained alternatives are exported, not just the final winner.",
                "Raw/SL/flow/proposals/fit and discarded candidates are not reconstructed here.",
                "39 payloads include saved previous observation buffers; "
                "no original video is opened.",
                "Retained source rows are post-fit strict supports, not pre-fit assignment plans.",
                "Review is inherited; no IDs, thresholds or state trajectory are changed.",
                "The completed68060 horizon is not a completed fullminute or thirty-minute result.",
            ],
        )
        emit("summary.json", result)
        (output / "RESULT.md").write_text(
            "# bounded尾診断用・直前全6親状態\n\n"
            "固定 {66524} と [66620,66920) の301 nativeについて、直前nativeの全6親7状態項目と、"
            "当該nativeの保持6件19項目を保存した。最終winnerだけには絞っていない。\n\n"
            "元66400親の全履歴を保持し、[66400,68060) の1660時刻を一度だけ親再帰で復元。"
            "0.5速度混合・2mm clip・body欠測時0.8減衰、node年齢・胸部連続支持年齢・path costを"
            "凍結advanceと同じ式順序で検証し、終端全6体7項目は保存値とbyte一致した。\n\n"
            "原画・SL配列・flow・候補生成・fit・GPU・advance・restoreは実行していない。"
            "保持sourceはfit後の支持であり、実prefit planや棄却候補ではない。"
            "生物ID判定・新方式採用・全1分/30分完走の主張は含まない。\n",
            encoding="utf-8",
        )
        emit("RESULT.md")
        checks.require(code == comparison.source_registry.local_code(), "loaded source set changed")
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
        for path, digest in {**hashes, **outputs}.items():
            checks.require(checks.sha(path) == digest, "bound input/source/output changed")
        record.param("all_source_input_snapshot_output_SHA_unchanged", True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output), indent=2))
