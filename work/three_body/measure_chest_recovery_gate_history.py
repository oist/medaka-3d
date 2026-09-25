"""Read-only saved six-beam gate arithmetic, no replay, raw images or fitting."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import compare_long_support_160s as prior
import numpy as np

from medaka3d.provenance import provenance

START, STOP = 66656, 66706
STAGE = "chest_recovery_saved_six_beam_gate"
FIELDS = (
    "frame_idx",
    "parent",
    "missing",
    "node_missing",
    "observed",
    "body_supported",
    "mask_supported",
    "identity_review",
    "proposal_kind",
    "direct_pectoral_age",
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    prior.require(output.is_relative_to(prior.REPO) and not output.exists(), "new output required")
    output.mkdir(parents=True, exist_ok=False)
    evidence = prior.common.Digests()
    with provenance(STAGE, output, name="measurement") as record:
        for source in prior.code_paths():
            evidence.add(source)
            copied = output / "code_snapshot" / source.relative_to(prior.REPO)
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, copied)
            evidence.add(copied, evidence.paths[str(source)])
        producer = prior.sealed_outputs(
            prior.RUN / "continuation.provenance.json",
            prior.RUN_SHA,
            "three_body_long_continuation",
            evidence,
        )
        audited = prior.sealed_outputs(
            prior.RUN.parent / "operation.provenance.json",
            prior.AUDIT_SHA,
            "long_supervised_phase_operation",
            evidence,
        )
        prior.common.input_binding(
            audited, prior.RUN / "continuation.provenance.json", prior.RUN_SHA
        )
        prior.common.input_binding(audited, prior.RUN / "candidate.npz", prior.CANDIDATE_SHA)
        report = prior.common.json_output(audited, "audit", prior.RUN.parent / "audit.json")
        prior.common.check_audit(report, prior.RUN, prior.CANDIDATE_SHA, stop=76360, restores=4)
        for source, digest in producer["params"]["bindings"]["code_sha256"].items():
            evidence.add(source, digest)
        checkpoint = prior.RUN / "checkpoint"
        manifest = prior.common.COMMON.read_json(checkpoint / "manifest.json")
        prior.require(
            manifest["status"] == "ok" and len(manifest["arrays"]) == 39,
            "sealed full checkpoint required",
        )
        payload = (checkpoint / manifest["payload"]["file"]).resolve()
        prior.require(payload.is_relative_to(checkpoint), "checkpoint payload escaped")
        prior.common.output_binding(producer, payload, manifest["payload"]["sha256"])
        keys = ["history_offsets", *("history_" + name for name in FIELDS)]
        with np.load(payload, allow_pickle=False) as data:
            history = {}
            for key in keys:
                schema = manifest["arrays"][key]
                value = data[schema["storage_key"]]
                prior.require(
                    list(value.shape) == schema["shape"]
                    and value.dtype.str == schema["dtype"]["str"]
                    and hashlib.sha256(value.tobytes()).hexdigest() == schema["bytes_sha256"],
                    "saved history field differs: " + key,
                )
                history[key] = value
        offsets = history["history_offsets"]
        prior.require(
            len(offsets) == 26561 and offsets[-1] == 159353, "wrong sealed history length"
        )
        chosen = np.zeros(26560, np.int64)
        index = 0
        for t in range(26559, -1, -1):
            prior.require(0 <= index < offsets[t + 1] - offsets[t], "invalid final parent index")
            chosen[t] = offsets[t] + index
            index = int(history["history_parent"][chosen[t]])
        t = np.arange(START, STOP) - 49800
        prior.require(
            np.all(offsets[t] - offsets[t - 1] == 6), "previous frame must retain six beams"
        )
        rows = offsets[t - 1, None] + np.arange(6)
        prior.require(
            np.array_equal(
                history["history_frame_idx"][rows],
                np.broadcast_to((np.arange(START, STOP) - 1)[:, None], rows.shape),
            ),
            "previous beam native frame mismatch",
        )
        parent = history["history_parent"][chosen[t]]
        prior.require(
            np.array_equal(rows[np.arange(50), parent], chosen[t - 1]),
            "final winner parent does not follow saved history",
        )
        values = dict(
            frame_idx=np.arange(START, STOP),
            previous_global_history_row=rows,
            final_winner_current_global_row=chosen[t],
            final_winner_parent_index=parent,
        )
        for name in FIELDS:
            values["previous_" + name] = history["history_" + name][rows].copy()
            values["final_winner_current_" + name] = history["history_" + name][chosen[t]].copy()
        missing = values["previous_missing"]
        chest_age = values["previous_node_missing"][..., 1]
        closed = ~np.any(missing[:, 0] > 0, axis=1)
        suppressed = (chest_age[:, 0] > 0) & (missing[:, 0] == 0)
        values["state0_global_observation_recovery_gate_open"] = ~closed
        values["previous_per_beam_per_fish_recovery_eligible"] = missing > 0
        summary = dict(
            schema=STAGE + "_v1",
            status="ok",
            frame_range=[START, STOP],
            original_checkpoint_sha256=evidence.paths[str(checkpoint / "manifest.json")],
            producer_seal_sha256=prior.RUN_SHA,
            independent_audit_sha256=prior.AUDIT_SHA,
            state0_global_gate_open_frames=int((~closed).sum()),
            state0_gate_closed_despite_chest_age_frames=int(
                (closed & (chest_age[:, 0] > 0).any(1)).sum()
            ),
            state0_chest_age_positive_but_own_body_missing_zero_fish_frames=int(suppressed.sum()),
            state0_chest_age_positive_by_slot=(chest_age[:, 0] > 0).sum(0).tolist(),
            state0_body_missing_positive_by_slot=(missing[:, 0] > 0).sum(0).tolist(),
            per_beam_global_gate_open_frames=(missing > 0).any(2).sum(0).tolist(),
            per_beam_chest_age_positive_frames=(chest_age > 0).any(2).sum(0).tolist(),
            all_six_gate_closed_despite_some_chest_age_frames=int(
                ((~(missing > 0).any((1, 2))) & (chest_age > 0).any((1, 2))).sum()
            ),
            final_winner_parent_nonzero_frames=int((parent != 0).sum()),
            final_winner_parent_index_histogram={
                str(k): int(v) for k, v in zip(*np.unique(parent, return_counts=True), strict=True)
            },
            gate_open_native_frames=values["frame_idx"][~closed].tolist(),
            no_raw_decode_or_projection_or_fit=True,
            scope="Exact saved preceding six-beam body/node age arithmetic. This establishes gate "
            "eligibility, not whether a useful ungenerated candidate exists or survives fitting.",
        )
        np.savez_compressed(output / "previous_six_beams.npz", **values)
        with (output / "frames.jsonl").open("w", encoding="utf-8") as stream:
            for i, frame in enumerate(values["frame_idx"]):
                stream.write(
                    json.dumps(
                        dict(
                            frame=int(frame),
                            previous_state0_missing=missing[i, 0].tolist(),
                            previous_state0_node_missing=values["previous_node_missing"][
                                i, 0
                            ].tolist(),
                            previous_all6_missing=missing[i].tolist(),
                            previous_all6_chest_age=chest_age[i].tolist(),
                            observation_recovery_global_gate_open=bool(not closed[i]),
                            final_winner_parent_index=int(parent[i]),
                        )
                    )
                    + "\n"
                )
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (output / "RESULT.md").write_text(
            "# 胸部欠測区間に入る直前の6beam gate\n\n"
            f"[66656,66706) 50frameの直前state0について、observation_recovery全体gateが開くのは"
            f"{summary['state0_global_gate_open_frames']} frame。"
            "胸部node_missingが正なのにbody missingで全体gateが閉じるのは "
            f"{summary['state0_gate_closed_despite_chest_age_frames']} frameです。\n\n"
            f"最終76360 backtraceの直前parentがstate0ではないのは"
            f"{summary['final_winner_parent_nonzero_frames']} frame。"
            "元の最終勝者だけから生成時state0の条件を推定していません。\n\n"
            "全50frame x 6beam x 3slotの元missing/node_missing/support/reviewと親indexを保存。"
            "これは既存gateの実状態の測定で、未生成候補の存在や採用可能性の証明ではありません。"
            "原画・再投影・新fit・GT・GPUは使用していません。\n",
            encoding="utf-8",
        )
        for path, digest in evidence.paths.items():
            if not Path(path).is_relative_to(output):
                record.input(path, path, sha256=digest)
        outputs = {}
        for path in output.rglob("*"):
            if path.is_file():
                digest = prior.baseline.sha256(path)
                outputs[str(path)] = digest
                record.output(path.relative_to(output).as_posix(), path, sha256=digest)
        evidence.final_guard()
        prior.common.COMMON.unchanged(outputs)
        record.param("all_input_source_snapshot_output_sha_unchanged", True)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
