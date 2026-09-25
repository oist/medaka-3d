"""CPU-only exact audit of the registered concurrent 200-frame tracking probe."""

import argparse
import dataclasses
import json
import shutil
from datetime import datetime
from pathlib import Path

import audit_continuation_comparison as old
import numpy as np

REPO, BASE = old.REPO, old.BASE
PLAN = REPO / "work/three_body/concurrent_gpu_probe_plan_v1"
PLAN_SHA = "f43d38f7b901e9231e148c4ea501da910ece09dd87db9728cf900f77d8650b2e"
PROTOCOL = REPO / "work/three_body/CONCURRENT_GPU_PROBE_PROTOCOL.md"
SINGLE = REPO / "work/three_body/continuation_transient_200_single_v1"
CURRENT = REPO / "work/three_body/continuation_transient_200_concurrent_v1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(REPO / "work"):
        raise ValueError("require a new workspace output")
    # Freeze auditor dependencies before any evidence verifier is invoked.
    sources = sorted(set(old.code_paths()) | {Path(__file__).resolve()})
    hashes = {str(path): BASE.sha256(path) for path in sources}
    plan = old.read_json(PLAN / "plan.json")
    old.bind(PLAN / "plan.json", PLAN_SHA, hashes)
    seal = old.read_json(PLAN / "registration.provenance.json")
    if seal.get("stage") != "register_concurrent_gpu_probe" or seal.get("status") != "ok":
        raise ValueError("registration is not complete")
    for category in ("inputs", "outputs"):
        for item in seal[category].values():
            old.bind(item["path"], item["sha256"], hashes)
    old.bind(
        PLAN / "registration.provenance.json",
        BASE.sha256(PLAN / "registration.provenance.json"),
        hashes,
    )
    old.bind(PROTOCOL, plan["protocol_sha256"], hashes)
    if (
        Path(plan["solo_reference"]).resolve() != SINGLE
        or (REPO / plan["output"]).resolve() != CURRENT
        or plan["concurrent_D2_root_session"] != 22901
    ):
        raise ValueError("fixed plan run binding differs")
    current_seal = old.read_json(CURRENT / "continuation.provenance.json")
    if datetime.fromisoformat(seal["finished_at"]) >= datetime.fromisoformat(
        current_seal["started_at"]
    ):
        raise ValueError("plan must have been sealed before inference")
    binding = plan["input_and_code_binding"]
    reference = old.verified._verify_run(Path(binding["reference_run"]), appearance=False)
    hashes.update(reference.hashes)
    phases = {
        name: old.verify_phase(root, 200, 0, 0, reference, hashes, binding)
        for name, root in (("historical_solo", SINGLE), ("concurrent", CURRENT))
    }
    output.mkdir(parents=True, exist_ok=False)
    with old.provenance("concurrent_tracking_independent_audit", output, name="audit") as record:
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
        snapshots = {}
        for source in sources:
            target = output / "code_snapshot" / source.relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            snapshots[str(target)] = hashes[str(source)]
        single, current = phases.values()
        if old.canonical(single["metadata"]) != old.canonical(current["metadata"]):
            raise ValueError("complete temporal metadata differs")
        rig = old.load_rig(reference.root / "calibration_snapshot")
        context = current["metadata"]["context"]
        if (
            context["tracker_config"] != reference.params["tracker_config"]
            or context["recovery_config"] != reference.params["node_recovery_config"]
            or context["persistence_config"] != reference.params["persistence_config"]
            or context["initial_pose"] != old.digest_array(reference.initial)
            or context["cameras"] != old.normalize([dataclasses.asdict(cam) for cam in rig.cameras])
        ):
            raise ValueError("actual seed/shape/config/calibration context differs")
        if set(single["arrays"]) != set(current["arrays"]) or len(current["arrays"]) != 39:
            raise ValueError("full checkpoint schema differs")
        checkpoint = {
            key: old.same_array(current["arrays"][key], single["arrays"][key], key)
            for key in single["arrays"]
        }
        reader = old.SequenceReader(reference.sequence, reference.input_root)
        checks = {
            name: {
                "history": old.audit_history(phase, reference.initial),
                "previous": old.audit_previous(phase, reader),
            }
            for name, phase in phases.items()
        }
        with np.load(reference.root / "candidate.npz", allow_pickle=False) as archive:
            original = {key: archive[key] for key in archive.files}
        if len(original) != 21:
            raise ValueError("original candidate schema differs")
        candidates = {}
        for name, phase in phases.items():
            with np.load(phase["root"] / "candidate.npz", allow_pickle=False) as archive:
                candidate = {key: archive[key] for key in archive.files}
            trace = old.traced_candidate(phase["arrays"])
            if set(candidate) != set(original) or set(trace) != set(original):
                raise ValueError("candidate/backtrace schema differs")
            candidates[name] = {}
            for key in original:
                candidates[name][key] = old.same_array(
                    candidate[key], original[key], name + "/" + key
                )
                old.same_array(trace[key], candidate[key], name + "/backtrace/" + key)
        timing = {
            name: {
                **{
                    key: phase["summary"][key]
                    for key in (
                        "preparation_seconds",
                        "tracking_seconds",
                        "checkpoint_seconds",
                        "checkpoint_payload_bytes",
                        "retained_history_rows",
                        "active_beam_count",
                    )
                },
                "final_progress": phase["progress"][-1],
            }
            for name, phase in phases.items()
        }
        ratio = (
            timing["concurrent"]["tracking_seconds"] / timing["historical_solo"]["tracking_seconds"]
        )
        summary = {
            "schema": "concurrent_tracking_independent_audit_v1",
            "status": "ok",
            "frame_range": [333519, 333719],
            "native_frames": 200,
            "original_solo_concurrent_21_arrays_byte_exact": True,
            "full_39_checkpoint_arrays_byte_exact": checkpoint,
            "entire_temporal_metadata_exact": True,
            "complete_input_code_runtime_bindings_exact": True,
            "candidate_arrays": candidates,
            "independent_checks": checks,
            "logical_initialization_count": 1,
            "restore_count": 0,
            "external_pose_reset_count": 0,
            "recovery_diagnostics": current["metadata"]["node_recovery_diagnostics"],
            "timing": timing,
            "descriptive_tracking_summary_ratio": ratio,
            "GPU_used_by_this_audit": False,
            "biological_ID_truth": False,
            "D2_prediction_parity_tested": False,
            "individual_D2_image_overlap_timestamps_available": False,
            "limits": [
                "Historical solo timing is not a fresh paired or controlled benchmark.",
                "Progress and summary tracking timers have different endpoints; both are retained.",
                "200-frame parity does not establish long-run parity or biological IDs.",
                "No actual age-eligible mouth recovery occurred in this interval.",
                "D2 was concurrent at process level; exact per-image overlap was not recorded.",
                "D2 output parity requires the separately registered 1810-image solo replay.",
            ],
        }
        report = (
            "# GPU 同時実行 200 フレームの独立監査\n\n"
            "元の完了追跡・過去の単独実行・D2 と同時実行した追跡は、候補21配列の "
            "dtype・shape・C-order bytes が全一致しました。保存状態39配列、全時間状態 metadata、"
            "入力・コード・実行環境 binding も一致しました。全 beam/history の親・速度・age・"
            "累積 cost と最終 backtrace を再検算し、"
            "直前3視点の画像・mask・SL・背景も照合しました。\n\n"
            f"summary tracking 時間は単独 {timing['historical_solo']['tracking_seconds']:.6f} 秒、"
            f"同時 {timing['concurrent']['tracking_seconds']:.6f} 秒 (比 {ratio:.4f}) です。"
            "別時刻の測定であり、統制した性能比較ではありません。progress 終端時間は"
            "異なる計測境界なので summary に別保存しています。\n\n"
            "この監査は CPU のみで、推論変更・新しい追跡はありません。初期化1回・復元0回・"
            "外部再初期化0回、全1194履歴行を保存しています。この200フレームに口の再捕捉候補"
            "の実発火はありません。生物学的ID真値・30分の品質は検証していません。\n\n"
            "D2 の予測値はまだ未比較です。個々のD2画像の同時処理時刻は未記録のため、"
            "指定する side1 1660枚 + side2 150枚はプロセス重複区間を含む集合であり、"
            "各画像が同時に処理されたとは断定しません。\n"
        )
        outputs = {}
        for name, content in (
            ("summary.json", json.dumps(summary, indent=2, allow_nan=False)),
            ("RESULT.md", report),
        ):
            path = output / name
            path.write_text(content, encoding="utf-8")
            outputs[str(path)] = BASE.sha256(path)
        for path, digest in {**snapshots, **outputs}.items():
            record.output(path, path, sha256=digest)
        old.unchanged({**hashes, **snapshots, **outputs})
        record.param("all_input_source_snapshot_output_sha_unchanged", True)
    print(json.dumps({"status": "ok", "output": str(output), "tracking_ratio_descriptive": ratio}))


if __name__ == "__main__":
    main()
