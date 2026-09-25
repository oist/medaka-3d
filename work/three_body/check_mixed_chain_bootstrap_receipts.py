"""Read only completed handoff receipt metadata, never the unfinished full chain."""

import json
import shutil
import sys
from pathlib import Path

import long_final_mixed_chain as mixed

from medaka3d.provenance import provenance

REPO = mixed.REPO
ROOT = REPO / "work/three_body/long_remaining_receipted_supervisor_v1/bootstrap"
PLAN = (
    REPO
    / "work/three_body/long_remaining_supervisor_v1/phases/f066400-076360/registration/plan.json"
)
CACHE = REPO / "work/three_body/long_mask_validation_receipts_v1"
OUTPUT = REPO / "work/three_body/long_final_mixed_chain_implementation_v2"
SEAL_SHA = "9292d244f770e3f57e65bb0acd62c8282a7967ce2691f5c36e0aaffccb14a6ac"
SUMMARY_SHA = "534f06ed62e82f40dda70eadcf3ba4e5e9628b90edbf42081b9709ec10450c4d"


def main():
    OUTPUT.mkdir(exist_ok=False)
    evidence = mixed.phase.Digests()
    code_paths = {Path(__file__).resolve(), Path(mixed.__file__).resolve()}
    code_paths.update(
        Path(f).resolve()
        for module in tuple(sys.modules.values())
        if (f := getattr(module, "__file__", None))
        and Path(f).suffix == ".py"
        and Path(f).resolve().is_relative_to(REPO)
    )
    with provenance(
        "long_final_mixed_chain_bounded_implementation", OUTPUT, name="implementation"
    ) as record:
        for path in sorted(code_paths):
            evidence.add(path)
            snapshot = OUTPUT / "code_snapshot" / path.relative_to(REPO)
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, snapshot)
            record.output(str(snapshot), snapshot, sha256=evidence.paths[str(path)])
        for path, digest in mixed.PINS.items():
            evidence.add(path, digest)
        evidence.add(ROOT / "audit.provenance.json", SEAL_SHA)
        seal = mixed.COMMON.read_json(ROOT / "audit.provenance.json")
        mixed.require(
            seal["stage"] == mixed.handoff.STAGE
            and seal["status"] == "ok"
            and seal["params"]["all_input_source_snapshot_output_sha_unchanged"] is True,
            "completed handoff bootstrap required",
        )
        for item in seal["outputs"].values():
            path = Path(item["path"]).resolve(strict=True)
            mixed.require(path.is_relative_to(ROOT), "foreign bootstrap output")
            evidence.add(path, item["sha256"])
        evidence.add(ROOT / "summary.json", SUMMARY_SHA)
        summary = mixed.legacy.json_output(seal, "summary", ROOT / "summary.json")
        mixed.require(
            summary["frame_range"] == [49800, 76360]
            and summary["checkpoint_restore_count"] == 4
            and summary["all_39_checkpoint_arrays_verified"] is True,
            "unexpected transferred phase",
        )
        evidence.add(PLAN)
        mixed.legacy.input_binding(seal, PLAN, evidence.paths[str(PLAN)])
        plan = mixed.COMMON.read_json(PLAN)
        mixed.require(
            len(plan["full_selected_chunk_records"]) == 16, "registered16 chunks required"
        )
        result = mixed.verify_receipt_usage(
            seal,
            ROOT,
            label="handoff_verify",
            plan=plan,
            evidence=evidence,
            cache_root=CACHE.resolve(strict=True),
        )
        test_path = REPO / "work/three_body/test_long_final_mixed_chain.py"
        evidence.add(test_path)
        sample = mixed.native.fixed_sample_frames()
        report = dict(
            schema="long_final_mixed_chain_implementation_v2",
            status="ok",
            prior_failed_artifact="long_final_mixed_chain_implementation_v1",
            prior_failure=(
                "Audit wrapper called nonexistent COMMON.sha256 after receipt verification; "
                "original source/output unchanged."
            ),
            new_cpu_tests=59,
            legacy_final_cpu_tests=36,
            native_samples_cpu_tests=29,
            bootstrap_root=str(ROOT),
            bootstrap_seal_sha256=SEAL_SHA,
            bootstrap_summary_sha256=SUMMARY_SHA,
            old_registration_plan_sha256=evidence.paths[str(PLAN)],
            source_sha256=evidence.paths[str(Path(mixed.__file__).resolve())],
            test_sha256=evidence.paths[str(test_path)],
            receipt_compatibility=result,
            actual_receipt_chunks=16,
            actual_receipt_views=3,
            receipt_proof_scope=(
                "Completed bootstrap outputs and exact receipt request/body/commit; "
                "mask NPZ, manifest/seal SHA, frozen validator/runtime; no semantic mask decode."
            ),
            bootstrap_historical_state_declarations_recomputed=False,
            source_and_used_receipt_inputs_rehashed=True,
            producer_other_inputs_not_rehashed=True,
            full_final_chain_called=False,
            candidate_checkpoint_arrays_opened=False,
            raw_decode=False,
            receipt_store_called=False,
            GPU_used=False,
            real_final_tracking_complete_not_asserted=True,
            native_sample_count=len(sample),
            native_image_count=len(sample) * 3,
            supported_chain=dict(
                seed_ancestors=4,
                old_operation=1,
                new_operations=28,
                logical_initializations=1,
                checkpoint_restores=32,
                external_resets=0,
            ),
            final_audit_complete=False,
            limitations=[
                "Full terminal chain remains unavailable/unvalidated.",
                "Recorded receipt uses require exactly one successful use per selected view/chunk; "
                "partly validated rejected attempts are outside this contract.",
                "39/21 state/backtrace replay is deferred to load_completed_final.",
                "All-frame support,179 flow boundaries,33 previous states, "
                "global assembly and final seal remain pending.",
            ],
        )
        output = OUTPUT / "summary.json"
        output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        record.output("summary", output, sha256=mixed.receipts.sha(output))
        text = (
            "# 混合版の最終chain接続(実終端監査ではありません)\n\n"
            "新moduleの59 CPU、旧final 36 CPU、固定native sample29 CPUが通過。"
            "固定543 native/1629画像は変更していません。\n\n"
            f"成功済bootstrapの16chunkx3viewの全{result['uses']} receiptを照合しました。"
            f"記録上hit={result['hit_count']}。NPZ bytes/SHAとreceipt metadataを再確認し、"
            "画像・マスク復元・追跡・GPU・未完了candidate/最終chainは開いていません。\n\n"
            "呼出側は新moduleの load_completed_final(root,evidence) と "
            "load_native_sample_inputs(root,evidence) を直接使用してください。"
            "戻り値を旧 verify_support_partitioned / read_native_samples へ渡せます。"
            "旧moduleのglobals置換は不要です。呼出前のcode snapshotと最後の"
            "evidence.final_guard()は呼出側責務です。\n\n"
            "新loaderは旧4祖先、旧第1operation、明示handoff witness/bootstrap、"
            "新28phaseを検証します。"
            "合計33process、論理初期化1・restore32・reset0。"
            "最終39/21実検証、全frame支持、179境界flow、"
            "33個のprevious observation、D2全体assemblyは残ります。\n"
        )
        md = OUTPUT / "RESULT.md"
        md.write_text(text, encoding="utf-8")
        record.output("result", md, sha256=mixed.receipts.sha(md))
        for path, digest in evidence.paths.items():
            record.input(path, path, sha256=digest)
        evidence.final_guard()
        for item in record.outputs.values():
            mixed.require(mixed.receipts.sha(item["path"]) == item["sha256"], "output changed")
        record.param("all_input_source_snapshot_output_sha_unchanged", True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
