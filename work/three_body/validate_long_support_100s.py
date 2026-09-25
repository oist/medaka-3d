"""One CPU all-support replay of the sealed 100-second prefix, not a final audit."""

import argparse
import dataclasses
import json
import shutil
import sys
import time
from pathlib import Path

import audit_long_final as common
import numpy as np

from medaka3d.calib import load_rig
from medaka3d.provenance import provenance

REPO, LAB = common.REPO, common.LAB
RUN = REPO / "work/three_body/long_continuation_through66400_v1"
AUDIT = REPO / "work/three_body/long_remaining_supervisor_v1/bootstrap"
FIRST, STOP = 49800, 66400
RUN_SHA = "9915c44204c7acaee0abcada3a3fa2dfa82b7ac017c0b7d4fc9b50b628d48a6e"
CANDIDATE_SHA = "9f3935a221d85ab2f46c016fd46b19e29721e1ae3993acb71e24077e5a8a7c83"
AUDIT_SHA = "d9d832f5842af3b5a0304cfb1e3e22bd7a619e03e621858aa3a3701819a3ff6d"
HELPER_SHA = "9140be0d5401228f38fefa04b1d4d4e68ea672ab8a0793130fbd62952c3e61eb"
require, equal = common.require, common.equal


def code_paths():
    roots = (LAB, REPO / "src", REPO / "work/three_body")
    result = {Path(__file__).resolve(), Path(common.__file__).resolve()}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == ".py" and any(path.is_relative_to(root) for root in roots):
                result.add(path)
    return sorted(result)


def completed_inputs(evidence):
    # Gate both small seals before any payload deserialization or expensive hashing.
    for root, filename in ((RUN, "continuation.provenance.json"), (AUDIT, "audit.provenance.json")):
        require(
            common.COMMON.read_json(root / filename).get("status") == "ok",
            "both producer and independent bootstrap must be complete",
        )
    evidence.add(RUN / "continuation.provenance.json", RUN_SHA)
    evidence.add(RUN / "candidate.npz", CANDIDATE_SHA)
    evidence.add(AUDIT / "audit.provenance.json", AUDIT_SHA)
    producer = common.read_record(
        RUN, "continuation.provenance.json", "three_body_long_continuation", evidence
    )
    audited = common.read_record(
        AUDIT, "audit.provenance.json", "long_supervised_bootstrap_audit", evidence
    )
    report = common.json_output(audited, "summary", AUDIT / "summary.json")
    common.check_audit(report, RUN, CANDIDATE_SHA, stop=STOP, restores=3)
    common.input_binding(audited, RUN / "continuation.provenance.json", RUN_SHA)
    common.input_binding(audited, RUN / "candidate.npz", CANDIDATE_SHA)
    summary = common.json_output(producer, "summary", RUN / "summary.json")
    common.phase.validate_summary(summary, stop=STOP, phase_start=59760, restores=3, chunks=10)
    common.output_binding(producer, RUN / "candidate.npz", CANDIDATE_SHA)
    binding = producer["params"]["bindings"]
    for path, digest in binding["code_sha256"].items():
        evidence.add(path, digest)
        snapshot = RUN / "code_snapshot" / Path(path).relative_to(REPO)
        common.output_binding(producer, snapshot, digest)
    registered = binding["inputs"]
    seed = registered["seed"]
    paths = {
        Path(path).resolve(): digest for path, digest in seed["SL_and_calibration_sha256"].items()
    }
    for path, digest in paths.items():
        evidence.add(path, digest)
        common.input_binding(producer, path, digest)
    manifest_paths = [path for path in paths if path.name == "calibration_manifest.json"]
    require(len(manifest_paths) == 1, "one fixed calibration required")
    calibration = manifest_paths[0]
    manifest = common.COMMON.read_json(calibration)
    for model in manifest["model_files"]:
        path = (calibration.parent / model["path"]).resolve()
        require(path in paths, "calibration opens unregistered camera")
    rig = load_rig(calibration.parent)
    metadata_path = RUN / "checkpoint/metadata.json"
    common.output_binding(producer, metadata_path, evidence.paths[str(metadata_path)])
    context = common.COMMON.read_json(metadata_path)["context"]
    equal(
        common.COMMON.normalize([dataclasses.asdict(camera) for camera in rig.cameras]),
        context["cameras"],
        "loaded calibration differs from tracking context",
    )
    require(
        context["views"] == list(rig.views) and context["world_units"] == "mm",
        "native camera order/world units differ",
    )
    protocol_path = Path(registered["protocol_path"]).resolve()
    evidence.add(protocol_path, registered["protocol_sha256"])
    protocol = common.COMMON.read_json(protocol_path)
    frames = np.arange(FIRST, STOP, dtype=np.int64)
    nodes = common.phase.old.prior.CachedNodes(seed["input_root"], protocol["experiment"], frames)
    for path in nodes.paths.values():
        require(Path(path).resolve() in paths, "unregistered SLEAP cache opened")
    candidate = common.phase.old.read_npz(RUN / "candidate.npz")
    require(
        len(candidate) == 21 and np.isfinite(candidate["pose_xyz"]).all(),
        "completed persistent 21-array geometry required",
    )
    common.COMMON.same_array(candidate["frame_idx"], frames, "complete 100s native frames")
    exact = {key: common.COMMON.digest_array(value) for key, value in candidate.items()}
    equal(
        exact,
        report["all_21_candidate_arrays_exact_backtrace"],
        "candidate differs from independently audited backtrace",
    )
    counts = common.SUPPORT.direct_counts(candidate)
    equal(counts, report["counts"], "candidate availability differs from bootstrap audit")
    return candidate, nodes, rig, counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    require(output.is_relative_to(REPO) and not output.exists(), "new workspace output required")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    evidence = common.Digests()
    code = code_paths()
    for path in code:
        evidence.add(path)
    evidence.add(common.__file__, HELPER_SHA)
    for path, digest in common.PINS.items():
        evidence.add(path, digest)
    with provenance("long_100s_partitioned_support_audit", output, name="audit") as record:
        record.params_from(args)
        for path in code:
            snapshot = output / "code_snapshot" / path.relative_to(REPO)
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, snapshot)
            evidence.add(snapshot, evidence.paths[str(path)])
        print("Verifying completed producer/bootstrap input and source hashes", flush=True)
        arrays, nodes, rig, counts = completed_inputs(evidence)
        blocks = []
        compute_start = time.perf_counter()

        def save_block(info, values):
            start, stop = info["frame_range"]
            path = output / f"support_f{start:06d}-{stop:06d}.npz"
            np.savez_compressed(path, **values)
            blocks.append(dict(**info, path=path.name, sha256=common.phase.BASE.sha256(path)))
            print(
                json.dumps(
                    dict(
                        block=len(blocks),
                        interval=[start, stop],
                        source_rows=info["source_rows_checked"],
                    )
                ),
                flush=True,
            )

        result = common.verify_support_partitioned(
            arrays, nodes, rig, block_frames=1660, on_block=save_block
        )
        require(
            result["blocks"] == 10 and result["native_frames"] == 16600,
            "100-second all-frame partition count differs",
        )
        report = dict(
            schema="long_100s_partitioned_support_audit_v1",
            status="ok",
            run=str(RUN),
            candidate_sha256=CANDIDATE_SHA,
            producer_seal_sha256=RUN_SHA,
            bootstrap_audit=str(AUDIT),
            bootstrap_seal_sha256=AUDIT_SHA,
            all_source_geometry=result,
            counts=counts,
            blocks=blocks,
            support_compute_and_save_seconds=time.perf_counter() - compute_start,
            validation_seconds_before_final_guard=time.perf_counter() - started,
            raw_images_independently_decoded=0,
            new_detector_or_tracking=False,
            independent_ID_truth=False,
            full_thirty_minute_audit_complete=False,
            scope="All declared source/observed geometry only. Prior 39/21/history audit is bound, "
            "not rerun. This is a provisional 100-second prefix, not a final 30-minute audit.",
        )
        (output / "summary.json").write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )
        (output / "RESULT.md").write_text(
            "# 100秒 prefix の全支持・分割検証\n\n"
            "成功済み [49800,66400) の全16,600 native frameを1,660枚ずつ10分割し、"
            "保存sourceの排他・2mm ray・12px原画投影・2視点以上のobservedを再計算しました。"
            "rayは元float32保存を維持し、欠測は分母に残しています。\n\n"
            f"胸部直接支持 {counts['direct_pectoral_fish_frames']}/49,800、"
            f"条件付き受理 {counts['accepted_pectoral_fish_frames']}/49,800、"
            f"review {counts['review_fish_frames']}魚frame。"
            f"全宣言source {result['source_rows_checked']}件の支持判定が一致しました。\n\n"
            "元candidateと成功bootstrap監査のSHAを結合し、全入力・コード・出力を前後照合。"
            "39配列・全履歴の再帰は既存独立bootstrap監査を参照し、今回は重複実行していません。"
            "原動画再decodeは0、GT/再追跡/新検出/GPU計算はありません。"
            "数値は幾何支持であり3個体IDの正解率ではなく、30分最終監査は未完です。\n",
            encoding="utf-8",
        )
        for path, digest in evidence.paths.items():
            if not Path(path).is_relative_to(output):
                record.input(path, path, sha256=digest)
        outputs = {}
        for path in output.rglob("*"):
            if path.is_file():
                digest = common.phase.BASE.sha256(path)
                outputs[str(path)] = digest
                record.output(path.relative_to(output).as_posix(), path, sha256=digest)
        evidence.final_guard()
        common.COMMON.unchanged(outputs)
        record.param("all_input_source_snapshot_output_sha_unchanged", True)
        record.param("GPU_new_tracking_or_GT", False)
    print(json.dumps(dict(status="ok", output=str(output), counts=counts)), flush=True)


if __name__ == "__main__":
    main()
