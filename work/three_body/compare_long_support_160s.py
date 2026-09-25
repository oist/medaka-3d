"""Fixed 160-second, eight-method CPU geometry/support comparison; no ID truth."""

import argparse
import dataclasses
import json
import shutil
import sys
import time
from pathlib import Path

import audit_long_final as common
import long_baseline_support as baseline
import numpy as np

from medaka3d.calib import load_rig
from medaka3d.eval.generative import observed_motion_summary
from medaka3d.eval.reacquisition import availability_episodes
from medaka3d.provenance import provenance

REPO, LAB = common.REPO, common.LAB
FIRST, STOP, BLOCK = 49800, 76360, 1660
RUN = REPO / "work/three_body/long_remaining_supervisor_v1/phases/f066400-076360/tracking"
INPUT_ROOT = REPO / "work/generative_tracking_inputs"
RUN_SHA = "95098e39bbd4d1e09d7ef98e3a7c244fe89ab2caf36a3650f360d35fc8289382"
CANDIDATE_SHA = "870836f9e086db6d1675dd283846e758038fb942fb6e8df6296f63aad98a5877"
AUDIT_SHA = "4bcd4fd1afc5c1e373c80087249ba2ec79da5afc7df3800fe5232f50c90b85f8"
STAGE = "long_160s_all7_geometry_support_comparison"
METHODS = ("new", *(f"c34_lambda_{value}" for value in baseline.LAMBDAS))
INPUT_AUDIT = REPO / "work/three_body/LONG_BASELINE_COMPARISON_INPUT_AUDIT.md"
PINS = {
    Path(
        common.__file__
    ).resolve(): "9140be0d5401228f38fefa04b1d4d4e68ea672ab8a0793130fbd62952c3e61eb",
    Path(
        baseline.__file__
    ).resolve(): "7e7ed92ee8c2879c1a09a3b504c9f5190d8f2b270aa78168ac184cfffc4cb5e4",
    INPUT_AUDIT: "c7dfb5abfd7fe5e4a592e9f5811012c8ad03249e86c936e327c16172258f89d5",
}
require, equal = common.require, common.equal


def code_paths():
    roots = (LAB, REPO / "src", REPO / "work/three_body")
    result = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == ".py" and any(path.is_relative_to(root) for root in roots):
                result.add(path)
    return sorted(result)


def sealed_outputs(path, digest, stage, evidence):
    """Rehash completed outputs, reference historical inputs without rereading videos.

    The pinned historical seal records the prior full input checks. This loader
    does not claim to independently rehash its raw-video or mask inputs today.
    Inputs actually used by this comparison are verified separately below.
    """
    evidence.add(path, digest)
    record = common.COMMON.read_json(path)
    require(
        record.get("status") == "ok" and record.get("stage") == stage,
        "required completed producer/operation seal absent",
    )
    for item in record["outputs"].values():
        output = Path(item["path"]).resolve()
        require(output.is_relative_to(path.parent), "historical output escaped its artifact")
        require(output.stat().st_size == item["size"], "historical output size differs")
        evidence.add(output, item["sha256"])
    return record


def completed_inputs(evidence):
    # Read both small success flags before opening candidate/checkpoint arrays.
    seal = RUN / "continuation.provenance.json"
    audit_seal = RUN.parent / "operation.provenance.json"
    for path in (seal, audit_seal):
        require(
            common.COMMON.read_json(path).get("status") == "ok",
            "producer and independent phase audit must both be complete",
        )
    producer = sealed_outputs(seal, RUN_SHA, "three_body_long_continuation", evidence)
    audited = sealed_outputs(audit_seal, AUDIT_SHA, "long_supervised_phase_operation", evidence)
    require(
        audited["params"]["tracking_child_exit_code"] == 0,
        "historical tracking process did not succeed",
    )
    common.input_binding(audited, seal, RUN_SHA)
    common.input_binding(audited, RUN / "candidate.npz", CANDIDATE_SHA)
    common.output_binding(producer, RUN / "candidate.npz", CANDIDATE_SHA)
    report = common.json_output(audited, "audit", RUN.parent / "audit.json")
    common.check_audit(report, RUN, CANDIDATE_SHA, stop=STOP, restores=4)
    summary = common.json_output(producer, "summary", RUN / "summary.json")
    common.phase.validate_summary(summary, stop=STOP, phase_start=66400, restores=4, chunks=16)
    bindings = producer["params"]["bindings"]
    for path, digest in bindings["code_sha256"].items():
        evidence.add(path, digest)
        snapshot = RUN / "code_snapshot" / Path(path).relative_to(REPO)
        common.output_binding(producer, snapshot, digest)
    seed = bindings["inputs"]["seed"]
    shared = {
        str(Path(path).resolve()): digest
        for path, digest in seed["SL_and_calibration_sha256"].items()
    }
    require(len(shared) == 7, "three SLEAP and four calibration files required")
    for path, digest in shared.items():
        evidence.add(path, digest)
        common.input_binding(producer, Path(path), digest)
    manifests = [Path(path) for path in shared if Path(path).name == "calibration_manifest.json"]
    require(len(manifests) == 1, "one registered calibration required")
    manifest = common.COMMON.read_json(manifests[0])
    for item in manifest["model_files"]:
        require(
            str((manifests[0].parent / item["path"]).resolve()) in shared,
            "camera model is not registered",
        )
    rig = load_rig(manifests[0].parent)
    context = common.COMMON.read_json(RUN / "checkpoint/metadata.json")["context"]
    equal(
        common.COMMON.normalize([dataclasses.asdict(camera) for camera in rig.cameras]),
        context["cameras"],
        "loaded calibration differs from tracking context",
    )
    require(
        context["views"] == list(rig.views) and context["world_units"] == "mm",
        "view order or world units differ",
    )
    frames = np.arange(FIRST, STOP, dtype=np.int64)
    paths = {
        view: Path(path)
        for view in baseline.VIEWS
        for path in shared
        if Path(path).name == f"{baseline.EXPERIMENT}__{view}.npz"
    }
    require(set(paths) == set(baseline.VIEWS), "three registered raw SLEAP caches required")
    nodes = baseline.RegisteredNodes(paths, frames)
    arrays = common.phase.old.read_npz(RUN / "candidate.npz")
    require(
        len(arrays) == 21 and np.isfinite(arrays["pose_xyz"]).all(),
        "completed persistent 21-array finite geometry required",
    )
    common.COMMON.same_array(arrays["frame_idx"], frames, "complete fixed160 native interval")
    equal(
        {k: common.COMMON.digest_array(v) for k, v in arrays.items()},
        report["all_21_candidate_arrays_exact_backtrace"],
        "independent backtrace mismatch",
    )
    counts = common.SUPPORT.direct_counts(arrays)
    equal(counts, report["counts"], "independently audited availability differs")
    return arrays, nodes, rig, shared, counts


def coordinate_metrics(xyz, frames):
    """Existing metrics use only consecutive finite native pairs, never gap bridges."""
    return dict(
        motion=observed_motion_summary(xyz, frames), availability=availability_episodes(xyz, frames)
    )


def metric_views(raw, strict, *, is_new):
    frames, pose, observed = (strict[k] for k in ("frame_idx", "pose_xyz", "observed"))
    saved = raw["pose_xyz"][:, :, 1] if is_new else raw["baseline_xyz"]
    common.COMMON.same_array(
        pose[:, :, 1],
        saved.astype(pose.dtype),
        "saved pectorals changed during support classification",
    )
    labels = ~raw["identity_review"] if is_new else raw["label_available"]
    require(
        labels.shape == (len(frames), 3) and labels.dtype == np.bool_,
        "original label/review mask must remain boolean T,3",
    )
    strict_xyz = np.where(observed[:, :, 1, None], saved, np.nan)
    native = raw["accepted_xyz"] if is_new else np.where(labels[..., None], saved, np.nan)
    strict_native = np.where((labels & observed[:, :, 1])[..., None], saved, np.nan)
    return dict(
        original_label_available_fish_frames=int(labels.sum()),
        original_label_available_by_slot=labels.sum(axis=0).tolist(),
        saved_pectoral_finite_fish_frames=int(np.isfinite(saved).all(-1).sum()),
        strict=coordinate_metrics(strict_xyz, frames),
        native_acceptance=coordinate_metrics(native, frames),
        strict_and_native_acceptance=coordinate_metrics(strict_native, frames),
        observed_nodes_by_slot=observed.sum(axis=0).tolist(),
        strict_sources_by_view_node=(strict["sources"] >= 0).sum(axis=(0, 1)).tolist(),
        original_review_fish_frames=int(raw["identity_review"].sum()) if is_new else None,
        acceptance_scope=(
            "original tracker review exclusion"
            if is_new
            else "original C34 slot_track>=0, independent of geometric support"
        ),
    )


def block_payload(raw, strict, info, support, *, is_new):
    lo, hi = info["start_index"], info["stop_index"]
    take = slice(lo, hi)
    result = {
        key: np.asarray(strict[key][take]).copy()
        for key in ("frame_idx", "pose_xyz", "sources", "observed")
    }
    result["sources"] = result["sources"].astype(np.int64)
    result.update({key: value.copy() for key, value in support.items()})
    result["original_sources"] = raw["sources"][take].copy()
    result["original_label_available"] = (
        ~raw["identity_review"][take] if is_new else raw["label_available"][take].copy()
    )
    result["original_identity_review"] = (
        raw["identity_review"][take].copy() if is_new else np.zeros((hi - lo, 3), bool)
    )
    result["original_xyz"] = (
        raw["pose_xyz"][take, :, 1].copy() if is_new else raw["baseline_xyz"][take].copy()
    )
    if not is_new:
        for key in (
            "original_source_rows",
            "slot_track",
            "finite_two_ray_observed",
            "reconstructed_pose_xyz",
        ):
            result[key] = raw[key][take].copy()
        for key in ("ray_distance_mm", "native_pixel_error"):
            result["original_" + key] = strict[key][take].copy()
    return result


def process_method(name, raw, nodes, rig, output):
    require(name in METHODS, "method outside fixed eight-method comparison")
    is_new = name == "new"
    strict = (
        raw if is_new else baseline.classify_strict_support(raw, nodes, rig, block_frames=BLOCK)
    )
    directory = output / name
    directory.mkdir(exist_ok=False)
    blocks = []

    def save(info, values):
        start, stop = info["frame_range"]
        path = directory / f"support_f{start:06d}-{stop:06d}.npz"
        np.savez_compressed(path, **block_payload(raw, strict, info, values, is_new=is_new))
        blocks.append(
            dict(**info, path=path.relative_to(output).as_posix(), sha256=baseline.sha256(path))
        )

    verified = common.verify_support_partitioned(
        strict, nodes, rig, block_frames=BLOCK, on_block=save
    )
    require(
        verified["native_frames"] == STOP - FIRST and verified["blocks"] == (STOP - FIRST) // BLOCK,
        "fixed native partition count differs",
    )
    metrics = metric_views(raw, strict, is_new=is_new)
    return dict(
        name=name,
        lambda_value=None if is_new else name.removeprefix("c34_lambda_"),
        frame_range=[FIRST, STOP],
        metrics=metrics,
        all_source_geometry=verified,
        blocks=blocks,
        original_review_applies=is_new,
        geometry_scope=(
            "saved tracker three-node geometry"
            if is_new
            else "saved C34 pectorals; source-triangulated mouth/tail diagnostics"
        ),
    )


def check_method_set(methods):
    equal([item["name"] for item in methods], list(METHODS), "all seven fixed C34 lambdas required")
    for item in methods:
        equal(item["frame_range"], [FIRST, STOP], "method interval differs")
        equal(
            [b["frame_range"] for b in item["blocks"]],
            [[f, f + BLOCK] for f in range(FIRST, STOP, BLOCK)],
            "block coverage differs",
        )


def markdown(report):
    lines = [
        "# 固定160秒・全7 C34条件との幾何支持比較",
        "",
        "[49800,76360) の全26,560 native frame。全方式に2mm ray/12px投影/2視点以上を適用。",
        "欠測を分母に残し、移動量は支持が連続したnative frame間だけで計算しました。",
        "",
        "|方式|胸部厳密支持 / 79,680|元方式受理|厳密支持∩元方式受理|胸部支持 S0/S1/S2|",
        "|---|---:|---:|---:|---|",
    ]
    for item in report["methods"]:
        m = item["metrics"]
        counts = [
            m[k]["motion"]["observed_animal_frames"]
            for k in ("strict", "native_acceptance", "strict_and_native_acceptance")
        ]
        slots = [slot[1] for slot in m["observed_nodes_by_slot"]]
        lines.append(f"|{item['name']}|{counts[0]}|{counts[1]}|{counts[2]}|{slots}|")
    lines += [
        "",
        "全8方式を固定順で掲載し、lambdaの選択や魚slotの対応付けはしていません。"
        "方式間のS0/S1/S2が同じ生物個体であるとは保証しません。",
        "C34胸部は元保存座標を維持し、口・尾は元sourceの再三角測量による派生診断です。"
        "元slot_trackの可用性と厳密支持を別々に保存しました。",
        "C34は全記録の後処理を含む旧出力で、初期化・将来情報・元実行来歴は新方式と同条件ではありません。"
        "新方式の49800初期9nodeもC34 λ0.767のその時点のsourceを借用しています。",
        "新方式は成功した160秒prefixの暫定backtraceです。既存39配列・21配列・履歴監査の封印を参照し、"
        "今回の再計算は全source支持と幾何指標です。"
        "原動画SHA再読・原画像decode・mask semantic検証・"
        "GPU・再追跡・GT読込はありません。実使用SL・校正・C34・候補・コード・出力のSHAを前後検査します。",
        "これは3個体ID正解率・同一条件の追跡精度・30分最終成績ではありません。",
        "",
    ]
    return "\n".join(lines)


def run(output):
    output = Path(output).resolve()
    require(output.is_relative_to(REPO) and not output.exists(), "new workspace output required")
    output.mkdir(parents=True, exist_ok=False)
    evidence = common.Digests()
    started = time.perf_counter()
    with provenance(STAGE, output, name="comparison") as record:
        record.param("fixed_frame_range", [FIRST, STOP])
        record.param("methods", list(METHODS))
        for path in code_paths():
            evidence.add(path)
            snapshot = output / "code_snapshot" / path.relative_to(REPO)
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, snapshot)
            evidence.add(snapshot, evidence.paths[str(path)])
        for path, digest in {**common.PINS, **PINS}.items():
            evidence.add(path, digest)
        print("Verify sealed160s candidate, prior audit and actual SL/calibration", flush=True)
        arrays, nodes, rig, shared, counts = completed_inputs(evidence)
        methods = [process_method("new", arrays, nodes, rig, output)]
        del arrays, nodes
        for value in baseline.LAMBDAS:
            print(f"C34 lambda={value}: fixed prefix load and strict support", flush=True)
            loaded = baseline.load_baseline_sources(
                INPUT_ROOT, value, FIRST, STOP, block_frames=BLOCK
            )
            for path, digest in shared.items():
                require(
                    loaded.verified_hashes.get(path) == digest,
                    "C34 SLEAP/calibration bytes differ from new method",
                )
            equal(
                common.COMMON.normalize([dataclasses.asdict(c) for c in loaded.rig.cameras]),
                common.COMMON.normalize([dataclasses.asdict(c) for c in rig.cameras]),
                "C34 calibrated camera context differs",
            )
            for path, digest in loaded.verified_hashes.items():
                evidence.add(path, digest)
            result = process_method(
                f"c34_lambda_{value}", loaded.arrays, loaded.nodes, loaded.rig, output
            )
            result["baseline_input_metadata"] = loaded.metadata
            result["input_sha256"] = dict(loaded.verified_hashes)
            methods.append(result)
            del loaded
        check_method_set(methods)
        report = dict(
            schema=STAGE + "_v1",
            status="ok",
            frame_range=[FIRST, STOP],
            native_frames=STOP - FIRST,
            block_frames=BLOCK,
            methods=methods,
            candidate_sha256=CANDIDATE_SHA,
            producer_seal_sha256=RUN_SHA,
            independent_phase_audit_seal_sha256=AUDIT_SHA,
            original_new_counts=counts,
            seconds_before_final_guard=time.perf_counter() - started,
            shared_SL_and_calibration_sha256=shared,
            original_baseline_catalog_sha256=baseline.CATALOG_SHA256,
            raw_video_rehashes=0,
            raw_images_decoded=0,
            mask_semantic_validations=0,
            GPU_or_new_tracking=False,
            independent_ID_truth=False,
            biological_slot_alignment=False,
            full_thirty_minute_comparison=False,
            historical_input_scope="Pinned completed phase audit is referenced; "
            "raw-video/packed-mask historical inputs are not independently rehashed here.",
            source_geometry_scope="C34 saved pectorals unchanged; mouth/tail derived. "
            "Original label/review masks are independent from strict geometric support.",
        )
        (output / "summary.json").write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )
        (output / "RESULT.md").write_text(markdown(report), encoding="utf-8")
        for path, digest in evidence.paths.items():
            if not Path(path).is_relative_to(output):
                record.input(path, path, sha256=digest)
        outputs = {}
        for path in output.rglob("*"):
            if path.is_file():
                digest = baseline.sha256(path)
                outputs[str(path)] = digest
                record.output(path.relative_to(output).as_posix(), path, sha256=digest)
        evidence.final_guard()
        common.COMMON.unchanged(outputs)
        record.param("all_used_input_source_snapshot_output_sha_unchanged", True)
    print(json.dumps(dict(status="ok", output=str(output), methods=list(METHODS))), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.output)


if __name__ == "__main__":
    main()
