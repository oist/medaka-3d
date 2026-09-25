"""Register, then separately replay the fixed 1810-image D2 concurrency probe.

Registration is CPU-only and never loads prediction arrays. Execution requires
the existing Detectron2 interpreter and an exclusively assigned GPU slot.
"""

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(REPO / "src"))
import prepare_chunked_source_masks as staging  # noqa: E402
from chunk_masks import validate_packed_masks  # noqa: E402

from medaka3d.provenance import provenance  # noqa: E402

SOURCE_CACHE = REPO / "work/three_body/source_cache"
PROTOCOL = LAB / "long_input_protocol.json"
PRODUCER = (
    REPO
    / "work/three_body/inputs/long_source_masks_49800-348600_v1"
    / "chunks/f053120-054780/attempt_0001"
)
TARGETS = (("side1_cam", 53120, 54780), ("side2_cam", 53120, 53270))
CONCURRENCY_EVIDENCE = (
    REPO / "work/three_body/CONCURRENT_GPU_PROBE_PROTOCOL.md",
    REPO / "work/three_body/concurrent_gpu_probe_plan_v1/plan.json",
    REPO / "work/three_body/concurrent_gpu_probe_plan_v1/registration.provenance.json",
    REPO / "work/three_body/continuation_transient_200_concurrent_v1/continuation.provenance.json",
)
FIELDS = {
    "image_shape_hw",
    "frame_idx",
    "frame_offsets",
    "mask_boxes",
    "mask_byte_offsets",
    "mask_bits",
    "scores",
    "pred_boxes",
    "classes",
    "instance_id",
    "source_track_id",
    "view_available",
    "frame_mask_sha256",
}
FRAME_FIELDS = {"frame_idx", "view_available", "frame_mask_sha256"}
ROW_FIELDS = {"mask_boxes", "scores", "pred_boxes", "classes", "instance_id", "source_track_id"}
CODE_FILES = tuple(
    sorted(
        {
            Path(__file__).resolve(),
            *(
                LAB / name
                for name in (
                    "prepare_chunked_source_masks.py",
                    "prepare_source_masks.py",
                    "prepare_sequence.py",
                    "source_sequence.py",
                    "observations.py",
                    "chunk_masks.py",
                )
            ),
            REPO / "src/medaka3d/provenance.py",
            REPO / "src/medaka3d/types.py",
        }
    )
)


def sha(path):
    return staging.sha256(Path(path))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def targets():
    return [[view, frame] for view, start, stop in TARGETS for frame in range(start, stop)]


def byte_equal(first, second):
    return (
        first.dtype == second.dtype
        and first.shape == second.shape
        and first.tobytes(order="C") == second.tobytes(order="C")
    )


def array_digest(value):
    return dict(
        dtype=value.dtype.str,
        shape=list(value.shape),
        sha256=hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    )


def slice_packed(data, start, stop):
    """Preserve all values/order, changing only offsets for a contiguous subwindow."""
    if set(data) != FIELDS or not 0 <= start < stop <= len(data["frame_idx"]):
        raise ValueError("packed schema or slice bounds differ")
    lo, hi = int(data["frame_offsets"][start]), int(data["frame_offsets"][stop])
    first, last = int(data["mask_byte_offsets"][lo]), int(data["mask_byte_offsets"][hi])
    result = {key: data[key][start:stop].copy() for key in FRAME_FIELDS}
    result.update({key: data[key][lo:hi].copy() for key in ROW_FIELDS})
    result.update(
        image_shape_hw=data["image_shape_hw"].copy(),
        frame_offsets=data["frame_offsets"][start : stop + 1].copy() - lo,
        mask_byte_offsets=data["mask_byte_offsets"][lo : hi + 1].copy() - first,
        mask_bits=data["mask_bits"][first:last].copy(),
    )
    return result


def compare_packed(actual, expected):
    """Record exact byte equality, including NaN/zero bits; never reorder instances."""
    if set(actual) != FIELDS or set(expected) != FIELDS:
        raise ValueError("comparison requires all thirteen packed fields")
    validate_packed_masks(actual)
    validate_packed_masks(expected)
    if not byte_equal(actual["frame_idx"], expected["frame_idx"]):
        raise ValueError("native frame list differs")
    arrays = {}
    for key in sorted(FIELDS):
        left, right = actual[key], expected[key]
        max_delta = None
        if left.dtype.kind == "f" and left.shape == right.shape:
            max_delta = float(np.max(np.abs(left.astype(np.float64) - right), initial=0))
        arrays[key] = dict(
            byte_exact=byte_equal(left, right),
            actual=array_digest(left),
            expected=array_digest(right),
            max_absolute_float_difference=max_delta,
        )
    frame_results = []
    for offset, native in enumerate(actual["frame_idx"]):
        a, b = slice_packed(actual, offset, offset + 1), slice_packed(expected, offset, offset + 1)
        fields = {key: byte_equal(a[key], b[key]) for key in sorted(FIELDS)}
        frame_results.append(
            dict(
                native_frame=int(native),
                all_thirteen_fields_byte_exact=all(fields.values()),
                fields_byte_exact=fields,
                actual_instances=int(a["frame_offsets"][-1]),
                expected_instances=int(b["frame_offsets"][-1]),
                decoded_mask_sha_actual=bytes(a["frame_mask_sha256"][0]).decode("ascii"),
                decoded_mask_sha_expected=bytes(b["frame_mask_sha256"][0]).decode("ascii"),
            )
        )
    return dict(
        all_thirteen_fields_byte_exact=all(item["byte_exact"] for item in arrays.values()),
        arrays=arrays,
        frames=frame_results,
    )


def unchanged(hashes):
    for path, expected in hashes.items():
        if sha(path) != expected:
            raise ValueError("input/source/output changed: " + str(path))


def fresh(path):
    path = Path(path).resolve()
    if path.exists() or path.is_symlink() or not path.is_relative_to(REPO / "work"):
        raise ValueError("require a new output inside work")
    path.mkdir(parents=True, exist_ok=False)
    return path


def snapshot(record, root, hashes):
    result = {}
    for path in CODE_FILES:
        target = root / "code_snapshot" / path.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if sha(target) != hashes[str(path)]:
            raise ValueError("source snapshot changed")
        result[str(target)] = hashes[str(path)]
        record.output(str(target), target, sha256=hashes[str(path)])
    return result


def register(root):
    # Fixed complete native list is established before any numerical output is read.
    chosen = targets()
    if len(chosen) != 1810 or len({tuple(v) for v in chosen}) != 1810:
        raise ValueError("fixed target enumeration differs")
    code = {str(path): sha(path) for path in CODE_FILES}
    seal_path, manifest_path = PRODUCER / "staging.provenance.json", PRODUCER / "manifest.json"
    seal, manifest = read_json(seal_path), read_json(manifest_path)
    if (
        seal.get("status") != "ok"
        or seal.get("stage") != "prepare_independent_mask_chunk"
        or manifest["source_interval_half_open"] != [53120, 54780]
        or Path(seal["outputs"]["manifest"]["path"]).resolve() != manifest_path
        or sha(manifest_path) != seal["outputs"]["manifest"]["sha256"]
    ):
        raise ValueError("selected producer chunk is not successfully sealed")
    evidence = {
        str(path): sha(path) for path in (seal_path, manifest_path, PROTOCOL, *CONCURRENCY_EVIDENCE)
    }
    probe_registration, probe_tracking = (
        read_json(CONCURRENCY_EVIDENCE[2]),
        read_json(CONCURRENCY_EVIDENCE[3]),
    )
    if (
        probe_registration.get("status") != "ok"
        or probe_registration.get("stage") != "register_concurrent_gpu_probe"
        or probe_tracking.get("status") != "ok"
        or probe_tracking.get("stage") != "three_body_persistent_continuation"
        or sha(CONCURRENCY_EVIDENCE[1])
        != "f43d38f7b901e9231e148c4ea501da910ece09dd87db9728cf900f77d8650b2e"
    ):
        raise ValueError("original concurrency probe is not the registered completed run")
    if manifest["protocol_sha256"] != evidence[str(PROTOCOL)]:
        raise ValueError("source protocol binding differs")
    root = fresh(root)
    with provenance("register_concurrent_d2_solo_replay", root, name="registration") as record:
        for path, digest in {**evidence, **code}.items():
            record.input(path, path, sha256=digest)
        snapshots = snapshot(record, root, code)
        plan = dict(
            schema="concurrent_d2_solo_replay_plan_v1",
            targets=chosen,
            image_count=1810,
            view_ranges=[dict(view=v, start=a, stop=b) for v, a, b in TARGETS],
            producer=str(PRODUCER),
            source_cache=str(SOURCE_CACHE),
            protocol=str(PROTOCOL),
            evidence_sha256=evidence,
            code_sha256=code,
            execution="solo GPU, unchanged retained side DefaultPredictor, native batch 1",
            comparisons_opened=False,
            GPU_used=False,
            prior_NPZ_metadata_read_before_inference=[
                "frame_idx",
                "image_shape_hw",
                "view_available",
                "source_track_id",
            ],
            compare="13 packed fields dtype/shape/C-order bytes, each decoded mask SHA, BGR SHA",
            limitations=[
                "Ranges encompass observed process overlap and include serial-control images.",
                "No per-image concurrency timestamp was recorded; individual overlap is unknown.",
                "Replay uses a newly constructed side predictor, shared across the two views.",
                "Differences cannot isolate concurrency from cold or nondeterministic effects.",
                "No top prediction is rerun and no future frame/ID/SL/seed is consumed.",
                "Parity is conditional on these fixed 1810 images, not all later frames.",
                "No runtime backend flag, threshold, model, precision or batch-size changes.",
            ],
        )
        path = root / "plan.json"
        path.write_text(json.dumps(plan, indent=2, allow_nan=False), encoding="utf-8")
        digest = sha(path)
        record.output("plan", path, sha256=digest)
        unchanged({**evidence, **code, **snapshots, str(path): digest})
        record.param("all_input_source_snapshot_output_sha_unchanged", True)
    print(json.dumps(dict(status="registered", plan=str(path), sha256=digest, GPU_used=False)))


def verify_plan(root, expected_sha256):
    root = Path(root).resolve(strict=True)
    seal_path, path = root / "registration.provenance.json", root / "plan.json"
    if sha(path) != expected_sha256:
        raise ValueError("registered plan SHA differs from the explicitly frozen value")
    seal, plan = read_json(seal_path), read_json(path)
    if (
        seal.get("status") != "ok"
        or seal.get("stage") != "register_concurrent_d2_solo_replay"
        or seal["params"].get("all_input_source_snapshot_output_sha_unchanged") is not True
        or Path(seal["outputs"]["plan"]["path"]).resolve() != path
        or plan.get("targets") != targets()
        or plan.get("image_count") != 1810
        or Path(plan["producer"]).resolve() != PRODUCER
        or Path(plan["source_cache"]).resolve() != SOURCE_CACHE
        or Path(plan["protocol"]).resolve() != PROTOCOL
        or set(plan["code_sha256"]) != {str(p) for p in CODE_FILES}
    ):
        raise ValueError("fixed replay registration differs")
    hashes = {str(path): sha(path), str(seal_path): sha(seal_path)}
    for section in ("inputs", "outputs"):
        for item in seal[section].values():
            actual = Path(item["path"]).resolve(strict=True)
            if section == "outputs" and not actual.is_relative_to(root):
                raise ValueError("registered output escaped plan root")
            if sha(actual) != item["sha256"]:
                raise ValueError("registration evidence changed")
            hashes[str(actual)] = item["sha256"]
    hashes.update(plan["evidence_sha256"])
    hashes.update(plan["code_sha256"])
    unchanged(hashes)
    return plan, seal, hashes


def execute(plan_root, output, expected_plan_sha256):
    # No predictor/GPU context is constructed in the register or help paths.
    plan, registration, hashes = verify_plan(plan_root, expected_plan_sha256)
    output = fresh(output)
    with provenance("concurrent_d2_solo_replay", output, name="replay") as record:
        if datetime.fromisoformat(registration["finished_at"]) >= datetime.fromisoformat(
            record.started_at
        ):
            raise ValueError("replay must start after registration")
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
        snapshots = snapshot(record, output, plan["code_sha256"])
        context = staging.prepare_context(PROTOCOL, SOURCE_CACHE)
        manifest = staging.validate_attempt(context, PRODUCER, 53120, 54780)
        producer_seal = read_json(PRODUCER / "staging.provenance.json")
        for info in (
            *context.input_records.values(),
            *context.code_records.values(),
            *producer_seal["outputs"].values(),
        ):
            hashes[str(Path(info["path"]).resolve())] = info["sha256"]
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
        record.param("binding", context.binding)
        record.param("plan_sha256", sha(Path(plan_root) / "plan.json"))
        record.param("registered_images", 1810)
        record.param("new_tracking_or_ID_inputs", False)
        (output / "masks").mkdir()
        started = time.perf_counter()
        predictor, model = context.runtime.make_predictor("side", context.assets, output)
        if not staging.exact_contract(model, manifest["models"]["side"]):
            raise ValueError("replay side model/config metadata differs from producer")
        views = {}
        try:
            for view, start, stop in TARGETS:
                views[view] = context.process_view(
                    context.sources[view],
                    view,
                    start,
                    stop,
                    context.protocol["background_frames"],
                    predictor,
                    context.runtime.synchronize,
                    output,
                    context.capture_factory,
                )
        finally:
            del predictor
        inference_complete = time.perf_counter()
        raw_results_path = output / "replay_views.json"
        raw_results_path.write_text(json.dumps(views, indent=2, allow_nan=False), encoding="utf-8")
        outputs = {str(raw_results_path): sha(raw_results_path)}
        # Mask pixels, scores, boxes and comparison values are opened only now.
        # validate_attempt already read the four structural independence fields.
        comparisons = {}
        for view, start, stop in TARGETS:
            count = stop - start
            expected_raw = manifest["raw"][view]["frame_bgr_sha256"][:count]
            raw_flags = [
                a == b for a, b in zip(views[view]["raw_hashes"], expected_raw, strict=True)
            ]
            if len(raw_flags) != count or not all(raw_flags):
                raise ValueError("solo and producer native BGR bytes differ")
            actual_path = output / views[view]["masks"]["path"]
            expected_path = PRODUCER / manifest["masks"][view]["path"]
            with np.load(actual_path, allow_pickle=False) as archive:
                actual = {key: archive[key] for key in archive.files}
            with np.load(expected_path, allow_pickle=False) as archive:
                whole = {key: archive[key] for key in archive.files}
            validate_packed_masks(whole)
            expected = slice_packed(whole, 0, count)
            comparisons[view] = compare_packed(actual, expected)
            comparisons[view]["raw_BGR_SHA_exact"] = raw_flags
            comparisons[view]["same_registered_background_SHA"] = (
                views[view]["background"]["sha256"] == manifest["backgrounds"][view]["sha256"]
            )
            if not comparisons[view]["same_registered_background_SHA"]:
                raise ValueError("registered background pixels/PNG bytes differ")
            for category in ("masks", "background"):
                info = views[view][category]
                outputs[str(output / info["path"])] = info["sha256"]
        for name in ("side_effective_config.yaml", "side_source_config.yaml"):
            outputs[str(output / name)] = sha(output / name)
        exact = all(v["all_thirteen_fields_byte_exact"] for v in comparisons.values())
        summary = dict(
            schema="concurrent_d2_solo_replay_v1",
            technical_completion=True,
            image_count=1810,
            comparisons=comparisons,
            all_predictions_byte_exact=exact,
            raw_images_exact=True,
            producer=str(PRODUCER),
            detector_runtime=context.runtime.metadata,
            replay_predictor_and_process_view_seconds=inference_complete - started,
            per_view_inference_seconds={
                v: sum(r["masks"]["inference_s_by_frame"]) for v, r in views.items()
            },
            old_mask_pixels_scores_boxes_loaded_after_all_replay_inference=True,
            prior_NPZ_metadata_read_before_inference=plan[
                "prior_NPZ_metadata_read_before_inference"
            ],
            exact_per_image_concurrency_known=False,
            biological_ID_truth=False,
            limits=plan["limitations"],
        )
        summary_path = output / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        outputs[str(summary_path)] = sha(summary_path)
        report_path = output / "RESULT.md"
        report_path.write_text(
            "# 固定 1810 画像 D2 単独再生\n\n"
            f"全13 packed fields の完全一致: {exact}。原画 BGR SHA は全1810枚一致。\n\n"
            "side1 native 53120:54780 の1660枚と side2 53120:53270 の150枚を事前固定し、"
            "同じ既存 FP32・batch1・モデル・設定で再生しました。構造metadataは事前検証し、"
            "比較するmask画素・score・boxを開いたのは全再生終了後です。個々の画像の"
            "実同時処理時刻は未記録で、serial control も含みます。差が出ても冷起動や"
            "非決定的処理と同時実行の影響はこの1回だけで切り分けられません。"
            "全30分への数値同値や生物学的ID精度を証明する結果ではありません。\n",
            encoding="utf-8",
        )
        outputs[str(report_path)] = sha(report_path)
        staging.verify_frozen_inputs(context)
        for path, digest in outputs.items():
            record.output(path, path, sha256=digest)
        unchanged({**hashes, **snapshots, **outputs})
        record.param("all_input_source_snapshot_output_sha_unchanged", True)
        record.param("all_predictions_byte_exact", exact)
    print(json.dumps(dict(status="complete", all_predictions_byte_exact=exact, output=str(output))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--register", action="store_true")
    group.add_argument("--execute", action="store_true")
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-plan-sha256")
    args = parser.parse_args()
    if args.register:
        if args.output is not None or args.expected_plan_sha256 is not None:
            parser.error("registration uses only --plan-root")
        register(args.plan_root)
    else:
        if args.output is None or args.expected_plan_sha256 is None:
            parser.error("execution requires --output and --expected-plan-sha256")
        execute(args.plan_root, args.output, args.expected_plan_sha256)


if __name__ == "__main__":
    main()
