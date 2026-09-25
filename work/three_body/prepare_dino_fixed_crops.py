"""Freeze all71 anonymous DINO crop inputs before any model inference.

Only18 previously sealed native PNGs and their18 observations are decoded.
All45 time/view requests and700 mask pairs remain, including empty or tokenless
rows. Historical geometry/color evidence is bound for a later, separate join;
it cannot select crops. No model, Torch, video, candidate, fit, GPU or ID edit.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path

import cv2
import dino_mask_partial_correspondence as partial
import fetch_dinov2_assets as assets
import numpy as np
import smoke_dinov2_cpu as smoke

WORK = Path(__file__).resolve().parent
STAGE = "dino_fixed71_crop_preparation_v1"
RAW = WORK / "bounded_shape_raw_review_v1"
IDENTITY = WORK / "identity_components_fixed6_v1"
RAW_SHA = "d5fa256848a0298fc0d5f5cbca2e7b7a0d55867b5874f9159d5859c56ddcbd70"
IDENTITY_SHA = "7d010cfa6c1de3e70ab7f03ad432723512a83dd1158c9743539ddba0ffba3295"
AUDIT = WORK / "bounded_mouth_first_chunk_interim_audit_v1"
AUDIT_SHA = "08a7440cb5c851c4a97d98d1d67f665f6762a89c524c2ba87bfa7c0245656f04"
ASSET_ROOT = WORK / "assets/dinov2_vits14_7764ea0_v2"
ASSET_SHA = "6e2e94af96c35e5c776308a1ec978d1f99cabdd4a0227009db41b7f4d3d69cf5"
FRAMES = (66524, 66600, 66628, 66703, 66800, 66908)
VIEWS = ("top_cam", "side1_cam", "side2_cam")
FIRST = (66524, "top_cam", 0)
CONDITIONS = ("raw", "masked")
PINS = {
    WORK
    / "smoke_dinov2_cpu.py": "853992165dcdcd2a1e1ba5b61a588f24781a97465f3ad17f8bed7b46ab4a1588",
    WORK
    / "fetch_dinov2_assets.py": "622b96de492451bfe67975406761ea6e72c2c597511f6733d17de0741ea94709",
    WORK / "dino_mask_partial_correspondence.py": (
        "de487b2095c09c27be14e7fec92980d5a84f8c0ec6bf08017d7342aab145e400"
    ),
    WORK / "DINO_PARTIAL_PROTOCOL.md": (
        "2fd1870cf23c2c8395d45622a5c198dd8060c46b7a68c71d4fe07417edd87f40"
    ),
}
ARRAY_SHAPES = {
    "bbox_xyxy": (4,),
    "square_xyxy": (4,),
    "source_xyxy": (4,),
    "padding_ltrb": (4,),
    "native_to_crop": (3, 3),
    "crop_to_native": (3, 3),
    "raw_rgb": (224, 224, 3),
    "masked_rgb": (224, 224, 3),
    "raw_normalized_chw": (3, 224, 224),
    "masked_normalized_chw": (3, 224, 224),
    "resized_mask": (224, 224),
    "resized_native_region": (224, 224),
    "token_index": (256,),
    "token_center_crop_xy": (256, 2),
    "token_center_native_xy": (256, 2),
    "token_native_xyxy": (256, 4),
    "token_footprint_pixels": (256,),
    "token_core_pixels": (256,),
    "token_eligible": (256,),
}


def require(value, message):
    if not value:
        raise ValueError(message)


def signature(array):
    return dict(
        dtype=array.dtype.str,
        shape=list(array.shape),
        sha256=hashlib.sha256(array.tobytes()).hexdigest(),
    )


def bind(path, digest, evidence):
    path = Path(path).resolve(strict=True)
    key = str(path)
    require(key not in evidence or evidence[key] == digest, "input digest conflicts")
    require(assets.sha(path) == digest, "input bytes changed: " + key)
    evidence[key] = digest
    return path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_seal(root, filename, stage, expected, evidence):
    """Verify all completed outputs; upstream inputs remain referenced evidence.

    This is deliberately not a rerun of the old raw-video/geometry audit. Its
    pinned seal authenticates those earlier checks; directly reused small input
    members are also bound below. No transitive MP4 is opened here.
    """
    root = Path(root).resolve(strict=True)
    path = bind(root / filename, expected, evidence)
    seal = read_json(path)
    require(
        seal.get("stage") == stage and seal.get("status") == "ok" and seal.get("finished_at"),
        "completed expected stage required",
    )
    for name, item in seal["outputs"].items():
        target = (root / name).resolve(strict=True)
        require(
            target.is_relative_to(root) and Path(item["path"]).resolve() == target,
            "sealed output role/path mismatch",
        )
        require(
            target.stat().st_size == item.get("size", item.get("size_bytes")),
            "sealed output size changed",
        )
        bind(target, item["sha256"], evidence)
    return seal


def member(seal, path, evidence):
    path = Path(path).resolve(strict=True)
    item = seal["inputs"].get(str(path))
    require(
        isinstance(item, dict) and Path(item["path"]).resolve() == path,
        "prior input membership missing",
    )
    return bind(path, item["sha256"], evidence)


def sealed_output(root, seal, name, evidence):
    path = (root / name).resolve(strict=True)
    item = seal["outputs"].get(name)
    require(
        isinstance(item, dict) and Path(item["path"]).resolve() == path,
        "required output role absent",
    )
    return bind(path, item["sha256"], evidence)


def fixed_rows(rows):
    keys = [(row["frame_idx"], row["view"]) for row in rows]
    require(
        len(keys) == 18 and set(keys) == {(n, v) for n in FRAMES for v in VIEWS},
        "exact18 native/view records required",
    )
    return dict(zip(keys, rows, strict=True))


def validate_source_contract(raw_summary, identity_summary, raw, identity):
    """Actual JSON schema; candidate contents remain prior sealed evidence.

    Identity summary has no horizon_stop. Check the same pinned old audit and
    both candidate input digests as the raw review, without reopening candidates.
    """
    require(
        raw_summary["status"] == identity_summary["status"] == "ok"
        and raw_summary["horizon_stop"] == 68060,
        "same68060 successful raw evidence required",
    )
    require(
        raw_summary["frames"] == identity_summary["frames"] == list(FRAMES)
        and identity_summary["methods"] == ["directional", "bounded"]
        and identity_summary["views"] == list(VIEWS),
        "fixed6 methods/views changed",
    )
    references = {}
    paths = [AUDIT / "audit.provenance.json"] + [
        AUDIT / f"{method}_same68060_candidate.npz" for method in ("directional", "bounded")
    ]
    for path in paths:
        key = str(path.resolve())
        left, right = raw["inputs"].get(key), identity["inputs"].get(key)
        require(
            isinstance(left, dict)
            and isinstance(right, dict)
            and Path(left["path"]).resolve() == Path(right["path"]).resolve() == path.resolve()
            and left["sha256"] == right["sha256"],
            "same pinned audit/candidate input lineage required",
        )
        if path.name == "audit.provenance.json":
            require(left["sha256"] == AUDIT_SHA, "fixed same68060 audit reference changed")
        references[key] = left["sha256"]
    return references


def load_inputs(evidence):
    raw = verify_seal(
        RAW, "review.provenance.json", "bounded_shape_raw_review_v1", RAW_SHA, evidence
    )
    identity = verify_seal(
        IDENTITY,
        "diagnosis.provenance.json",
        "identity_components_fixed6_v1",
        IDENTITY_SHA,
        evidence,
    )
    member(identity, RAW / "review.provenance.json", evidence)
    raw_summary = read_json(sealed_output(RAW, raw, "summary.json", evidence))
    identity_summary = read_json(sealed_output(IDENTITY, identity, "summary.json", evidence))
    horizon_references = validate_source_contract(raw_summary, identity_summary, raw, identity)
    require(
        identity_summary["all_mask_descriptors"] == 71
        and identity_summary["finite_distance_elements"] == 149
        and identity_summary["requested_pair_views"] == 90,
        "prior full mask/partialD denominator changed",
    )
    shared = identity_summary["shared_SL_calibration_sha256"]
    require(
        len(shared) == 7
        and sum(Path(p).suffix == ".npz" for p in shared) == 3
        and sum(Path(p).suffix == ".json" for p in shared) == 4,
        "sameSL3/calibration4 required",
    )
    for path, digest in shared.items():
        bind(path, digest, evidence)
        member(raw, path, evidence)
        member(identity, path, evidence)
    rows_path = sealed_output(RAW, raw, "native_observation_evidence.json", evidence)
    member(identity, rows_path, evidence)
    observations = fixed_rows(read_json(rows_path))
    descriptors = fixed_rows(
        read_json(sealed_output(IDENTITY, identity, "all_mask_descriptors.json", evidence))
    )
    for native in FRAMES:
        for view in VIEWS:
            for name in (f"raw_{native}_{view}.png", f"observation_{native}_{view}.npz"):
                path = sealed_output(RAW, raw, name, evidence)
                member(identity, path, evidence)
    # File hashes above bind geometry refs and partialD; do not deserialize them
    # or let slot geometry select/align any crop in this preparation stage.
    return dict(
        raw=raw,
        identity=identity,
        observations=observations,
        descriptors=descriptors,
        shared=shared,
        horizon_references=horizon_references,
    )


def check_native(image_bgr, masks, available, detections, record, descriptors):
    require(
        isinstance(image_bgr, np.ndarray)
        and image_bgr.dtype == np.uint8
        and image_bgr.shape == (1200, 1200, 3),
        "original native BGR1200 required",
    )
    require(
        isinstance(available, np.ndarray)
        and available.shape == ()
        and available.dtype == bool
        and bool(available),
        "registered available observation required",
    )
    require(
        isinstance(masks, np.ndarray)
        and masks.dtype == bool
        and masks.shape == (record["mask_count"], 1200, 1200),
        "all native mask rows required",
    )
    require(
        signature(image_bgr)["sha256"] == record["bgr_sha256"]
        and signature(masks)["sha256"] == record["mask_sha256"],
        "native BGR/mask signature changed",
    )
    require(
        signature(detections) == record["detection_signature"], "same native SLEAP bytes required"
    )
    require(
        descriptors["all_mask_count"] == len(masks) and len(descriptors["masks"]) == len(masks),
        "all anonymous descriptor rows required",
    )
    for index, (mask, row) in enumerate(zip(masks, descriptors["masks"], strict=True)):
        require(
            row["mask_row"] == index and row["mask_sha256"] == signature(mask)["sha256"],
            "mask row/order not bound to old anonymous evidence",
        )


def prepared_crop(image_bgr, mask, descriptor_row):
    """Explicit BGR->RGB; call the frozen utility once and preserve its output."""
    before = signature(image_bgr), signature(mask)
    rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
    crop = partial.prepare_mask_crop(rgb, mask)
    require(
        before == (signature(image_bgr), signature(mask)), "crop utility mutated original inputs"
    )
    require(
        crop["core_sha256"] == descriptor_row["core_sha256"]
        and crop["core_pixels"] == descriptor_row["core_pixels"],
        "same native5x5 eroded core required",
    )
    require(crop["schema"] == partial.VERSION, "crop utility version differs")
    arrays = {k: v for k, v in crop.items() if isinstance(v, np.ndarray)}
    require(set(arrays) == set(ARRAY_SHAPES), "complete crop array schema changed")
    for key, value in arrays.items():
        require(
            value.shape == ARRAY_SHAPES[key] and not value.dtype.hasobject,
            "crop array shape/dtype changed",
        )
        dtype = (
            np.float32
            if key.endswith(("_rgb", "_chw"))
            else bool
            if key in ("resized_mask", "resized_native_region", "token_eligible")
            else np.float64
            if key
            in (
                "native_to_crop",
                "crop_to_native",
                "token_center_crop_xy",
                "token_center_native_xy",
            )
            else np.int64
        )
        require(value.dtype == dtype, "crop array exact dtype changed: " + key)
    require(
        crop["image_rgb_sha256"] == signature(rgb)["sha256"]
        and crop["mask_sha256"] == before[1]["sha256"],
        "crop did not preserve native RGB/mask origin",
    )
    for condition in CONDITIONS:
        require(
            arrays[condition + "_normalized_chw"].dtype == np.float32
            and arrays[condition + "_rgb"].dtype == np.float32,
            "fixed float32 preprocessing required",
        )
    require(
        arrays["token_eligible"].dtype == bool
        and np.array_equal(arrays["token_index"], np.arange(256))
        and crop["eligible_token_count"] == int(arrays["token_eligible"].sum()),
        "all256 row-major tokens required",
    )
    nonempty = bool(mask.any())
    require(
        crop["crop_status"] == ("ready" if nonempty else "empty_mask"), "empty-mask status mismatch"
    )
    for condition in CONDITIONS:
        values = arrays[condition + "_normalized_chw"]
        require(
            np.isfinite(values).all() if nonempty else np.isnan(values).all(),
            "model input finite/empty status mismatch",
        )
    metadata = {k: v for k, v in crop.items() if k not in arrays}
    metadata["array_signatures"] = {k: signature(v) for k, v in arrays.items()}
    return arrays, metadata


def make_plan(counts):
    requests = partial.enumerate_mask_pairs(counts)
    require(
        partial.FRAMES == FRAMES and partial.VIEWS == VIEWS,
        "fixed utility frame/view scope changed",
    )
    require(
        all(type(n) is int and n >= 0 for n in counts.values()), "fixed observed masks required"
    )
    require(sum(counts.values()) == 71 and len(requests) == 45, "fixed all71/45 scope changed")
    totals = {
        view: sum(len(r["mask_pairs"]) for r in requests if r["view"] == view) for view in VIEWS
    }
    require(
        totals == dict(zip(VIEWS, (200, 219, 281), strict=True)), "fixed700 mask pair scope changed"
    )
    crops = [
        dict(frame_idx=n, view=v, mask_row=i, crop_id=f"f{n}_{v}_m{i:03d}")
        for n in FRAMES
        for v in VIEWS
        for i in range(counts[n, v])
    ]
    request_rows, pair_rows = [], []
    for request_index, request in enumerate(requests):
        row = {k: v for k, v in request.items() if k != "mask_pairs"}
        row.update(request_index=request_index, mask_pair_count=len(request["mask_pairs"]))
        request_rows.append(row)
        for left, right in request["mask_pairs"]:
            pair_rows.append(
                dict(
                    pair_index=len(pair_rows),
                    request_index=request_index,
                    earlier_frame=row["earlier_frame"],
                    later_frame=row["later_frame"],
                    view=row["view"],
                    earlier_mask_row=int(left),
                    later_mask_row=int(right),
                    earlier_crop_id=f"f{row['earlier_frame']}_{row['view']}_m{left:03d}",
                    later_crop_id=f"f{row['later_frame']}_{row['view']}_m{right:03d}",
                )
            )
    return dict(
        schema=STAGE,
        frames=list(FRAMES),
        views=list(VIEWS),
        fixed_posthoc_frames=True,
        crops=crops,
        requests=request_rows,
        pairs=pair_rows,
        mask_pair_counts_by_view=totals,
        conditions=list(CONDITIONS),
        maximum_forward_calls=142,
        model_inference_in_this_stage=False,
        empty_mask_policy="retain row; NaN input; no forward; never substitute another mask",
        geometry_join=(
            "after image results only; prior partialD149 is evidence, never a crop-selection rule"
        ),
        first_crop=dict(zip(("frame_idx", "view", "mask_row"), FIRST, strict=True)),
    )


def final_guard(evidence):
    for path, digest in evidence.items():
        require(assets.sha(path) == digest, "final evidence guard changed: " + path)


def no_model_imports():
    require(
        not any(
            n == "torch" or n.startswith("torch.") or n == "dinov2" or n.startswith("dinov2.")
            for n in sys.modules
        ),
        "preparation must not import Torch/DINOv2",
    )


def runtime():
    return dict(
        python=sys.version,
        numpy=np.__version__,
        opencv=cv2.__version__,
        torch_distribution_version=importlib.metadata.version("torch"),
        model_imported=False,
    )


def run(output):
    output = Path(output).resolve()
    require(
        output.is_relative_to(WORK) and not output.exists(), "new workspace crop output required"
    )
    no_model_imports()
    evidence = {}
    code = (
        Path(__file__).resolve(),
        Path(partial.__file__).resolve(),
        Path(smoke.__file__).resolve(),
        Path(assets.__file__).resolve(),
        WORK / "DINO_PARTIAL_PROTOCOL.md",
        WORK / "LEARNED_ID_FEASIBILITY.md",
    )
    for path in code:
        bind(path, PINS.get(path, assets.sha(path)), evidence)
    inputs = load_inputs(evidence)
    manifest, asset_evidence = smoke.verify_assets(ASSET_ROOT, ASSET_SHA)
    for path, digest in asset_evidence.items():
        bind(path, digest, evidence)
    counts = {key: row["mask_count"] for key, row in inputs["observations"].items()}
    plan = make_plan(counts)
    plan.update(
        asset_manifest_sha256=assets.sha(ASSET_ROOT / "assets.manifest.json"),
        official_model=manifest["model"],
        official_commit=manifest["commit"],
        source_hashes={str(p): evidence[str(p)] for p in code},
    )
    environment, started = runtime(), assets.now()
    output.mkdir(parents=True)
    outputs = {}

    def record(path):
        path = Path(path)
        outputs[path.relative_to(output).as_posix()] = dict(
            path=str(path.resolve()), sha256=assets.sha(path), size_bytes=path.stat().st_size
        )

    def write(name, value):
        path = output / name
        assets.write_json(path, value)
        record(path)

    try:
        write("plan.json", plan)
        snapshots = output / "code_snapshot"
        snapshots.mkdir()
        for path in code:
            dest = snapshots / path.name
            shutil.copyfile(path, dest)
            require(assets.sha(dest) == evidence[str(path)], "loaded source snapshot changed")
            record(dest)
        (output / "crops").mkdir()
        crop_rows, native_rows = [], []
        for native in FRAMES:
            for view in VIEWS:
                png = RAW / f"raw_{native}_{view}.png"
                obs = RAW / f"observation_{native}_{view}.npz"
                image = cv2.imread(str(png), cv2.IMREAD_COLOR)
                # Deliberately do not deserialize saved poses or projected arrays.
                with np.load(obs, allow_pickle=False) as z:
                    masks, available, detections = (
                        z[k].copy() for k in ("masks", "mask_available", "detections")
                    )
                prior = inputs["descriptors"][native, view]
                raw_row = inputs["observations"][native, view]
                check_native(image, masks, available, detections, raw_row, prior)
                native_rows.append(
                    dict(
                        **raw_row,
                        png_sha256=evidence[str(png)],
                        observation_npz_sha256=evidence[str(obs)],
                        image_rgb_sha256=signature(np.ascontiguousarray(image[:, :, ::-1]))[
                            "sha256"
                        ],
                    )
                )
                for index, mask in enumerate(masks):
                    arrays, meta = prepared_crop(image, mask, prior["masks"][index])
                    crop_id = f"f{native}_{view}_m{index:03d}"
                    path = output / "crops" / (crop_id + ".npz")
                    with path.open("xb") as stream:
                        np.savez_compressed(stream, **arrays)
                    record(path)
                    meta.update(
                        frame_idx=native,
                        view=view,
                        mask_row=index,
                        crop_id=crop_id,
                        path=path.relative_to(output).as_posix(),
                        sha256=outputs[path.relative_to(output).as_posix()]["sha256"],
                        native_png_sha256=evidence[str(png)],
                        native_observation_npz_sha256=evidence[str(obs)],
                    )
                    crop_rows.append(meta)
                    if (native, view, index) == FIRST:
                        for condition in CONDITIONS:
                            first = output / ("first_crop_" + condition + "_normalized.npy")
                            with first.open("xb") as stream:
                                np.save(
                                    stream,
                                    arrays[condition + "_normalized_chw"],
                                    allow_pickle=False,
                                )
                            record(first)
        require(
            [r["crop_id"] for r in crop_rows] == [r["crop_id"] for r in plan["crops"]],
            "all71 crop order/presence changed",
        )
        write("native_evidence.json", native_rows)
        write("crops.json", crop_rows)
        first_row = crop_rows[0]
        require(
            (first_row["frame_idx"], first_row["view"], first_row["mask_row"]) == FIRST,
            "fixed first crop changed",
        )
        summary = dict(
            schema=STAGE,
            status="ok",
            frames=list(FRAMES),
            views=list(VIEWS),
            raw_png_decodes=18,
            raw_video_decodes=0,
            mask_crops=71,
            requests=45,
            mask_pairs=700,
            background_conditions=list(CONDITIONS),
            maximum_forward_calls=142,
            ready_crops=sum(r["crop_status"] == "ready" for r in crop_rows),
            empty_crops=sum(r["crop_status"] == "empty_mask" for r in crop_rows),
            zero_eligible_token_crops=sum(r["eligible_token_count"] == 0 for r in crop_rows),
            eligible_tokens=sum(r["eligible_token_count"] for r in crop_rows),
            all_token_rows=71 * 256,
            model_forward_calls=0,
            model_loaded=False,
            GPU=False,
            partialD_join_performed=False,
            original_partialD_finite_elements=149,
            fixed_first_crop=first_row,
            shared_SL_calibration_sha256=inputs["shared"],
            same68060_evidence=True,
            prior_sealed_horizon_references=inputs["horizon_references"],
            final_ID_truth=False,
            raw_input_verification_scope=(
                "all pinned prior outputs and directly reused input memberships/SHA; "
                "native18BGR/all71mask/SL signatures; upstream rawMP4 checks are prior "
                "sealed evidence, not repeated"
            ),
            limitations=[
                "Fixed posthoc difficult frames; not an independent encounter test.",
                "All masks remain anonymous, including empty, merged or reflected detections.",
                "Core-local token cells do not guarantee context-free Transformer features.",
                "Only crop preparation; model compatibility, learned correspondences "
                "and biological identity are not tested.",
            ],
        )
        write("summary.json", summary)
        report = output / "RESULT.md"
        report.write_text(
            "# 固定DINO入力準備\n\n18既存PNG・71全マスクを同じ整数crop規約で保存。"
            "全45要求/700組、背景2条件の最大142 forwardを先に固定しました。"
            "モデル実行は0回です。\n\n"
            f"非空{summary['ready_crops']}、空{summary['empty_crops']}、"
            f"採用token0のcrop {summary['zero_eligible_token_crops']}。"
            "全18176 token行を保存し、不能行を他マスクへ置き換えていません。\n\n"
            "first_crop_raw_normalized.npy は固定66524/top_cam/mask0です。"
            "空ならsmokeを行わず空を報告します。元RGB/背景置換の両CHWを保存しています。\n\n"
            "既存の部分D149と幾何参照は来歴として結合しただけで、crop選択に使っていません。"
            "最終モデル推論・700組の対応計算・条件付き幾何照合は別段階です。"
            "原MP4、3D候補、GTは読み込んでいません。\n",
            encoding="utf-8",
        )
        record(report)
        no_model_imports()
        require(runtime() == environment, "runtime changed during crop preparation")
        final_guard(evidence)
        for item in outputs.values():
            require(assets.sha(item["path"]) == item["sha256"], "output changed before commit")
        assets.write_json(
            output / "preparation.provenance.json",
            dict(
                stage=STAGE,
                status="ok",
                started_at=started,
                finished_at=assets.now(),
                inputs=evidence,
                outputs=outputs,
                params=plan,
                environment=environment,
            ),
        )
        return summary
    except BaseException as exc:
        assets.write_json(
            output / "failure.json",
            dict(
                stage=STAGE,
                status="failed",
                started_at=started,
                finished_at=assets.now(),
                error_type=type(exc).__name__,
                error=str(exc),
                inputs=evidence,
                completed_outputs=outputs,
            ),
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "mask_crops",
                    "ready_crops",
                    "empty_crops",
                    "mask_pairs",
                    "model_forward_calls",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
