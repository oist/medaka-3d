"""Fixed593 crops /1186 CPU forwards /3868 adjacent mask-pair conditions.

Same official CPU model and frozen correspondence arithmetic; new shared storage.
All anonymous masks remain. No video decoding, fit, geometry join or ID change.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import OrderedDict
from pathlib import Path

import diagnose_dino_fixed_masks as old
import dino_shared_store as store
import numpy as np
import prepare_dino_minute_crops as crops

prep, assets = old.prep, old.assets
WORK = prep.WORK
STAGE = "dino_minute_adjacent_image_correspondence_v1"
PREPARATION = WORK / "dino_minute_adjacent_crop_preparation_v1"
PREPARATION_SHA = "292b0805d2a15bf04fd839cf68357332ab6dedd6b669477fb98b311aa5ee6914"
COMPATIBILITY = WORK / "dino_shared_store_compatibility_v1"
PINS = {
    Path(
        old.__file__
    ).resolve(): "6d0d455e5c1c25d7dfba1f99666be7ec9e7442c41be7841c0084cabd3090e79a",
    Path(
        store.__file__
    ).resolve(): "891f8726b9c80fe017367007b2b339fc1d43cf49c3ae611f3b5a93d5496941d2",
    Path(
        crops.__file__
    ).resolve(): "47913aba5d44d7cd0516b75693b72bb4bce786f55cfea662f4b8be10bf8cb926",
    **prep.PINS,
    crops.PLAN: crops.PLAN_SHA,
}
require = store.require


def validate_preparation(plan, rows, natives, summary):
    require(
        plan["schema"] == crops.STAGE
        and plan["frames"] == list(crops.FRAMES)
        and plan["views"] == list(crops.VIEWS)
        and plan["conditions"] == list(crops.CONDITIONS),
        "fixed minute crop plan differs",
    )
    keys = [(r["frame_idx"], r["view"]) for r in natives]
    require(
        keys == [(n, v) for n in crops.FRAMES for v in crops.VIEWS],
        "all180 native evidence rows required",
    )
    expected = crops.enumerate_plan(
        {key: r["mask_count"] for key, r in zip(keys, natives, strict=True)}
    )
    for key in ("crops", "requests", "pairs", "conditions", "maximum_forward_calls"):
        require(plan[key] == expected[key], "adjacent plan identity differs: " + key)
    require(
        len(rows) == 593
        and len(plan["pairs"]) == 1934
        and plan["maximum_forward_calls"] == 1186
        and len(plan["requests"]) == 177,
        "actual fixed593/1934 cardinality differs",
    )
    require(
        [{k: r[k] for k in ("frame_idx", "view", "mask_row", "crop_id")} for r in rows]
        == plan["crops"],
        "all593 ordered mask rows required",
    )
    require(
        summary["status"] == "ok"
        and summary["mask_crops"] == 593
        and summary["mask_pairs"] == 1934
        and summary["model_forward_calls"] == 0
        and summary["candidate_arrays_read"] is False
        and summary["raw_video_delivered_images"] == 180
        and summary["geometry_join_performed"] is False,
        "completed independent crop stage required",
    )


def validate_compatibility(summary, registration):
    require(
        summary.get("schema") == "dino_shared_store_compatibility_v1"
        and summary.get("status") == "ok"
        and all(
            summary.get(k) is True
            for k in (
                "passed",
                "all_pairs_byte_exact",
                "all_scalar_metadata_exact",
                "original_token_bytes_exact",
                "full_distances_and_all_NN_ties_preserved",
            )
        )
        and summary.get("shared_crops") == 71
        and summary.get("shared_conditions") == 142
        and summary.get("pair_conditions") == 1400
        and summary.get("logical_arrays_restored") == 44800
        and summary.get("model_calls") == 0
        and summary.get("raw_images_opened") == 0,
        "full successful old142/1400 compatibility proof required",
    )
    require(
        registration["bindings"]["codec_source_sha256"] == PINS[Path(store.__file__).resolve()]
        and registration["bindings"]["approved_plan_sha256"] == crops.PLAN_SHA,
        "qualified codec or plan changed",
    )


def infer_prepared(rows, read_arrays, forward, save):
    """CPU callback only. Every nonempty crop, even eligible0, receives both forwards."""
    records = []
    for row in rows:
        arrays = read_arrays(row)
        before = store.signatures(arrays)
        require(
            set(arrays) == set(prep.ARRAY_SHAPES) and before == row["array_signatures"],
            "complete original19 prepared arrays required",
        )
        ready = row["crop_status"] == "ready"
        require(ready or row["crop_status"] == "empty_mask", "unknown crop status")
        geometry = {k: arrays[k].copy() for k in old.GEOMETRY_KEYS}
        for condition in crops.CONDITIONS:
            chw = arrays[condition + "_normalized_chw"]
            require(
                chw.dtype == np.float32
                and chw.shape == (3, 224, 224)
                and (np.isfinite(chw).all() if ready else np.isnan(chw).all()),
                "unchanged prepared CHW input required",
            )
            if ready:
                tokens, elapsed = forward(chw)
                require(np.isfinite(elapsed) and elapsed >= 0, "forward timing invalid")
            else:
                tokens, elapsed = np.full((256, 384), np.nan, np.float32), None
            require(
                isinstance(tokens, np.ndarray)
                and tokens.dtype == np.float32
                and tokens.shape == (256, 384),
                "all256 float32 LayerNorm tokens required",
            )
            require(store.signatures(arrays) == before, "model mutated prepared arrays")
            tokens = tokens.copy()
            save(row, condition, tokens, geometry)
            records.append(
                dict(
                    crop_id=row["crop_id"],
                    frame_idx=row["frame_idx"],
                    view=row["view"],
                    mask_row=row["mask_row"],
                    condition=condition,
                    crop_status=row["crop_status"],
                    token_status=row["token_status"],
                    eligible_token_count=row["eligible_token_count"],
                    forward_called=ready,
                    forward_seconds=elapsed,
                    token_signature=store.signature(tokens),
                    finite_token_count=int(np.isfinite(tokens).all(-1).sum()),
                    normalized_input_signature=store.signature(chw),
                )
            )
    return records


class PairSource:
    """Bounded64-feature cache; saved geometry/feature file SHA validated on every miss."""

    def __init__(self, writer):
        self.writer = writer
        self.cache = OrderedDict()

    def read(self, feature_id):
        ref = self.writer.index["features"][feature_id]
        if feature_id not in self.cache:
            features = store.read_npz(self.writer.root, ref)
            geometry = store.read_npz(self.writer.root, self.writer.index["crops"][ref["crop_id"]])
            self.cache[feature_id] = (features["layernorm_tokens"], geometry)
        self.cache.move_to_end(feature_id)
        if len(self.cache) > 64:
            self.cache.popitem(last=False)
        return self.cache[feature_id]


def calculate_pairs(plan, source, save, on_progress=None):
    rows = []
    for pair in plan["pairs"]:
        for condition in crops.CONDITIONS:
            left_id, right_id = (
                pair[k] + "__" + condition for k in ("earlier_crop_id", "later_crop_id")
            )
            left, lg = source.read(left_id)
            right, rg = source.read(right_id)
            before = (
                store.signature(left),
                store.signature(right),
                store.signatures(lg),
                store.signatures(rg),
            )
            result = old.partial.token_correspondence(left, right, lg, rg)
            arrays, metadata = old.pack_correspondence(result)
            require(
                before
                == (
                    store.signature(left),
                    store.signature(right),
                    store.signatures(lg),
                    store.signatures(rg),
                ),
                "correspondence mutated shared input",
            )
            key = f"p{pair['pair_index']:04d}_{condition}"
            save(key, left_id, right_id, arrays, metadata)
            rows.append(
                dict(
                    **pair,
                    condition=condition,
                    status=result["status"],
                    comparable=result["comparable"],
                    shared_pair_id=key,
                    metadata=metadata,
                )
            )
        if on_progress is not None:
            on_progress(len(rows))
    return rows


def run(output, compatibility_sha256):
    output = Path(output).resolve()
    require(output.is_relative_to(WORK) and not output.exists(), "new image output required")
    require(
        isinstance(compatibility_sha256, str)
        and len(compatibility_sha256) == 64
        and all(c in "0123456789abcdef" for c in compatibility_sha256),
        "explicit successful compatibility SHA required",
    )
    prep.no_model_imports()
    code = (
        Path(__file__).resolve(),
        Path(old.__file__).resolve(),
        Path(store.__file__).resolve(),
        Path(crops.__file__).resolve(),
        Path(prep.__file__).resolve(),
        Path(old.partial.__file__).resolve(),
        Path(assets.__file__).resolve(),
        Path(old.smoke.__file__).resolve(),
        crops.PLAN,
    )
    evidence = {}
    outputs = {}
    timings = {}
    start = time.perf_counter()
    started = assets.now()
    for path in code:
        prep.bind(path, PINS.get(path, assets.sha(path)), evidence)
    output.mkdir(parents=True)

    def record(path):
        outputs[path.relative_to(output).as_posix()] = dict(
            path=str(path), sha256=assets.sha(path), size_bytes=path.stat().st_size
        )

    def write(name, value):
        path = output / name
        assets.write_json(path, value)
        record(path)

    try:
        t = time.perf_counter()
        preparation = old.verified(
            PREPARATION, "preparation.provenance.json", crops.STAGE, PREPARATION_SHA, evidence
        )
        compatibility = old.verified(
            COMPATIBILITY,
            "compatibility.provenance.json",
            "dino_shared_store_compatibility_v1",
            compatibility_sha256,
            evidence,
        )

        def saved_json(root, seal, name):
            return prep.read_json(prep.sealed_output(root, seal, name, evidence))

        plan, rows, natives, summary = (
            saved_json(PREPARATION, preparation, n)
            for n in ("plan.json", "crops.json", "native_evidence.json", "summary.json")
        )
        validate_preparation(plan, rows, natives, summary)
        validate_compatibility(
            saved_json(COMPATIBILITY, compatibility, "summary.json"),
            saved_json(COMPATIBILITY, compatibility, "registration.json"),
        )
        for path in code[1:-1]:
            if str(path) in preparation["inputs"]:
                old.flat_input(preparation, path, evidence)
            if str(path) in compatibility["inputs"]:
                old.flat_input(compatibility, path, evidence)
        old.flat_input(preparation, Path(crops.__file__).resolve(), evidence)
        old.flat_input(compatibility, Path(store.__file__).resolve(), evidence)
        smoke_seal = old.verified(
            old.SMOKE_ROOT, "smoke.provenance.json", old.smoke.STAGE, old.SMOKE_SHA, evidence
        )
        smoke_summary = saved_json(old.SMOKE_ROOT, smoke_seal, "summary.json")
        old.validate_smoke(smoke_summary)
        _, asset_evidence = old.smoke.verify_assets(prep.ASSET_ROOT, prep.ASSET_SHA)
        for path, digest in asset_evidence.items():
            prep.bind(path, digest, evidence)
        environment = assets.runtime()
        timings["input_verification_seconds"] = time.perf_counter() - t
        for path in code:
            dest = output / "code_snapshot" / path.name
            dest.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, dest)
            require(assets.sha(dest) == evidence[str(path)], "snapshot changed")
            record(dest)
        bindings = dict(
            preparation_sha256=PREPARATION_SHA,
            compatibility_sha256=compatibility_sha256,
            source_plan_sha256=crops.PLAN_SHA,
            codec_source_sha256=PINS[Path(store.__file__).resolve()],
            original_model_runner_sha256=PINS[Path(old.__file__).resolve()],
            device="cpu",
            dtype="float32",
            torch_cpu_threads=2,
        )
        write(
            "plan.json",
            dict(
                schema=STAGE,
                bindings=bindings,
                frames=plan["frames"],
                views=plan["views"],
                crops=plan["crops"],
                requests=plan["requests"],
                pairs=plan["pairs"],
                conditions=list(crops.CONDITIONS),
                maximum_forward_calls=1186,
                pair_condition_records=3868,
                geometry_join_performed=False,
                posthoc_after_fixed6_results=True,
                temporal_holdout=False,
                biological_ID_truth=False,
            ),
        )
        writer = store.StoreWriter(output / "shared_store", bindings=bindings)
        t = time.perf_counter()
        model = old.CPUModel(evidence, smoke_summary)
        timings["model_load_seconds"] = time.perf_counter() - t
        saved_crops = set()
        save_seconds = 0.0

        def load_crop(row):
            path = prep.sealed_output(PREPARATION, preparation, row["path"], evidence)
            require(evidence[str(path)] == row["sha256"], "prepared crop SHA differs")
            return old.npz(path)

        def save_feature(row, condition, tokens, geometry):
            nonlocal save_seconds
            t = time.perf_counter()
            key = row["crop_id"]
            if key not in saved_crops:
                writer.add_crop(key, geometry)
                saved_crops.add(key)
            writer.add_feature(key + "__" + condition, key, tokens)
            save_seconds += time.perf_counter() - t
            if len(writer.index["features"]) % 100 == 0:
                print(
                    json.dumps(
                        dict(
                            event="CPU_features_saved",
                            count=len(writer.index["features"]),
                            total=1186,
                        )
                    ),
                    flush=True,
                )

        t = time.perf_counter()
        inference = infer_prepared(rows, load_crop, model, save_feature)
        timings["inference_load_save_wall_seconds"] = time.perf_counter() - t
        timings["shared_feature_normalization_save_seconds"] = save_seconds
        timings["model_forward_seconds"] = sum(r["forward_seconds"] or 0.0 for r in inference)
        require(
            len(inference) == 1186 and len(saved_crops) == 593, "all593x2 inference rows required"
        )
        write("inference_records.json", inference)
        model.verify_loaded(evidence)
        source = PairSource(writer)
        t = time.perf_counter()
        pairs = calculate_pairs(
            plan,
            source,
            writer.add_pair,
            lambda count: (
                print(
                    json.dumps(
                        dict(event="adjacent_pair_conditions_saved", count=count, total=3868)
                    ),
                    flush=True,
                )
                if count % 200 == 0
                else None
            ),
        )
        require(len(pairs) == 3868, "full adjacent pair denominator required")
        writer.finish()
        source.cache.clear()
        timings["pair_arithmetic_save_commit_seconds"] = time.perf_counter() - t
        write("pair_records.json", pairs)
        reader = store.StoreReader(
            writer.root,
            expected_bindings=bindings,
            expected_manifest_sha256=assets.sha(writer.root / "store.manifest.json"),
        )
        t = time.perf_counter()
        sizes = reader.verify_all()
        timings["shared_store_SHA_verification_seconds"] = time.perf_counter() - t
        for path in writer.root.rglob("*"):
            if path.is_file():
                record(path)
        model.verify_loaded(evidence)
        require(assets.runtime() == environment, "runtime drift")
        t = time.perf_counter()
        prep.final_guard(evidence)
        for row in outputs.values():
            require(assets.sha(row["path"]) == row["sha256"], "output changed before seal")
        timings["final_SHA_guard_seconds"] = time.perf_counter() - t
        timings["total_through_guards_seconds"] = time.perf_counter() - start
        result = dict(
            schema=STAGE,
            status="ok",
            mask_crops=593,
            background_conditions=list(crops.CONDITIONS),
            inference_rows=1186,
            forward_calls=sum(r["forward_called"] for r in inference),
            empty_crop_rows=sum(r["crop_status"] == "empty_mask" for r in rows),
            zero_eligible_token_crops=sum(r["eligible_token_count"] == 0 for r in rows),
            geometry_token_rows=593 * 256,
            all_token_rows=1186 * 256,
            requested_adjacent_time_pairs=59,
            requested_pair_views=177,
            mask_pairs=1934,
            pair_condition_records=3868,
            numerically_comparable_pairs=sum(r["comparable"] for r in pairs),
            pair_status_counts={
                s: sum(r["status"] == s for r in pairs)
                for s in sorted({r["status"] for r in pairs})
            },
            shared_store_bytes=sizes["bytes"],
            shared_store_manifest_sha256=reader.manifest_sha256,
            full256_tokens_and_all_NN_ties=True,
            geometry_join_performed=False,
            raw_video_decodes=0,
            model_training=False,
            GPU=False,
            ID_or_review_changed=False,
            independent_ID_truth=False,
            temporal_holdout=False,
            timings=timings,
            environment=environment,
            limitations=[
                "All anonymous mask pairs; numerical NN is not correct biological identity.",
                "Missing and eligible0 crops/pairs remain; no time substitution.",
                "Expansion chosen after six-frame results; no temporal holdout claim.",
            ],
        )
        write("summary.json", result)
        assets.write_json(
            output / "diagnosis.provenance.json",
            dict(
                stage=STAGE,
                status="ok",
                started_at=started,
                finished_at=assets.now(),
                inputs=evidence,
                outputs=outputs,
                environment=environment,
            ),
        )
        return result
    except BaseException as exc:
        assets.write_json(
            output / "failure.json",
            dict(
                stage=STAGE,
                status="failed",
                error_type=type(exc).__name__,
                error=str(exc),
                inputs=evidence,
                completed_outputs=outputs,
                timings=timings,
            ),
        )
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--compatibility-sha256", required=True)
    args = p.parse_args()
    result = run(args.output, args.compatibility_sha256)
    print(
        json.dumps({k: result[k] for k in ("status", "forward_calls", "pair_condition_records")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
