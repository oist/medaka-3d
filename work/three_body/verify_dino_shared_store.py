"""Convert the sealed old142/1400 saved arrays and certify lossless shared storage.

No inference, images, old geometry labels, video, fit or GPU. Successful output
is a storage compatibility proof, not a new image/identity experiment.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import diagnose_dino_fixed_masks as original
import dino_shared_store as store
import numpy as np

prep, assets = original.prep, original.assets
WORK = prep.WORK
ROOT = WORK / "dino_fixed71_image_correspondence_v1"
INPUT_SHA = "6a975f9577b8af141e2da3660a91a4af79017a3a0b9fb88f0ce5ec9a18d87ae3"
SOURCE_SHA = "6d0d455e5c1c25d7dfba1f99666be7ec9e7442c41be7841c0084cabd3090e79a"
PLAN = WORK / "DINO_MINUTE_ADJACENT_PLAN.md"
PLAN_SHA = "55ad7c8ad7d719eaa9a81c4e9ce14e9471b497f34b687f8c9a62552902eb9517"
STAGE = "dino_shared_store_compatibility_v1"
require = store.require


def validate_membership(plan, inference, pairs):
    require(
        plan["schema"] == original.STAGE
        and plan["conditions"] == ["raw", "masked"]
        and len(plan["crops"]) == 71
        and len(plan["pairs"]) == 700
        and plan["maximum_forward_calls"] == 142
        and plan["pair_condition_records"] == 1400,
        "fixed old image plan differs",
    )
    expected = [(c["crop_id"], condition) for c in plan["crops"] for condition in prep.CONDITIONS]
    require(
        [(r["crop_id"], r["condition"]) for r in inference] == expected,
        "all142 token order required",
    )
    require(len(pairs) == 1400, "all1400 pairs required")
    for index, (p, r) in enumerate(
        (p, r)
        for p, rows in zip(
            plan["pairs"], [pairs[i : i + 2] for i in range(0, 1400, 2)], strict=True
        )
        for r in rows
    ):
        require(
            all(r[k] == v for k, v in p.items()) and r["condition"] == prep.CONDITIONS[index % 2],
            "all700x2 pair identity/order required",
        )


def convert_tokens(inference, read_arrays, writer):
    geometries = {}
    for row in inference:
        arrays = read_arrays(row)
        require(
            set(arrays) == set(original.GEOMETRY_KEYS) | {"final_layernorm_patch_tokens"},
            "all8 token members required",
        )
        tokens = arrays["final_layernorm_patch_tokens"]
        require(
            tokens.dtype == np.float32
            and tokens.shape == (256, 384)
            and store.signature(tokens) == row["token_signature"],
            "original full256 LayerNorm bytes differ",
        )
        geometry = {k: arrays[k] for k in original.GEOMETRY_KEYS}
        crop_id = row["crop_id"]
        signature = store.signatures(geometry)
        if crop_id not in geometries:
            writer.add_crop(crop_id, geometry)
            geometries[crop_id] = signature
        else:
            require(geometries[crop_id] == signature, "same crop geometry changed by condition")
        writer.add_feature(crop_id + "__" + row["condition"], crop_id, tokens)
    return geometries


def pair_id(row):
    return f"p{row['pair_index']:04d}_{row['condition']}"


def run(output):
    output = Path(output).resolve()
    require(
        output.is_relative_to(WORK) and not output.exists(), "new compatibility output required"
    )
    prep.no_model_imports()
    code = (
        Path(__file__).resolve(),
        Path(store.__file__).resolve(),
        Path(original.__file__).resolve(),
        Path(prep.__file__).resolve(),
        Path(original.partial.__file__).resolve(),
        Path(assets.__file__).resolve(),
        Path(original.smoke.__file__).resolve(),
        PLAN,
    )
    evidence = {}
    outputs = {}
    timings = {}
    started = assets.now()
    start = time.perf_counter()
    for path in code:
        prep.bind(
            path,
            SOURCE_SHA
            if path == Path(original.__file__).resolve()
            else PLAN_SHA
            if path == PLAN
            else assets.sha(path),
            evidence,
        )
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
        seal = original.verified(
            ROOT, "diagnosis.provenance.json", original.STAGE, INPUT_SHA, evidence
        )
        for path in code[2:-1]:
            original.flat_input(seal, path, evidence)
        timings["input_SHA_verification_seconds"] = time.perf_counter() - t

        def source_json(name):
            return prep.read_json(prep.sealed_output(ROOT, seal, name, evidence))

        plan, inference, pairs = (
            source_json(n) for n in ("plan.json", "inference_records.json", "pair_records.json")
        )
        validate_membership(plan, inference, pairs)
        for path in code:
            dest = output / "code_snapshot" / path.name
            dest.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, dest)
            require(assets.sha(dest) == evidence[str(path)], "source snapshot changed")
            record(dest)
        bindings = dict(
            source_stage=original.STAGE,
            source_seal_sha256=INPUT_SHA,
            source_plan_sha256=evidence[str(ROOT / "plan.json")],
            codec_source_sha256=evidence[str(Path(store.__file__).resolve())],
            approved_plan_sha256=PLAN_SHA,
            model_calls=0,
            geometry_join=False,
        )
        write(
            "registration.json",
            dict(
                stage=STAGE,
                bindings=bindings,
                source_hashes={str(p): evidence[str(p)] for p in code},
                tokens142=True,
                pair_conditions1400=True,
                logical_arrays_per_pair=32,
                lossless_dtype_shape_bytes=True,
            ),
        )
        writer = store.StoreWriter(output / "shared_store", bindings=bindings)

        def load(row):
            path = prep.sealed_output(ROOT, seal, row["path"], evidence)
            require(evidence[str(path)] == row["sha256"], "input row SHA differs")
            return original.npz(path)

        t = time.perf_counter()
        geometries = convert_tokens(inference, load, writer)
        timings["shared_token_normalization_save_seconds"] = time.perf_counter() - t
        require(len(geometries) == 71, "all71 shared geometries required")
        t = time.perf_counter()
        for i, row in enumerate(pairs):
            writer.add_pair(
                pair_id(row),
                row["earlier_crop_id"] + "__" + row["condition"],
                row["later_crop_id"] + "__" + row["condition"],
                load(row),
                row["metadata"],
            )
            if (i + 1) % 200 == 0:
                print(
                    json.dumps(dict(event="pairs_converted", completed=i + 1, total=1400)),
                    flush=True,
                )
        writer.finish()
        timings["pair_conversion_and_commit_seconds"] = time.perf_counter() - t
        reader = store.StoreReader(
            output / "shared_store",
            expected_bindings=bindings,
            expected_manifest_sha256=assets.sha(output / "shared_store/store.manifest.json"),
        )
        rows = []
        t = time.perf_counter()
        for i, row in enumerate(pairs):
            expected = load(row)
            actual, metadata = reader.read_pair(pair_id(row))
            require(
                store.signatures(expected) == store.signatures(actual),
                "independent fullarray restoration differs",
            )
            require(
                store.canonical(metadata) == store.canonical(row["metadata"]),
                "scalar/array metadata differs",
            )
            rows.append(
                dict(
                    pair_index=row["pair_index"],
                    condition=row["condition"],
                    all32_arrays_byte_exact=True,
                    metadata_canonical_byte_exact=True,
                    logical_arrays=store.signatures(actual),
                )
            )
            if (i + 1) % 200 == 0:
                print(
                    json.dumps(dict(event="pairs_restored", completed=i + 1, total=1400)),
                    flush=True,
                )
        for row in inference:
            restored = reader.read_feature(row["crop_id"] + "__" + row["condition"])
            require(
                store.signature(restored["layernorm_tokens"]) == row["token_signature"],
                "shared original tokens drift",
            )
        sizes = reader.verify_all()
        timings["fresh_read_restore_all1400_seconds"] = time.perf_counter() - t
        write("pair_restoration_checks.json", rows)
        write("original_inference_records.json", inference)
        write(
            "original_pair_index.json",
            [{k: v for k, v in r.items() if k != "metadata"} for r in pairs],
        )
        for path in (output / "shared_store").rglob("*"):
            if path.is_file():
                record(path)
        old_pair_bytes = sum((ROOT / r["path"]).stat().st_size for r in pairs)
        old_token_bytes = sum((ROOT / r["path"]).stat().st_size for r in inference)
        t = time.perf_counter()
        prep.final_guard(evidence)
        for item in outputs.values():
            require(assets.sha(item["path"]) == item["sha256"], "new output changed")
        prep.no_model_imports()
        timings["final_SHA_guard_seconds"] = time.perf_counter() - t
        timings["total_through_guards_seconds"] = time.perf_counter() - start
        result = dict(
            schema=STAGE,
            status="ok",
            passed=True,
            shared_crops=71,
            shared_conditions=142,
            pair_conditions=1400,
            logical_arrays_per_pair=32,
            logical_arrays_restored=44800,
            all_pairs_byte_exact=True,
            all_scalar_metadata_exact=True,
            original_token_bytes_exact=True,
            source_pair_npz_bytes=old_pair_bytes,
            source_token_npz_bytes=old_token_bytes,
            shared_store_bytes=sizes["bytes"],
            shared_store_files=sizes["files"],
            shared_store_reduction_vs_old_pair_and_token_percent=100
            * (1 - sizes["bytes"] / (old_pair_bytes + old_token_bytes)),
            full_distances_and_all_NN_ties_preserved=True,
            model_calls=0,
            raw_images_opened=0,
            geometry_join=False,
            GPU=False,
            ID_or_review_changed=False,
            timings=timings,
            timing_scope="I/O, SHA, conversion and checks; no inference timing",
            manifest_sha256=reader.manifest_sha256,
        )
        write("summary.json", result)
        report = output / "RESULT.md"
        report.write_text(
            "# DINO共有保存の互換検証\n\n既存142条件・全1400pairの32論理配列(44,800配列)と"
            "全scalar metadataをlossless復元。元全256 LayerNorm tokensも保持。\n\n"
            f"旧pair+token NPZ {old_pair_bytes + old_token_bytes:,} bytes、"
            "共有store(manifest含む)"
            f"{sizes['bytes']:,} bytes。監査レポートやsnapshotの容量はこの比較に含めない。\n\n"
            "新しいモデル推論・画像decode・幾何結合・ID操作は行っていない。\n",
            encoding="utf-8",
        )
        record(report)
        assets.write_json(
            output / "compatibility.provenance.json",
            dict(
                stage=STAGE,
                status="ok",
                started_at=started,
                finished_at=assets.now(),
                inputs=evidence,
                outputs=outputs,
                model_calls=0,
                raw_images_opened=0,
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
    result = run(p.parse_args().output)
    print(
        json.dumps(
            {k: result[k] for k in ("status", "all_pairs_byte_exact", "shared_store_bytes")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
