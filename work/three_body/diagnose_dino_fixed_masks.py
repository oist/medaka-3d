"""CPU-only fixed71-crop/1400-pair image evidence, before any geometry join.

The completed crop plan is mandatory. Official pinned DINOv2, batch1 float32,
CPU2 and the successful smoke's exact model path are used. All256 patch tokens,
all256x256 pair distances and every exact NN tie remain. No new images, crops,
models, training, GPU, biological labels or geometry-dependent selection.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import prepare_dino_fixed_crops as prep

assets, partial, smoke = prep.assets, prep.partial, prep.smoke
WORK = prep.WORK
STAGE = "dino_fixed71_image_correspondence_v1"
PREPARATION = WORK / "dino_fixed71_crop_preparation_v1"
PREPARATION_SHA = "92ff2b384628eb4cfb6126707fdbab5d67fa81b1c54d4eb3a6b130b32d7c2e47"
FIRST_ROOT = WORK / "dino_first_crop_smoke_input_v1"
FIRST_SHA = "d24b0d18b816b9903763e538dd9d4c7b19b738823d3837dd8e2eecd751ba3993"
SMOKE_ROOT = WORK / "dino_first_crop_cpu_smoke_v1"
SMOKE_SHA = "2aa3452e8aad45390a23361ffd9c6f3a84d15c9d39e08ca324d81ab51cb2e3b3"
PREP_SOURCE_SHA = "eeea2cff12f8dca71b1a553960bfc0ef6380361f0c3ec090cc74757e748d36d0"
GEOMETRY_KEYS = (
    "token_index",
    "token_center_crop_xy",
    "token_center_native_xy",
    "token_native_xyxy",
    "token_footprint_pixels",
    "token_core_pixels",
    "token_eligible",
)
require = prep.require


def byte_exact(left, right):
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and left.tobytes() == right.tobytes()
    )


def flat_input(seal, path, evidence):
    """New preparation/smoke seals use path->SHA, unlike older record seals."""
    path = Path(path).resolve(strict=True)
    digest = seal["inputs"].get(str(path))
    require(isinstance(digest, str), "explicit flat input binding absent")
    return prep.bind(path, digest, evidence)


def verified(root, filename, stage, expected, evidence):
    root = Path(root).resolve(strict=True)
    path = prep.bind(root / filename, expected, evidence)
    seal = prep.read_json(path)
    # Root's fixed first-crop v1 seal deliberately has no timestamp fields.
    # Only that exact root/stage/seal SHA schema is exempt; main prep/smoke are not.
    first_schema = (
        root == FIRST_ROOT.resolve()
        and filename == "preparation.provenance.json"
        and stage == "dino_first_crop_smoke_input_v1"
        and expected == FIRST_SHA
    )
    require(
        seal.get("stage") == stage
        and seal.get("status") == "ok"
        and (first_schema or bool(seal.get("finished_at"))),
        "successful completed evidence required",
    )
    for name, item in seal["outputs"].items():
        output = (root / name).resolve(strict=True)
        require(
            output.is_relative_to(root) and Path(item["path"]).resolve() == output,
            "evidence output role/path drift",
        )
        size = item.get("size_bytes", item.get("size"))
        require(size is None or output.stat().st_size == size, "output size drift")
        prep.bind(output, item["sha256"], evidence)
    return seal


def npz(path):
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key].copy() for key in values.files}


def validate_smoke(summary):
    exact = dict(
        schema=smoke.STAGE,
        status="ok",
        model=assets.MODEL,
        commit=assets.COMMIT,
        crop_count=1,
        forward_calls=2,
        repeat_same_crop_byte_exact=True,
        patch_shape=[1, 256, 384],
        final_layernorm_not_global_pool=True,
        device="cpu",
        dtype="float32",
        torch_cpu_threads=2,
        XFORMERS_DISABLED=True,
        torch_load_weights_only=True,
        strict_state_load=True,
        GPU_used=False,
        training=False,
    )
    require(
        all(summary.get(k) == v and type(summary[k]) is type(v) for k, v in exact.items()),
        "exact successful CPU smoke configuration required",
    )
    require(summary["runtime"] == assets.runtime(), "CPU smoke software/runtime drift")
    require(
        summary["XFORMERS_module_flags"]
        == {
            name: dict(enabled=False, available=False)
            for name in ("attention", "block", "swiglu_ffn")
        },
        "smoke used another attention backend",
    )
    require(
        summary["preprocessing"]["resize_in_this_runner"] is False
        and summary["preprocessing"]["normalization_in_this_runner"] is False,
        "same prepared CHW convention required",
    )


def validate_crop_plan(plan, rows):
    require(
        plan["schema"] == prep.STAGE
        and plan["frames"] == list(prep.FRAMES)
        and plan["views"] == list(prep.VIEWS),
        "fixed crop scope changed",
    )
    counts = {(n, v): 0 for n in prep.FRAMES for v in prep.VIEWS}
    for row in rows:
        key = row["frame_idx"], row["view"]
        require(key in counts and type(row["mask_row"]) is int, "unregistered crop row")
        counts[key] += 1
    expected = prep.make_plan(counts)
    for key in (
        "crops",
        "requests",
        "pairs",
        "conditions",
        "maximum_forward_calls",
        "first_crop",
        "mask_pair_counts_by_view",
    ):
        require(plan[key] == expected[key], "frozen crop/pair plan drift: " + key)
    require(
        [{k: r[k] for k in ("frame_idx", "view", "mask_row", "crop_id")} for r in rows]
        == expected["crops"],
        "all71 crop row/order mismatch",
    )


def load_inputs(expected_prep, evidence):
    prep.no_model_imports()
    require(expected_prep == PREPARATION_SHA, "fixed completed all71 preparation required")
    for path, digest in {**prep.PINS, Path(prep.__file__).resolve(): PREP_SOURCE_SHA}.items():
        prep.bind(path, digest, evidence)
    seal = verified(PREPARATION, "preparation.provenance.json", prep.STAGE, expected_prep, evidence)
    # All source/input bytes actually used by preparation remain bound. These
    # comprise saved images/SL/asset files, not original MP4 or candidates.
    for path in seal["inputs"]:
        flat_input(seal, path, evidence)
    plan = prep.read_json(prep.sealed_output(PREPARATION, seal, "plan.json", evidence))
    rows = prep.read_json(prep.sealed_output(PREPARATION, seal, "crops.json", evidence))
    summary = prep.read_json(prep.sealed_output(PREPARATION, seal, "summary.json", evidence))
    validate_crop_plan(plan, rows)
    require(
        summary["model_forward_calls"] == 0
        and summary["model_loaded"] is False
        and summary["GPU"] is False
        and summary["partialD_join_performed"] is False,
        "pure prepared image inputs required",
    )
    for row in rows:
        require(
            row["path"] == f"crops/{row['crop_id']}.npz",
            "crop file must follow registered anonymous row",
        )
        path = prep.sealed_output(PREPARATION, seal, row["path"], evidence)
        require(evidence[str(path)] == row["sha256"], "crop row/file digest mismatch")
    first_seal = verified(
        FIRST_ROOT,
        "preparation.provenance.json",
        "dino_first_crop_smoke_input_v1",
        FIRST_SHA,
        evidence,
    )
    first_metadata = prep.read_json(
        prep.sealed_output(FIRST_ROOT, first_seal, "metadata.json", evidence)
    )
    first_scope = dict(
        fixed_frame=66524,
        fixed_view="top_cam",
        fixed_mask_row=0,
        source_mask_count=4,
        cropped_mask_count=1,
        model_load=False,
        GPU_used=False,
    )
    require(
        all(first_seal.get(key) == value for key, value in first_scope.items()),
        "registered independent first-crop scope changed",
    )
    require(
        all(rows[0].get(key) == value for key, value in first_metadata.items()),
        "first fixed native crop metadata differs from all71 preparation",
    )
    for path in (
        prep.RAW / "review.provenance.json",
        Path(partial.__file__).resolve(),
        WORK / "DINO_PARTIAL_PROTOCOL.md",
    ):
        flat_input(first_seal, path, evidence)
    first = {}
    for condition in prep.CONDITIONS:
        path = prep.sealed_output(
            FIRST_ROOT, first_seal, condition + "_normalized_chw.npy", evidence
        )
        other = prep.sealed_output(
            PREPARATION, seal, "first_crop_" + condition + "_normalized.npy", evidence
        )
        left, right = np.load(path, allow_pickle=False), np.load(other, allow_pickle=False)
        require(
            byte_exact(left, right),
            "all71 first crop differs from independently prepared smoke input",
        )
        require(evidence[str(path)] == evidence[str(other)], "first NPY files differ bytewise")
        first[condition] = left.copy()
    smoke_seal = verified(SMOKE_ROOT, "smoke.provenance.json", smoke.STAGE, SMOKE_SHA, evidence)
    for path in smoke_seal["inputs"]:
        flat_input(smoke_seal, path, evidence)
    smoke_summary = prep.read_json(
        prep.sealed_output(SMOKE_ROOT, smoke_seal, "summary.json", evidence)
    )
    validate_smoke(smoke_summary)
    golden = npz(prep.sealed_output(SMOKE_ROOT, smoke_seal, "patch_tokens.npz", evidence))
    require(
        byte_exact(golden["normalized_input"], smoke.prepare_input(first["raw"])),
        "smoke belongs to another normalized input",
    )
    require(
        golden["final_layernorm_patch_tokens"].dtype == np.float32
        and golden["final_layernorm_patch_tokens"].shape == (1, 256, 384),
        "original smoke LayerNorm field required",
    )
    require(
        smoke_summary["crop_sha256"] == evidence[str(FIRST_ROOT / "raw_normalized_chw.npy")],
        "smoke file binding mismatch",
    )
    _, verified_assets = smoke.verify_assets(prep.ASSET_ROOT, prep.ASSET_SHA)
    for path, digest in verified_assets.items():
        prep.bind(path, digest, evidence)
    require(
        smoke_summary["asset_manifest_sha256"]
        == plan["asset_manifest_sha256"]
        == assets.sha(prep.ASSET_ROOT / "assets.manifest.json"),
        "same official assets required",
    )
    return dict(plan=plan, rows=rows, first=first, golden=golden, smoke=smoke_summary)


class CPUModel:
    """Same official local model operations as frozen successful CPU smoke."""

    def __init__(self, evidence, smoke_summary):
        prep.no_model_imports()
        os.environ["XFORMERS_DISABLED"] = "1"
        sys.dont_write_bytecode = True
        self.source_root = (prep.ASSET_ROOT / "source").resolve(strict=True)
        sys.path.insert(0, str(self.source_root))
        import torch

        self.torch = torch
        torch.set_num_threads(2)
        torch.random.default_generator.manual_seed(0)
        require(not torch.cuda.is_initialized(), "CPU-only model required")
        backbones = importlib.import_module("dinov2.hub.backbones")
        for name in ("attention", "block", "swiglu_ffn"):
            module = importlib.import_module("dinov2.layers." + name)
            require(
                not module.XFORMERS_ENABLED and not module.XFORMERS_AVAILABLE,
                "official CPU non-xFormers path required",
            )
        model = backbones.dinov2_vits14(pretrained=False)
        state = torch.load(
            prep.ASSET_ROOT / "weights/dinov2_vits14_pretrain.pth",
            map_location="cpu",
            weights_only=True,
        )
        require(
            isinstance(state, dict)
            and all(
                isinstance(k, str) and isinstance(v, torch.Tensor) and v.device.type == "cpu"
                for k, v in state.items()
            ),
            "plain local CPU weights only; no pickle fallback",
        )
        model.load_state_dict(state, strict=True)
        del state
        model.eval().requires_grad_(False)
        model.to(device="cpu", dtype=torch.float32)
        self.model = model
        self.smoke_source = smoke_summary["loaded_source"]
        self.verify_loaded(evidence)

    def verify_loaded(self, evidence):
        loaded = {}
        for name, module in tuple(sys.modules.items()):
            if name == "dinov2" or name.startswith("dinov2."):
                path = Path(module.__file__).resolve(strict=True)
                require(
                    path.is_relative_to(self.source_root) and str(path) in evidence,
                    "unregistered official module imported",
                )
                loaded[name] = dict(path=str(path), sha256=assets.sha(path))
                require(
                    loaded[name]["sha256"] == evidence[str(path)], "loaded official source changed"
                )
        require(
            loaded == self.smoke_source, "official import closure differs from successful smoke"
        )
        require(
            not self.torch.cuda.is_initialized() and self.torch.get_num_threads() == 2,
            "CPU2 runtime changed",
        )
        return loaded

    def __call__(self, chw):
        tensor = self.torch.from_numpy(smoke.prepare_input(chw))
        with self.torch.inference_mode():
            begin = time.perf_counter()
            result = self.model.forward_features(tensor)
            elapsed = time.perf_counter() - begin
            tokens = result["x_norm_patchtokens"].detach().cpu().clone().numpy()[0].copy()
        return tokens, elapsed


def crop_arrays(row):
    arrays = npz(PREPARATION / row["path"])
    require(set(arrays) == set(prep.ARRAY_SHAPES), "all19 original crop fields required")
    require(
        {k: prep.signature(v) for k, v in arrays.items()} == row["array_signatures"],
        "crop dtype/shape/bytes changed",
    )
    return arrays


def infer_crops(rows, read_arrays, forward, save, *, first_inputs, smoke_tokens):
    """All rows/conditions, no token eligibility test before a nonempty forward.

    Forward callback returns original float32[256,384] final LayerNorm tokens,
    elapsed seconds. Empty masks retain NaN tokens without model calls. The
    synthetic tests may supply a short row sequence; full71 is enforced outside.
    """
    tokens, geometry, records = {}, {}, []
    for index, row in enumerate(rows):
        arrays = read_arrays(row)
        frozen = {key: prep.signature(value) for key, value in arrays.items()}
        crop_id = row["crop_id"]
        geometry[crop_id] = {key: arrays[key].copy() for key in GEOMETRY_KEYS}
        for condition in prep.CONDITIONS:
            values = arrays[condition + "_normalized_chw"]
            if index == 0:
                require(
                    byte_exact(values, first_inputs[condition]),
                    "first crop/prepared source differs",
                )
            ready = row["crop_status"] == "ready"
            require(ready or row["crop_status"] == "empty_mask", "unexpected crop status")
            if ready:
                result, elapsed = forward(values)
            else:
                require(np.isnan(values).all(), "empty mask input must stay NaN")
                result, elapsed = np.full((256, 384), np.nan, np.float32), None
            require(
                isinstance(result, np.ndarray)
                and result.shape == (256, 384)
                and result.dtype == np.float32,
                "all256 original float32 LayerNorm tokens required",
            )
            if index == 0 and condition == "raw":
                require(
                    byte_exact(result[None], smoke_tokens),
                    "first full-image-run tokens differ bytewise from fixed CPU smoke",
                )
            require(
                frozen == {key: prep.signature(value) for key, value in arrays.items()},
                "model forward mutated prepared crop input",
            )
            owned = result.copy()
            key = crop_id, condition
            require(key not in tokens, "duplicate crop/condition")
            tokens[key] = owned
            record = dict(
                crop_id=crop_id,
                frame_idx=row["frame_idx"],
                view=row["view"],
                mask_row=row["mask_row"],
                condition=condition,
                crop_status=row["crop_status"],
                token_status=row["token_status"],
                eligible_token_count=int(geometry[crop_id]["token_eligible"].sum()),
                forward_called=ready,
                forward_seconds=elapsed,
                finite_token_count=int(np.isfinite(owned).all(-1).sum()),
                token_signature=prep.signature(owned),
                normalized_input_signature=prep.signature(values),
            )
            save(record, owned, geometry[crop_id])
            records.append(record)
    return tokens, geometry, records


def pack_correspondence(result):
    """Lossless ndarray/scalar split; NaN/inf JSON scalars carry explicit markers."""
    arrays = {}

    def visit(value, path):
        if isinstance(value, np.ndarray):
            name = "__".join(path)
            require(
                name not in arrays and not value.dtype.hasobject,
                "ambiguous/object correspondence array",
            )
            arrays[name] = value.copy()
            return dict(array=name, **prep.signature(value))
        if isinstance(value, dict):
            return {key: visit(item, (*path, key)) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [visit(item, (*path, str(index))) for index, item in enumerate(value)]
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return dict(nonfinite="NaN" if np.isnan(value) else "+inf" if value > 0 else "-inf")
        require(
            value is None or type(value) in (str, bool, int, float),
            "unsupported correspondence metadata",
        )
        return value

    metadata = visit(result, ())
    return arrays, metadata


def correspondence_rows(pairs, tokens, geometry, save):
    """Every registered pair, raw then masked; do not join or rank 3D slots."""
    records = []
    for pair in pairs:
        left, right = pair["earlier_crop_id"], pair["later_crop_id"]
        for condition in prep.CONDITIONS:
            before = (
                prep.signature(tokens[left, condition]),
                prep.signature(tokens[right, condition]),
            )
            result = partial.token_correspondence(
                tokens[left, condition], tokens[right, condition], geometry[left], geometry[right]
            )
            require(
                before
                == (
                    prep.signature(tokens[left, condition]),
                    prep.signature(tokens[right, condition]),
                ),
                "token correspondence mutated model outputs",
            )
            arrays, metadata = pack_correspondence(result)
            record = dict(
                **pair,
                condition=condition,
                status=result["status"],
                comparable=result["comparable"],
                metadata=metadata,
            )
            save(record, arrays)
            records.append(record)
    return records


def run(output, expected_preparation_sha256):
    output = Path(output).resolve()
    require(
        output.is_relative_to(WORK) and not output.exists(),
        "new workspace image diagnostic output required",
    )
    prep.no_model_imports()
    evidence = {}
    code = (
        Path(__file__).resolve(),
        Path(prep.__file__).resolve(),
        Path(partial.__file__).resolve(),
        Path(smoke.__file__).resolve(),
        Path(assets.__file__).resolve(),
        WORK / "DINO_PARTIAL_PROTOCOL.md",
    )
    for path in code:
        prep.bind(path, assets.sha(path), evidence)
    inputs = load_inputs(expected_preparation_sha256, evidence)
    started, environment = assets.now(), assets.runtime()
    output.mkdir(parents=True)
    outputs = {}

    def record_file(path):
        key = path.relative_to(output).as_posix()
        require(key not in outputs, "output was already recorded")
        outputs[key] = dict(path=str(path), sha256=assets.sha(path), size_bytes=path.stat().st_size)

    def write_json(name, value):
        path = output / name
        assets.write_json(path, value)
        record_file(path)

    def write_npz(name, arrays):
        path = output / name
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        record_file(path)
        return dict(path=name, sha256=outputs[name]["sha256"])

    try:
        plan = dict(
            schema=STAGE,
            preparation_sha256=expected_preparation_sha256,
            first_crop_sha256=FIRST_SHA,
            cpu_smoke_sha256=SMOKE_SHA,
            crop_plan_sha256=assets.sha(PREPARATION / "plan.json"),
            crops=inputs["plan"]["crops"],
            pairs=inputs["plan"]["pairs"],
            conditions=list(prep.CONDITIONS),
            maximum_forward_calls=142,
            pair_condition_records=1400,
            all_token_count_per_crop=256,
            device="cpu",
            dtype="float32",
            torch_cpu_threads=2,
            first_smoke_tokens="original final LayerNorm float32 required byteexact",
            model_initialization=(
                "official dinov2_vits14(pretrained=False), default518 position table; "
                "localCPU weights_only=True strict"
            ),
            preprocessing="prepared CHW; no resize or normalization here",
            correspondence_version=partial.VERSION,
            geometry_join_performed=False,
            source_hashes={str(p): evidence[str(p)] for p in code},
        )
        write_json("plan.json", plan)
        (output / "code_snapshot").mkdir()
        for path in code:
            dest = output / "code_snapshot" / path.name
            shutil.copyfile(path, dest)
            require(assets.sha(dest) == evidence[str(path)], "source snapshot changed")
            record_file(dest)
        (output / "tokens").mkdir()
        (output / "pairs").mkdir()
        model = CPUModel(evidence, inputs["smoke"])

        def save_tokens(row, values, geometry):
            row.update(
                write_npz(
                    f"tokens/{row['crop_id']}_{row['condition']}.npz",
                    dict(final_layernorm_patch_tokens=values, **geometry),
                )
            )

        begin = time.perf_counter()
        tokens, geometry, inference = infer_crops(
            inputs["rows"],
            crop_arrays,
            model,
            save_tokens,
            first_inputs=inputs["first"],
            smoke_tokens=inputs["golden"]["final_layernorm_patch_tokens"],
        )
        inference_wall = time.perf_counter() - begin
        require(
            len(inference) == 142 and sum(r["forward_called"] for r in inference) <= 142,
            "all142 fixed token rows required",
        )
        write_json("inference_records.json", inference)
        # All image features are saved before pair arithmetic. Prior geometry or
        # color-distance payloads are never opened in this stage.

        def save_pair(row, arrays):
            row.update(write_npz(f"pairs/p{row['pair_index']:04d}_{row['condition']}.npz", arrays))

        begin = time.perf_counter()
        pairs = correspondence_rows(inputs["plan"]["pairs"], tokens, geometry, save_pair)
        pair_wall = time.perf_counter() - begin
        require(len(pairs) == 1400, "all1400 pair-condition rows required")
        write_json("pair_records.json", pairs)
        loaded = model.verify_loaded(evidence)
        summary = dict(
            schema=STAGE,
            status="ok",
            mask_crops=71,
            background_conditions=list(prep.CONDITIONS),
            token_records=142,
            forward_calls=sum(r["forward_called"] for r in inference),
            maximum_forward_calls=142,
            all_saved_token_rows=142 * 256,
            mask_pairs=700,
            pair_condition_records=1400,
            comparable_pair_condition_records=sum(r["comparable"] for r in pairs),
            pair_status_counts={
                status: sum(r["status"] == status for r in pairs)
                for status in sorted({r["status"] for r in pairs})
            },
            original_first_crop_inputs_byte_exact=True,
            original_smoke_LayerNorm_tokens_byte_exact=True,
            inference_seconds_excluding_load_io=sum(r["forward_seconds"] or 0 for r in inference),
            inference_and_token_save_wall_seconds=inference_wall,
            correspondence_and_save_wall_seconds=pair_wall,
            runtime=environment,
            loaded_source=loaded,
            device="cpu",
            torch_cpu_threads=2,
            GPU=False,
            model_training=False,
            geometry_join_performed=False,
            old_partialD_payload_loaded=False,
            ID_labels_changed=False,
            biological_ID_truth=False,
            fixed_posthoc_cases=True,
            limitations=[
                "Image-only anonymous patch correspondences, not biological identity accuracy.",
                "Mask/core location does not remove Transformer attention "
                "or resize-boundary context.",
                "Every NN minimum remains even at large distance; "
                "no acceptance threshold or forced assignment.",
                "All ties and missing/invalid/tokenless rows remain; "
                "no replacement frames or crops.",
            ],
        )
        write_json("summary.json", summary)
        report = output / "RESULT.md"
        report.write_text(
            "# 固定DINO画像対応診断\n\n71匿名マスク・背景2条件、"
            "全142 token行と全700組x2条件を保存。"
            "全256 tokenと全距離行列・全NN同点を保持しています。\n\n"
            f"実forward {summary['forward_calls']}回、比較可能なpair条件 "
            f"{summary['comparable_pair_condition_records']}/1400。"
            f"モデル推論合計 {summary['inference_seconds_excluding_load_io']:.3f}秒"
            "(load/保存を除く)。CPU2、GPUなし。\n\n"
            "固定最初cropの両入力と、元CPU smokeのLayerNorm全tokenはbyteexactです。"
            "既存部分D149や3D slotとの数値結合は実施していません。"
            "対応の有無・距離は生物ID精度ではなく、IDやreviewを変更しません。\n",
            encoding="utf-8",
        )
        record_file(report)
        require(assets.runtime() == environment, "software runtime changed")
        prep.final_guard(evidence)
        for item in outputs.values():
            require(assets.sha(item["path"]) == item["sha256"], "saved image evidence changed")
        assets.write_json(
            output / "diagnosis.provenance.json",
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
    parser.add_argument("--expected-preparation-sha256", required=True)
    args = parser.parse_args()
    result = run(args.output, args.expected_preparation_sha256)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "forward_calls",
                    "pair_condition_records",
                    "comparable_pair_condition_records",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
