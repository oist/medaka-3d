"""Saved-only minute background diagnostics; no model, images or 3D associations.

The original fixed-six summary arithmetic is retained. All adjacent requests,
mask pairs and unscorable conditions remain, including both-empty NN edge sets.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import Counter
from itertools import pairwise
from pathlib import Path

import diagnose_dino_minute_images as image_run
import numpy as np
import summarize_dino_fixed_images as prior

store, old = image_run.store, image_run.old
prep, assets = old.prep, old.assets
require = store.require
STAGE = "dino_minute_adjacent_image_summary_v1"
ROOT = prep.WORK / "dino_minute_adjacent_image_correspondence_v1"
SEAL_SHA = "89a94ad59f9e3b41d6d95ef8e835808807adf737bdd175f46e730be006426aa4"
PINS = {
    Path(
        image_run.__file__
    ).resolve(): "95a41ff6201cc9c64cd404a9d59fa495264eafeaca204010f616dbd076594ea6",
    Path(
        prior.__file__
    ).resolve(): "1196d96fa360766ed35a11294796823c9cb2a4f5078de7782da01549b7ac5566",
    **image_run.PINS,
}
NUMERIC = (
    "earlier_matched_fraction",
    "later_matched_fraction",
    "coverage_mean",
    "forward_tie_fraction",
    "reverse_tie_fraction",
    "mean_mutual_cosine_distance",
    "mutual_edge_count",
)
CHANGES = (
    "masked_minus_raw_coverage",
    "masked_minus_raw_mean_mutual_distance",
    "mutual_edge_jaccard",
)


def validate_rows(plan, inference, rows):
    """Generic small-fixture boundary; actual 593/1934/177 is separately fixed."""
    require(plan["conditions"] == ["raw", "masked"], "fixed background conditions required")
    require(
        [(r["earlier_frame"], r["later_frame"], r["view"]) for r in plan["requests"]]
        == [(a, b, v) for a, b in pairwise(plan["frames"]) for v in plan["views"]],
        "all adjacent view requests required without bridges",
    )
    expected_inference = [
        dict(**crop, condition=c) for crop in plan["crops"] for c in plan["conditions"]
    ]
    require(
        len(inference) == len(expected_inference)
        and [{k: r[k] for k in e} for r, e in zip(inference, expected_inference, strict=True)]
        == expected_inference,
        "complete ordered crop-condition rows required",
    )
    require(
        len({(r["crop_id"], r["condition"]) for r in inference}) == len(inference), "duplicate crop"
    )
    expected_pairs = [dict(**p, condition=c) for p in plan["pairs"] for c in plan["conditions"]]
    require(
        len(rows) == len(expected_pairs)
        and [{k: r[k] for k in e} for r, e in zip(rows, expected_pairs, strict=True)]
        == expected_pairs,
        "complete ordered pair-condition rows required",
    )
    require(
        [p["pair_index"] for p in plan["pairs"]] == list(range(len(plan["pairs"]))),
        "pair index drift",
    )
    require(len({r["shared_pair_id"] for r in rows}) == len(rows), "duplicate shared pair")
    for i, request in enumerate(plan["requests"]):
        require(request["request_index"] == i, "request index drift")
        a, b = request["earlier_mask_count"], request["later_mask_count"]
        require(all(n is None or (type(n) is int and n >= 0) for n in (a, b)), "invalid mask count")
        status = (
            "view_unavailable"
            if a is None or b is None
            else "no_mask_pairs"
            if not a or not b
            else "available"
        )
        require(
            request["status"] == status and request["mask_pair_count"] == (a or 0) * (b or 0),
            "request missing-state drift",
        )
        members = [p for p in plan["pairs"] if p["request_index"] == i]
        require(len(members) == request["mask_pair_count"], "request Cartesian count differs")
        require(
            [(p["earlier_mask_row"], p["later_mask_row"]) for p in members]
            == [(x, y) for x in range(a or 0) for y in range(b or 0)],
            "all anonymous mask Cartesian products required",
        )
        for p in members:
            require(
                all(p[k] == request[k] for k in ("earlier_frame", "later_frame", "view")),
                "request identity differs",
            )
    require(
        all(0 <= p["request_index"] < len(plan["requests"]) for p in plan["pairs"]),
        "unknown request",
    )


def group(records):
    """Original distribution definitions; explicit empty-set and missing counters."""
    result = dict(
        pairs=len(records),
        exact_mutual_edge_set_equal=sum(r["exact_mutual_edge_set_equal"] for r in records),
        both_empty_mutual_edge_sets=sum(r["mutual_edge_union"] == 0 for r in records),
        nonempty_exact_mutual_edge_set_equal=sum(
            r["exact_mutual_edge_set_equal"] and r["mutual_edge_union"] > 0 for r in records
        ),
        **{k: prior.distribution([r[k] for r in records]) for k in CHANGES},
    )
    for condition in ("raw", "masked"):
        result[condition] = dict(
            comparable=sum(r[condition]["comparable"] for r in records),
            unscorable=sum(not r[condition]["comparable"] for r in records),
            status_counts=dict(sorted(Counter(r[condition]["status"] for r in records).items())),
            **{k: prior.distribution([r[condition][k] for r in records]) for k in NUMERIC},
        )
    return result


def analyze(plan, inference, rows, mutual):
    validate_rows(plan, inference, rows)
    by_crop = {(r["crop_id"], r["condition"]): r for r in inference}
    table = {(r["pair_index"], r["condition"]): r for r in rows}
    paired = []
    for p in plan["pairs"]:
        raw, masked = table[p["pair_index"], "raw"], table[p["pair_index"], "masked"]
        quantities = {}
        for row in (raw, masked):
            meta = row["metadata"]
            lc, rc = (meta[side]["eligible_count"] for side in ("left", "right"))
            require(
                lc == by_crop[row["earlier_crop_id"], row["condition"]]["eligible_token_count"]
                and rc == by_crop[row["later_crop_id"], row["condition"]]["eligible_token_count"],
                "saved same-crop eligibility differs",
            )
            quantities[row["condition"]] = dict(
                comparable=row["comparable"],
                status=row["status"],
                earlier_eligible=lc,
                later_eligible=rc,
                earlier_valid=meta["left"]["valid_count"],
                later_valid=meta["right"]["valid_count"],
                earlier_invalid_eligible=meta["left"]["invalid_eligible_count"],
                later_invalid_eligible=meta["right"]["invalid_eligible_count"],
                earlier_matched_fraction=prior.scalar(meta["left"]["matched_fraction"]),
                later_matched_fraction=prior.scalar(meta["right"]["matched_fraction"]),
                coverage_mean=float(
                    np.mean(
                        [prior.scalar(meta[side]["matched_fraction"]) for side in ("left", "right")]
                    )
                ),
                forward_tie_fraction=prior.scalar(meta["left"]["tie_fraction"]),
                reverse_tie_fraction=prior.scalar(meta["right"]["tie_fraction"]),
                mean_mutual_cosine_distance=prior.scalar(
                    meta["mutual_distance_distribution"]["mean"]
                ),
                mutual_edge_count=meta["mutual_distance_distribution"]["count"],
            )
        require(
            all(
                quantities["raw"][k] == quantities["masked"][k]
                for k in ("earlier_eligible", "later_eligible")
            ),
            "background conditions changed geometry eligibility",
        )
        a, b = mutual(raw), mutual(masked)
        for value, row in ((a, raw), (b, masked)):
            require(
                value.dtype == bool and value.shape == (256, 256),
                "all256x256 mutual bitmap required",
            )
            require(
                int(value.sum()) == quantities[row["condition"]]["mutual_edge_count"],
                "mutual edge count differs",
            )
        intersection, union = int((a & b).sum()), int((a | b).sum())
        paired.append(
            dict(
                **p,
                minimum_eligible_tokens=min(
                    quantities["raw"]["earlier_eligible"], quantities["raw"]["later_eligible"]
                ),
                **quantities,
                masked_minus_raw_coverage=quantities["masked"]["coverage_mean"]
                - quantities["raw"]["coverage_mean"],
                masked_minus_raw_mean_mutual_distance=quantities["masked"][
                    "mean_mutual_cosine_distance"
                ]
                - quantities["raw"]["mean_mutual_cosine_distance"],
                mutual_edge_intersection=intersection,
                mutual_edge_union=union,
                mutual_edge_jaccard=intersection / union if union else np.nan,
                exact_mutual_edge_set_equal=bool(np.array_equal(a, b)),
            )
        )
    raw_inference = [r for r in inference if r["condition"] == "raw"]
    summary = dict(
        schema=STAGE,
        status="ok",
        fixed_crops=len(plan["crops"]),
        fixed_mask_pairs=len(paired),
        pair_condition_records=len(rows),
        requested_pair_views=len(plan["requests"]),
        request_status_counts=dict(sorted(Counter(r["status"] for r in plan["requests"]).items())),
        all_token_rows=len(inference) * 256,
        finite_token_rows=sum(r["finite_token_count"] for r in inference),
        empty_mask_crops=sum(r["crop_status"] == "empty_mask" for r in raw_inference),
        zero_eligible_token_crops=sum(r["eligible_token_count"] == 0 for r in raw_inference),
        crop_eligible_token_counts=prior.distribution(
            [r["eligible_token_count"] for r in raw_inference]
        ),
        by_view={
            v: dict(
                crop_eligible_token_counts=prior.distribution(
                    [r["eligible_token_count"] for r in raw_inference if r["view"] == v]
                ),
                requested_pair_views=sum(r["view"] == v for r in plan["requests"]),
                **group([r for r in paired if r["view"] == v]),
            )
            for v in plan["views"]
        },
        all_views_descriptive_pool=group(paired),
        by_exact_minimum_eligible_tokens={
            str(n): group([r for r in paired if r["minimum_eligible_tokens"] == n])
            for n in sorted({r["minimum_eligible_tokens"] for r in paired})
        },
        zero_eligible_pairs=group([r for r in paired if r["minimum_eligible_tokens"] == 0]),
        nonzero_eligible_pairs=group([r for r in paired if r["minimum_eligible_tokens"] > 0]),
        eligibility_groups_are_thresholds=False,
        geometry_join_performed=False,
        model_calls=0,
        raw_images_opened=0,
        GPU=False,
        ID_truth=False,
        ID_or_review_changed=False,
        temporal_holdout=False,
        interpretation=[
            "All anonymous mask Cartesian products remain; NN coverage is not correct-ID coverage.",
            "Both-empty edge sets compare equal mathematically, but are separately counted "
            "and are not evidence of correspondence.",
            "NaN metric rows remain in denominators; quantiles use finite values "
            "and show their counts.",
            "Mask count None means view unavailable, zero means available without masks; "
            "request rows preserve both.",
            "Background differences include attention context and mask-boundary resize mixing; "
            "neither cause is isolated.",
            "Eligibility strata are exact descriptive counts, not thresholds. "
            "Repeated crops/pairs are not independent encounters.",
            "Expansion followed six-frame results; no temporal holdout "
            "or biological ID truth claim.",
        ],
    )
    return paired, summary


def validate_actual(plan, summary):
    require(
        plan["schema"] == image_run.STAGE
        and plan["frames"] == list(image_run.crops.FRAMES)
        and plan["views"] == list(image_run.crops.VIEWS)
        and plan["maximum_forward_calls"] == 1186
        and plan["pair_condition_records"] == 3868
        and len(plan["crops"]) == 593
        and len(plan["pairs"]) == 1934
        and len(plan["requests"]) == 177,
        "fixed minute image scope required",
    )
    require(
        summary["schema"] == image_run.STAGE
        and summary["status"] == "ok"
        and summary["forward_calls"] == 1186
        and summary["pair_condition_records"] == 3868
        and summary["geometry_join_performed"] is False
        and summary["ID_or_review_changed"] is False
        and summary["GPU"] is False
        and summary["full256_tokens_and_all_NN_ties"] is True,
        "successful image-only producer required",
    )
    b = plan["bindings"]
    require(
        b["preparation_sha256"] == image_run.PREPARATION_SHA
        and b["source_plan_sha256"] == image_run.crops.PLAN_SHA
        and b["codec_source_sha256"] == image_run.PINS[Path(store.__file__).resolve()]
        and b["device"] == "cpu"
        and b["dtype"] == "float32"
        and b["torch_cpu_threads"] == 2,
        "fixed image/codec/model binding differs",
    )


def read_mutual(reader, row):
    key = row["shared_pair_id"]
    ref = reader.index["pairs"][key]
    require(
        ref["left_id"] == row["earlier_crop_id"] + "__" + row["condition"]
        and ref["right_id"] == row["later_crop_id"] + "__" + row["condition"],
        "shared pair identity differs",
    )
    arrays, metadata = reader.read_pair(key)
    require(
        store.canonical(metadata) == store.canonical(row["metadata"]), "shared metadata differs"
    )
    value = arrays["mutual_nn"]
    require(
        all(row["metadata"]["mutual_nn"][k] == v for k, v in store.signature(value).items()),
        "mutual bitmap signature differs",
    )
    return value


def numeric_arrays(paired):
    return {
        **{k: np.asarray([r[k] for r in paired], np.float64) for k in CHANGES},
        **{
            c + "__" + k: np.asarray([r[c][k] for r in paired], np.float64)
            for c in ("raw", "masked")
            for k in NUMERIC
        },
    }


def run(output):
    output = Path(output).resolve()
    require(
        output.is_relative_to(prep.WORK) and not output.exists(), "new workspace output required"
    )
    prep.no_model_imports()
    evidence, outputs, timings = {}, {}, {}
    sources = {Path(__file__).resolve(), *PINS}
    for p in sorted(sources):
        prep.bind(p, PINS.get(p, assets.sha(p)), evidence)
    started, wall = assets.now(), time.perf_counter()
    output.mkdir(parents=True)

    def record(path):
        outputs[path.relative_to(output).as_posix()] = dict(
            path=str(path), sha256=assets.sha(path), size_bytes=path.stat().st_size
        )

    def write(name, data):
        path = output / name
        assets.write_json(path, data)
        record(path)

    try:
        t = time.perf_counter()
        seal = old.verified(ROOT, "diagnosis.provenance.json", image_run.STAGE, SEAL_SHA, evidence)
        for p in sources - {Path(__file__).resolve(), Path(prior.__file__).resolve()}:
            if str(p) in seal["inputs"]:
                old.flat_input(seal, p, evidence)
        old.flat_input(seal, Path(image_run.__file__).resolve(), evidence)
        old.flat_input(seal, Path(store.__file__).resolve(), evidence)

        def saved(name):
            return prep.read_json(prep.sealed_output(ROOT, seal, name, evidence))

        plan, producer, inference, rows = (
            saved(n)
            for n in ("plan.json", "summary.json", "inference_records.json", "pair_records.json")
        )
        validate_actual(plan, producer)
        validate_rows(plan, inference, rows)
        manifest = prep.sealed_output(ROOT, seal, "shared_store/store.manifest.json", evidence)
        require(
            evidence[str(manifest)] == producer["shared_store_manifest_sha256"],
            "store manifest binding differs",
        )
        reader = store.StoreReader(
            manifest.parent,
            expected_bindings=plan["bindings"],
            expected_manifest_sha256=evidence[str(manifest)],
        )
        require(
            set(reader.index["pairs"]) == {r["shared_pair_id"] for r in rows},
            "all store pair membership required",
        )
        require(
            set(reader.index["features"])
            == {r["crop_id"] + "__" + r["condition"] for r in inference},
            "all store feature membership required",
        )
        timings["input_SHA_metadata_seconds"] = time.perf_counter() - t
        for p in sorted(sources):
            dest = output / "code_snapshot" / p.name
            dest.parent.mkdir(exist_ok=True)
            shutil.copyfile(p, dest)
            require(assets.sha(dest) == evidence[str(p)], "source snapshot changed")
            record(dest)
        write(
            "plan.json",
            dict(
                schema=STAGE,
                image_seal_sha256=SEAL_SHA,
                source_plan_sha256=image_run.crops.PLAN_SHA,
                frames=plan["frames"],
                views=plan["views"],
                conditions=plan["conditions"],
                requests=plan["requests"],
                mask_pairs=1934,
                pair_condition_records=3868,
                metric_semantics_source_sha256=PINS[Path(prior.__file__).resolve()],
                geometry_join=False,
                model_calls=0,
                thresholds_changed=False,
            ),
        )
        t = time.perf_counter()
        paired, result = analyze(plan, inference, rows, lambda row: read_mutual(reader, row))
        require(
            sum(r[c]["comparable"] for r in paired for c in ("raw", "masked"))
            == producer["numerically_comparable_pairs"],
            "producer comparable count differs",
        )
        require(
            dict(Counter(r[c]["status"] for r in paired for c in ("raw", "masked")))
            == producer["pair_status_counts"],
            "producer status counts differ",
        )
        timings["shared_restore_and_summary_seconds"] = time.perf_counter() - t
        t = time.perf_counter()
        reader.verify_all()
        timings["store_SHA_seconds"] = time.perf_counter() - t
        arrays = numeric_arrays(paired)
        with (output / "paired_arrays.npz").open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        record(output / "paired_arrays.npz")
        _, metadata = old.pack_correspondence(dict(records=paired))
        write("paired.json", metadata)
        lines = [
            "# DINO固定1分の画像結果集計",
            "",
            "固定60時刻・全1934匿名マスク組・背景2条件。3Dや生物IDとは未結合です。",
            "",
            "|view|全組|計算可能 raw/masked|coverage中央値 raw/masked|"
            "差の中央値|NN集合Jaccard中央値|",
            "|---|---:|---:|---:|---:|---:|",
        ]

        def display(v):
            return "未定義" if v is None else f"{v:.6f}"

        for view, value in result["by_view"].items():
            lines.append(
                f"|{view}|{value['pairs']}|{value['raw']['comparable']}/{value['masked']['comparable']}|{display(value['raw']['coverage_mean']['median'])}/{display(value['masked']['coverage_mean']['median'])}|{display(value['masked_minus_raw_coverage']['median'])}|{display(value['mutual_edge_jaccard']['median'])}|"
            )
        lines += [
            "",
            "欠測・採用token0の行を保持し、量の分布は有限数/不能数を併記。両条件のNN集合が空で一致する場合を別計数しています。coverageは正しい個体対応率ではありません。",
            "",
            "背景差にはattentionの文脈とmask境界のresize混合が共存します。繰返しmask/時刻は独立遭遇ではなく、この拡張は固定6時刻結果を見た後に選ばれました。閾値やIDは変更していません。",
        ]
        (output / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        record(output / "RESULT.md")
        t = time.perf_counter()
        prep.final_guard(evidence)
        for r in outputs.values():
            require(assets.sha(r["path"]) == r["sha256"], "output changed")
        prep.no_model_imports()
        timings["final_SHA_seconds"] = time.perf_counter() - t
        timings["total_through_guards_seconds"] = time.perf_counter() - wall
        result["timings"] = timings
        result["numeric_array_signatures"] = store.signatures(arrays)
        write("summary.json", result)
        assets.write_json(
            output / "summary.provenance.json",
            dict(
                stage=STAGE,
                status="ok",
                started_at=started,
                finished_at=assets.now(),
                inputs=evidence,
                outputs=outputs,
                model_calls=0,
                geometry_join=False,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    result = run(parser.parse_args().output)
    print(
        json.dumps(
            {k: result[k] for k in ("status", "fixed_mask_pairs", "pair_condition_records")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
