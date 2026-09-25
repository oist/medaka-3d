"""Saved-only image diagnostics; no geometry, model or biological labels.

Report all700 paired background changes and exact eligibility strata without
selecting new cases. Quantiles are descriptive: repeated masks/time pairs are
not independent encounters. Matching a nearest neighbor is not accepting an ID.
"""

from __future__ import annotations

import json
from pathlib import Path

import diagnose_dino_fixed_masks as image_run
import numpy as np

prep, assets = image_run.prep, image_run.assets
ROOT = prep.WORK / "dino_fixed71_image_correspondence_v1"
SEAL_SHA = "6a975f9577b8af141e2da3660a91a4af79017a3a0b9fb88f0ce5ec9a18d87ae3"
OUTPUT = prep.WORK / "dino_fixed71_image_summary_v1"
STAGE = "dino_fixed71_saved_image_summary_v1"


def distribution(values):
    values = np.asarray(values, np.float64)
    finite = values[np.isfinite(values)]
    return dict(
        count=len(values),
        finite_count=len(finite),
        nonfinite_count=len(values) - len(finite),
        **dict(
            zip(
                ("minimum", "q25", "median", "q75", "maximum"),
                np.quantile(finite, [0, 0.25, 0.5, 0.75, 1]).tolist(),
                strict=True,
            )
        )
        if len(finite)
        else dict(minimum=None, q25=None, median=None, q75=None, maximum=None),
        mean=float(finite.mean()) if len(finite) else None,
    )


def scalar(value):
    return np.nan if isinstance(value, dict) and "nonfinite" in value else float(value)


def analyze(inference, rows, mutual):
    expected = {(index, condition) for index in range(700) for condition in prep.CONDITIONS}
    table = {(r["pair_index"], r["condition"]): r for r in rows}
    prep.require(
        len(rows) == 1400 and set(table) == expected, "all1400 unique saved pairs required"
    )
    by_crop = {(r["crop_id"], r["condition"]): r for r in inference}
    prep.require(len(inference) == len(by_crop) == 142, "all142 saved token rows required")
    raw_inference = [r for r in inference if r["condition"] == "raw"]
    pair_summaries = []
    for index in range(700):
        raw, masked = table[index, "raw"], table[index, "masked"]
        for key in (
            "earlier_frame",
            "later_frame",
            "view",
            "earlier_crop_id",
            "later_crop_id",
            "earlier_mask_row",
            "later_mask_row",
        ):
            prep.require(raw[key] == masked[key], "background conditions changed pair membership")
        quantities = {}
        for row in (raw, masked):
            meta = row["metadata"]
            lcount, rcount = (meta[side]["eligible_count"] for side in ("left", "right"))
            prep.require(
                lcount == by_crop[row["earlier_crop_id"], row["condition"]]["eligible_token_count"]
                and rcount
                == by_crop[row["later_crop_id"], row["condition"]]["eligible_token_count"],
                "saved same-crop eligibility changed",
            )
            quantities[row["condition"]] = dict(
                comparable=row["comparable"],
                status=row["status"],
                earlier_eligible=lcount,
                later_eligible=rcount,
                earlier_matched_fraction=scalar(meta["left"]["matched_fraction"]),
                later_matched_fraction=scalar(meta["right"]["matched_fraction"]),
                coverage_mean=float(
                    np.mean([scalar(meta[side]["matched_fraction"]) for side in ("left", "right")])
                ),
                forward_tie_fraction=scalar(meta["left"]["tie_fraction"]),
                reverse_tie_fraction=scalar(meta["right"]["tie_fraction"]),
                mean_mutual_cosine_distance=scalar(meta["mutual_distance_distribution"]["mean"]),
                mutual_edge_count=meta["mutual_distance_distribution"]["count"],
            )
        a, b = mutual(raw), mutual(masked)
        for value, row in ((a, raw), (b, masked)):
            prep.require(
                value.dtype == bool and value.shape == (256, 256), "all256x256 mutual mask required"
            )
            prep.require(
                int(value.sum()) == quantities[row["condition"]]["mutual_edge_count"],
                "mutual edge count differs",
            )
        intersection, union = int((a & b).sum()), int((a | b).sum())
        pair_summaries.append(
            dict(
                pair_index=index,
                request_index=raw["request_index"],
                view=raw["view"],
                earlier_frame=raw["earlier_frame"],
                later_frame=raw["later_frame"],
                earlier_mask_row=raw["earlier_mask_row"],
                later_mask_row=raw["later_mask_row"],
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

    def group(records):
        result = dict(
            pairs=len(records),
            exact_mutual_edge_set_equal=sum(r["exact_mutual_edge_set_equal"] for r in records),
            **{
                key: distribution([r[key] for r in records])
                for key in (
                    "masked_minus_raw_coverage",
                    "masked_minus_raw_mean_mutual_distance",
                    "mutual_edge_jaccard",
                )
            },
        )
        for condition in prep.CONDITIONS:
            result[condition] = {
                key: distribution([r[condition][key] for r in records])
                for key in (
                    "coverage_mean",
                    "mean_mutual_cosine_distance",
                    "forward_tie_fraction",
                    "reverse_tie_fraction",
                    "mutual_edge_count",
                )
            }
        return result

    summary = dict(
        schema=STAGE,
        status="ok",
        fixed_crops=71,
        fixed_mask_pairs=700,
        pair_condition_records=1400,
        all_token_rows=142 * 256,
        finite_token_rows=sum(r["finite_token_count"] for r in inference),
        crop_eligible_token_counts=distribution([r["eligible_token_count"] for r in raw_inference]),
        by_view={
            view: dict(
                crop_eligible_token_counts=distribution(
                    [r["eligible_token_count"] for r in raw_inference if r["view"] == view]
                ),
                **group([r for r in pair_summaries if r["view"] == view]),
            )
            for view in prep.VIEWS
        },
        all_views_descriptive_pool=group(pair_summaries),
        by_exact_minimum_eligible_tokens={
            str(count): group([r for r in pair_summaries if r["minimum_eligible_tokens"] == count])
            for count in sorted({r["minimum_eligible_tokens"] for r in pair_summaries})
        },
        eligibility_groups_are_thresholds=False,
        geometry_join_performed=False,
        ID_truth=False,
        interpretation=[
            "All mask Cartesian products include different individuals, duplicates "
            "and reflections; no labels distinguish these.",
            "Coverage counts unique tokens participating in mutual nearest neighbors, "
            "not correct-match coverage.",
            "Background differences include changed attention context and native-mask-edge "
            "resize mixing; this does not isolate either cause.",
            "Paired mutual-edge Jaccard compares the same patch-cell indices "
            "under both image conditions.",
            "Exact eligibility-count strata are descriptive, "
            "not new gates or accepted correspondences.",
            "Views are reported separately; pooled descriptions do not imply "
            "calibrated cross-view distances.",
            "Repeated crops and time pairs are not independent encounters; "
            "no inferential significance or identity accuracy is claimed.",
        ],
    )
    return pair_summaries, summary


def main():
    prep.require(not OUTPUT.exists(), "new summary output required")
    prep.no_model_imports()
    evidence = {str(Path(__file__).resolve()): assets.sha(__file__)}
    for module in (image_run, prep, assets, image_run.partial, image_run.smoke):
        path = Path(module.__file__).resolve()
        prep.bind(path, assets.sha(path), evidence)
    seal = image_run.verified(
        ROOT, "diagnosis.provenance.json", image_run.STAGE, SEAL_SHA, evidence
    )
    rows = prep.read_json(prep.sealed_output(ROOT, seal, "pair_records.json", evidence))
    inference = prep.read_json(prep.sealed_output(ROOT, seal, "inference_records.json", evidence))

    def mutual(row):
        path = prep.sealed_output(ROOT, seal, row["path"], evidence)
        prep.require(row["sha256"] == evidence[str(path)], "pair file binding differs")
        with np.load(path, allow_pickle=False) as values:
            array = values["mutual_nn"].copy()
        signature = prep.signature(array)
        prep.require(
            all(row["metadata"]["mutual_nn"][key] == value for key, value in signature.items()),
            "mutual mask signature differs",
        )
        return array

    started = assets.now()
    paired, summary = analyze(inference, rows, mutual)
    OUTPUT.mkdir()
    # Arrays keep any NaN exactly; JSON uses explicit nonfinite markers.
    paired_arrays, paired_metadata = image_run.pack_correspondence(
        dict(
            records=paired,
            masked_minus_raw_coverage=np.asarray([r["masked_minus_raw_coverage"] for r in paired]),
            masked_minus_raw_mean_mutual_distance=np.asarray(
                [r["masked_minus_raw_mean_mutual_distance"] for r in paired]
            ),
            mutual_edge_jaccard=np.asarray([r["mutual_edge_jaccard"] for r in paired]),
        )
    )
    with (OUTPUT / "paired_arrays.npz").open("xb") as stream:
        np.savez_compressed(stream, **paired_arrays)
    assets.write_json(OUTPUT / "paired.json", paired_metadata)
    assets.write_json(OUTPUT / "summary.json", summary)
    (OUTPUT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    pooled = summary["all_views_descriptive_pool"]
    lines = [
        "# DINO固定画像の保存結果集計",
        "",
        "全71匿名マスク/700組/背景2条件を保持。3D/既存部分D149とは未結合です。",
        "",
        f"有限token行 {summary['finite_token_rows']}/{summary['all_token_rows']}。"
        f"採用token数/マスクは {summary['crop_eligible_token_counts']}。",
        "",
        "|view|pairs|raw coverage median|masked coverage median|"
        "paired change median|NN edge Jaccard median|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for view, value in summary["by_view"].items():
        lines.append(
            f"|{view}|{value['pairs']}|{value['raw']['coverage_mean']['median']:.6f}|{value['masked']['coverage_mean']['median']:.6f}|{value['masked_minus_raw_coverage']['median']:.6f}|{value['mutual_edge_jaccard']['median']:.6f}|"
        )
    lines += [
        "",
        f"同じmutual NN edge集合は {pooled['exact_mutual_edge_set_equal']}/700。"
        "NNは距離が大きくても存在するため、coverageを正しい個体対応率とは呼びません。",
        "",
        "背景置換差にはattentionの文脈とmask境界のresize混合が共存し、片方だけの因果効果とは断定できません。viewごとの分布と正確な採用token数層を保存し、新しい受理閾値・時刻選別・ID判定は作っていません。",
    ]
    (OUTPUT / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    prep.final_guard(evidence)
    outputs = {
        p.name: dict(path=str(p), sha256=assets.sha(p), size_bytes=p.stat().st_size)
        for p in OUTPUT.iterdir()
        if p.is_file()
    }
    assets.write_json(
        OUTPUT / "summary.provenance.json",
        dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=assets.now(),
            inputs=evidence,
            outputs=outputs,
            geometry_join_performed=False,
            model_calls=0,
        ),
    )
    print(json.dumps(dict(status="ok", outputs=len(outputs), pooled=pooled)))


if __name__ == "__main__":
    main()
