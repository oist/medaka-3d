"""Join sealed fixed60 geometry and all adjacent DINO mask correspondences.

No model, images, candidate arrays or new geometry are evaluated. The fixed6
median-mutual/row-rank/permutation equations are retained. Numeric slot matches
remain conditional on each saved geometry; they are not biological ID labels.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import Counter
from itertools import combinations, pairwise
from pathlib import Path

import diagnose_dino_minute_images as images
import dino_shared_store as store
import join_dino_geometry_fixed6 as fixed
import numpy as np
import prepare_dino_minute_crops as crop

WORK, REPO = crop.WORK, crop.REPO
FRAMES, VIEWS, CONDITIONS = crop.FRAMES, tuple(crop.VIEWS), tuple(crop.CONDITIONS)
METHODS = ("original", "tail")
STAGE = "dino_minute_adjacent_geometry_join_v1"
GEOMETRY = WORK / "dino_minute_adjacent_geometry_preparation_v2"
GEOMETRY_SHA = "32f601b81241b9fc3e8eb93d7820fd90efcbc080473a7c381d06c3a96926d15c"
IMAGE = WORK / "dino_minute_adjacent_image_correspondence_v1"
IMAGE_SHA = "89a94ad59f9e3b41d6d95ef8e835808807adf737bdd175f46e730be006426aa4"
MANIFEST_SHA = "048f34a3bed2844b4754d73b8288fde6c7b249fd03a7ed20d58463843ddb4190"
COMPATIBILITY_SHA = "5d2704442b195d11e0d1abe180f7367d9fb1084b6b42f396d5bf1ec5f64581f4"
PINS = {
    crop.PLAN: crop.PLAN_SHA,
    Path(crop.__file__).resolve(): images.PINS[Path(crop.__file__).resolve()],
    Path(store.__file__).resolve(): images.PINS[Path(store.__file__).resolve()],
    Path(
        fixed.__file__
    ).resolve(): "e4e39aea0e7236ceba5e490b64200cedbfa30992e550ab00c447e2b4dad1c6d0",
    Path(
        images.__file__
    ).resolve(): "95a41ff6201cc9c64cd404a9d59fa495264eafeaca204010f616dbd076594ea6",
    WORK / "prepare_dino_minute_geometry_v2.py": (
        "390d76e6803a86f6a03772466cfe69d2529aa8b0f8966713cd298142d3db5ca0"
    ),
}
assets, require = crop.assets, store.require


def endpoint(reference, crop_rows):
    """Validate saved geometry/core decisions without reprojecting or reading color."""
    masks = reference["all_masks"]
    require(len(reference["slots"]) == 3, "all3 endpoint slots required")
    count = reference["mask_count"]
    require(
        reference["mask_available"] is (count is not None)
        and (count is None or (type(count) is int and count >= 0))
        and len(masks) == (count or 0) == len(crop_rows),
        "missing/zero/allN geometry and crop counts differ",
    )
    for index, (mask, saved) in enumerate(zip(masks, crop_rows, strict=True)):
        require(mask["mask_row"] == saved["mask_row"] == index, "allN mask row order")
        for key in ("crop_id", "mask_pixels", "core_pixels", "mask_sha256", "core_sha256"):
            require(mask[key] == saved[key], "geometry/crop core binding differs: " + key)
        require(
            mask["crop_path"] == saved["path"]
            and mask["crop_sha256"] == saved["sha256"]
            and mask["core_valid"] is (mask["core_pixels"] >= 12),
            "fixed native core12/crop binding differs",
        )
    assigned, eligible, reasons = [], [], []
    for slot, row in enumerate(reference["slots"]):
        require(row["slot"] == slot, "saved slot order differs")
        index, why = row["assigned_mask_row"], []
        require(type(index) is int and -1 <= index < len(masks), "assigned row outside allN")
        if row["assignment_reason"] != "usable":
            why.append(row["assignment_reason"])
        body = row["body_coherence"]
        if body["requested_points"] != 33 or body["valid_points"] != 33:
            why.append("incomplete_33_projection")
        if len(body["best_mask_rows"]) != 1:
            why.append("coherence_no_unique_minimum")
        elif index >= 0 and index != body["best_mask_rows"][0]:
            why.append("assignment_coherence_disagree")
        if not np.isfinite(fixed.number(body["C_single"])):
            why.append("coherence_nonfinite")
        direct = all(row["observed_mouth_chest"]) and all(
            x >= 0 for x in row["view_sources_mouth_chest"]
        )
        require(row["direct_mouth_chest"] is direct, "saved direct mouth/chest differs")
        require(
            row["geometry_eligible"] is (not why) and row["reasons"] == why,
            "sealed geometry reason/gate differs",
        )
        if not why:
            require(direct and index >= 0, "eligible geometry lacks direct source/mask")
        valid = index >= 0 and masks[index]["core_valid"]
        if index >= 0 and not valid:
            why.append("assigned_core_invalid")
        good = bool(row["geometry_eligible"] and valid)
        require(
            row["core_valid"] is valid
            and row["core_pixels"] == (masks[index]["core_pixels"] if index >= 0 else None)
            and row["crop_id"] == (masks[index]["crop_id"] if index >= 0 else None)
            and row["endpoint_eligible"] is good
            and row["endpoint_reasons"] == why
            and type(row["identity_review"]) is bool,
            "sealed endpoint/core/review differs",
        )
        assigned.append(index)
        eligible.append(good)
        reasons.append(why)
    distinct = all(i >= 0 for i in assigned) and len(set(assigned)) == 3
    ready = bool(distinct and all(eligible))
    require(
        reference["assigned_mask_rows"] == assigned
        and reference["slot_geometry_eligible"]
        == [r["geometry_eligible"] for r in reference["slots"]]
        and reference["slot_endpoint_eligible"] == eligible
        and reference["all3_distinct"] is distinct
        and reference["all3_geometry_eligible"]
        is bool(distinct and all(r["geometry_eligible"] for r in reference["slots"]))
        and reference["all3_endpoint_eligible"] is ready
        and reference["review_excluded_from_eligibility"] is False
        and reference["token_validity_evaluated"] is False,
        "saved endpoint aggregate/eligibility policy differs",
    )
    return dict(
        assigned=assigned,
        eligible=eligible,
        reasons=reasons,
        ready=ready,
        review=[r["identity_review"] for r in reference["slots"]],
    )


def validate_geometry(rows, crop_rows, counts):
    expected = [(m, n, v) for n in FRAMES for v in VIEWS for m in METHODS]
    require(
        [(r["method"], r["frame_idx"], r["view"]) for r in rows] == expected,
        "all360 ordered geometry groups required",
    )
    grouped = {(n, v): [] for n in FRAMES for v in VIEWS}
    for row in crop_rows:
        grouped[row["frame_idx"], row["view"]].append(row)
    result = {}
    for row in rows:
        key = row["method"], row["frame_idx"], row["view"]
        require(row["mask_count"] == counts[key[1:]], "geometry/crop availability differs")
        result[key] = endpoint(row, grouped[key[1:]])
    return result


def requests():
    return [
        dict(request_index=index, method=m, earlier_frame=a, later_frame=b, view=v)
        for index, (a, b, v, m) in enumerate(
            (a, b, v, m) for a, b in pairwise(FRAMES) for v in VIEWS for m in METHODS
        )
    ]


def image_metrics(plan, rows, read_pair):
    """Read every image pair; retain the fixed6 exact median, including one edge."""
    require(len(rows) == 2 * len(plan["pairs"]), "all pair conditions required")
    table, records = {}, []
    for index, row in enumerate(rows):
        pair, condition = plan["pairs"][index // 2], CONDITIONS[index % 2]
        require(
            all(row[k] == v for k, v in pair.items()) and row["condition"] == condition,
            "ordered registered anonymous pair differs",
        )
        key_id = f"p{pair['pair_index']:04d}_{condition}"
        require(row["shared_pair_id"] == key_id, "shared pair ID/order differs")
        arrays, metadata = read_pair(key_id)
        require(metadata == row["metadata"], "shared scalar/signature metadata differs")
        distance, mutual = arrays["distances"], arrays["mutual_nn"]
        require(
            distance.shape == mutual.shape == (256, 256)
            and distance.dtype == np.float64
            and mutual.dtype == bool,
            "full256 distance/mutual arrays required",
        )
        selected = distance[mutual]
        require(np.isfinite(selected).all(), "mutual edge is not finite")
        median = float(np.quantile(selected, 0.5)) if len(selected) else np.nan
        saved = metadata["mutual_distance_distribution"]
        stored = fixed.number(saved["median"])
        require(
            (median == stored or (np.isnan(median) and np.isnan(stored)))
            and saved["count"] == len(selected),
            "saved median/count changed",
        )
        require(
            row["comparable"] is bool(len(selected))
            and metadata["comparable"] is row["comparable"]
            and metadata["status"] == row["status"],
            "saved image availability differs",
        )
        metric = dict(
            pair_index=pair["pair_index"],
            shared_pair_id=key_id,
            condition=condition,
            status=row["status"],
            comparable=row["comparable"],
            median_mutual_cosine_distance=median,
            mutual_edge_count=len(selected),
        )
        for side, prefix in (("left", "earlier"), ("right", "later")):
            metric[prefix + "_eligible"] = metadata[side]["eligible_count"]
            metric[prefix + "_valid"] = metadata[side]["valid_count"]
            metric[prefix + "_coverage"] = fixed.number(metadata[side]["matched_fraction"])
        metric["forward_tie_fraction"] = fixed.number(metadata["left"]["tie_fraction"])
        metric["reverse_tie_fraction"] = fixed.number(metadata["right"]["tie_fraction"])
        key = (
            *(
                pair[k]
                for k in (
                    "earlier_frame",
                    "later_frame",
                    "view",
                    "earlier_mask_row",
                    "later_mask_row",
                )
            ),
            condition,
        )
        require(key not in table, "duplicate anonymous image pair")
        table[key] = metric
        records.append({**pair, **metric})
    return table, records


def conditional_matrix(request, references, lookup):
    method, early, late, view = (
        request[k] for k in ("method", "earlier_frame", "later_frame", "view")
    )
    left, right = references[method, early, view], references[method, late, view]
    eligible = np.asarray(left["eligible"])[:, None] & np.asarray(right["eligible"])[None, :]
    distance, indices = np.full((2, 3, 3), np.nan), np.full((3, 3), -1, np.int64)
    entries = []
    for i in range(3):
        for j in range(3):
            for ci, condition in enumerate(CONDITIONS):
                metric = None
                if eligible[i, j]:
                    key = early, late, view, left["assigned"][i], right["assigned"][j], condition
                    require(key in lookup, "eligible pair absent from registered allN table")
                    metric = lookup[key]
                    require(indices[i, j] in (-1, metric["pair_index"]), "background pair differs")
                    indices[i, j] = metric["pair_index"]
                    distance[ci, i, j] = metric["median_mutual_cosine_distance"]
                entries.append(
                    dict(
                        earlier_slot=i,
                        later_slot=j,
                        condition=condition,
                        earlier_mask_row=left["assigned"][i],
                        later_mask_row=right["assigned"][j],
                        earlier_review=left["review"][i],
                        later_review=right["review"][j],
                        geometry_available=bool(eligible[i, j]),
                        endpoint_reasons=dict(
                            earlier=left["reasons"][i], later=right["reasons"][j]
                        ),
                        image_status=metric["status"]
                        if metric
                        else "geometry_ineligible_not_joined",
                        mask_pair_index=int(indices[i, j]),
                        metric=copy.deepcopy(metric),
                    )
                )
    return dict(
        distance=distance,
        mask_pair_index=indices,
        entries=entries,
        geometry_available=eligible,
        all3_gate=bool(left["ready"] and right["ready"]),
    )


def join_requests(planned, references, lookup):
    matrices, indices, costs, rows, cases, paired, eligible = [], [], [], [], [], [], []
    for request in planned:
        joined = conditional_matrix(request, references, lookup)
        matrices.append(joined["distance"])
        indices.append(joined["mask_pair_index"])
        eligible.append(joined["geometry_available"])
        each_cost, each_rows, aggregate = [], [], []
        for ci, condition in enumerate(CONDITIONS):
            matrix = joined["distance"][ci]
            value = fixed.six_costs(matrix, joined["all3_gate"])
            each_cost.append(value)
            ready = bool(np.isfinite(value).all())
            aggregate.append(
                dict(
                    condition=condition,
                    scorable=ready,
                    best_permutation_indices=np.flatnonzero(value == value.min()).tolist()
                    if ready
                    else [],
                    reason="all3_geometry_and_all9_image_distances"
                    if ready
                    else "all3_geometry_failed_or_image_distance_missing",
                )
            )
            current = []
            for slot in range(3):
                row = dict(
                    **request,
                    condition=condition,
                    source_slot=slot,
                    distances=matrix[slot].tolist(),
                    **fixed.row_ranking(matrix[slot], same_slot=slot),
                )
                row["targets"] = [
                    e
                    for e in joined["entries"]
                    if e["earlier_slot"] == slot and e["condition"] == condition
                ]
                rows.append(row)
                current.append(row)
            each_rows.append(current)
        costs.append(each_cost)
        cases.append(
            dict(
                **request,
                all3_geometry_gate=joined["all3_gate"],
                entries=joined["entries"],
                aggregate=aggregate,
            )
        )
        for slot in range(3):
            raw, masked = each_rows[0][slot], each_rows[1][slot]
            both = raw["same_and_cross_comparable"] and masked["same_and_cross_comparable"]
            paired.append(
                dict(
                    **request,
                    source_slot=slot,
                    raw=raw,
                    masked=masked,
                    both_same_and_cross_comparable=both,
                    masked_minus_raw_margin=masked["same_slot_margin"] - raw["same_slot_margin"]
                    if both
                    else np.nan,
                    best_target_set_equal=raw["best_targets"] == masked["best_targets"]
                    if both
                    else None,
                )
            )
    arrays = dict(
        distance=np.asarray(matrices),
        mask_pair_index=np.asarray(indices),
        permutation_cost=np.asarray(costs),
        permutations=fixed.PERMUTATIONS.copy(),
        geometry_available=np.asarray(eligible),
        earlier_frame=np.asarray([r["earlier_frame"] for r in planned], np.int64),
        later_frame=np.asarray([r["later_frame"] for r in planned], np.int64),
        method=np.asarray([r["method"] for r in planned]),
        view=np.asarray([r["view"] for r in planned]),
    )
    arrays["finite_distance"] = np.isfinite(arrays["distance"])
    for field in ("same_slot_margin", "same_slot_rank_min", "same_slot_rank_max"):
        arrays[field] = np.asarray([r[field] for r in rows], np.float64).reshape(len(planned), 2, 3)
    require(
        not (arrays["finite_distance"] & ~arrays["geometry_available"][:, None]).any(),
        "image score expanded frozen geometry eligibility",
    )
    return arrays, rows, cases, paired


def cross_view_ranks(planned, arrays):
    """Compare exact ranks only; never add distances from different cameras."""
    lookup = {
        (r["method"], r["earlier_frame"], r["later_frame"], r["view"]): i
        for i, r in enumerate(planned)
    }
    records = []
    for method in METHODS:
        for early, late in pairwise(FRAMES):
            for ci, condition in enumerate(CONDITIONS):
                for first, second in combinations(VIEWS, 2):
                    a, b = (
                        arrays["permutation_cost"][lookup[method, early, late, v], ci]
                        for v in (first, second)
                    )
                    ready = bool(np.isfinite(a).all() and np.isfinite(b).all())
                    abest = np.flatnonzero(a == a.min()).tolist() if np.isfinite(a).all() else []
                    bbest = np.flatnonzero(b == b.min()).tolist() if np.isfinite(b).all() else []
                    records.append(
                        dict(
                            method=method,
                            earlier_frame=early,
                            later_frame=late,
                            condition=condition,
                            first_view=first,
                            second_view=second,
                            comparison_available=ready,
                            first_best_permutation_indices=abest,
                            second_best_permutation_indices=bbest,
                            first_minima_ranks_in_second=[
                                dict(
                                    permutation_index=p,
                                    rank_min=1 + int((b < b[p]).sum()),
                                    rank_max=int((b <= b[p]).sum()),
                                )
                                for p in abest
                            ]
                            if ready
                            else [],
                            best_sets_equal=abest == bbest if ready else None,
                            conditional_on_saved_geometry=True,
                        )
                    )
    return records


def group_stats(rows):
    comparable = [r for r in rows if r["same_and_cross_comparable"]]
    return dict(
        rows=len(rows),
        no_finite_target=sum(r["finite_target_count"] == 0 for r in rows),
        same_only=sum(r["reason"] == "same_only_no_cross_comparison" for r in rows),
        cross_only=sum(r["reason"] == "same_missing_cross_only" for r in rows),
        same_and_cross=len(comparable),
        conditional_same_slot_minimum=sum(
            r["conditional_same_slot_is_minimum"] is True for r in comparable
        ),
        positive_margin=sum(r["same_slot_margin"] > 0 for r in comparable),
        exact_zero_margin=sum(r["same_slot_margin"] == 0 for r in comparable),
        negative_margin=sum(r["same_slot_margin"] < 0 for r in comparable),
        singleton_rank_not_counted_as_success=True,
    )


def method_comparison(rows):
    """Pair equal numeric slots as a diagnostic index, not biological alignment."""
    keyed = {}
    for row in rows:
        key = tuple(
            row[k] for k in ("earlier_frame", "later_frame", "view", "condition", "source_slot")
        )
        require(row["method"] not in keyed.setdefault(key, {}), "duplicate method source row")
        keyed[key][row["method"]] = row
    records = []
    for key, values in keyed.items():
        require(set(values) == set(METHODS), "two method source rows required")
        a, b = (values[m] for m in METHODS)
        own_both = a["same_and_cross_comparable"] and b["same_and_cross_comparable"]
        common = sorted(set(a["finite_targets"]) & set(b["finite_targets"]))
        ranked = []
        for row in (a, b):
            original = np.asarray(row["distances"], np.float64)
            require(
                original.shape == (3,)
                and np.flatnonzero(np.isfinite(original)).tolist() == row["finite_targets"],
                "method row values/finite targets differ",
            )
            values = np.full(3, np.nan)
            values[common] = original[common]
            ranked.append(
                dict(
                    distances=values.tolist(),
                    **fixed.row_ranking(values, same_slot=row["source_slot"]),
                )
            )
        ca, cb = ranked
        common_both = ca["same_and_cross_comparable"] and cb["same_and_cross_comparable"]
        identical, pair_rows = [], []
        for target in common:
            ta, tb = a["targets"][target], b["targets"][target]
            require(
                ta["later_slot"] == tb["later_slot"] == target, "method target entry order differs"
            )
            same_pair = ta["mask_pair_index"] == tb["mask_pair_index"]
            if same_pair:
                require(
                    ta["mask_pair_index"] >= 0
                    and ta["earlier_mask_row"] == tb["earlier_mask_row"]
                    and ta["later_mask_row"] == tb["later_mask_row"]
                    and ta["metric"]["shared_pair_id"] == tb["metric"]["shared_pair_id"]
                    and np.asarray(a["distances"][target], np.float64).tobytes()
                    == np.asarray(b["distances"][target], np.float64).tobytes(),
                    "identical anonymous pair differs in refs or distance bytes",
                )
                identical.append(target)
            pair_rows.append(
                dict(
                    target_slot=target,
                    identical_anonymous_pair=same_pair,
                    original={
                        k: ta[k] for k in ("earlier_mask_row", "later_mask_row", "mask_pair_index")
                    },
                    tail={
                        k: tb[k] for k in ("earlier_mask_row", "later_mask_row", "mask_pair_index")
                    },
                )
            )
        records.append(
            dict(
                zip(
                    ("earlier_frame", "later_frame", "view", "condition", "source_slot"),
                    key,
                    strict=True,
                )
            )
            | dict(
                original_same_and_cross=a["same_and_cross_comparable"],
                tail_same_and_cross=b["same_and_cross_comparable"],
                own_target_sets_both_same_and_cross=own_both,
                common_targets_both_same_and_cross=common_both,
                common_finite_numeric_target_slots=common,
                common_target_anonymous_pairs=pair_rows,
                identical_anonymous_pair_target_slots=identical,
                different_anonymous_pair_target_slots=[s for s in common if s not in identical],
                identical_pair_subset_same_and_cross=a["source_slot"] in identical
                and len(identical) > 1,
                original_only_targets=sorted(set(a["finite_targets"]) - set(b["finite_targets"])),
                tail_only_targets=sorted(set(b["finite_targets"]) - set(a["finite_targets"])),
                original_own_targets_margin=a["same_slot_margin"],
                tail_own_targets_margin=b["same_slot_margin"],
                own_target_sets_margin_delta=b["same_slot_margin"] - a["same_slot_margin"]
                if own_both
                else np.nan,
                original_common_targets=ca,
                tail_common_targets=cb,
                common_targets_margin_delta=cb["same_slot_margin"] - ca["same_slot_margin"]
                if common_both
                else np.nan,
                same_numeric_slot_not_biological_alignment=True,
            )
        )
    return records


def input_member(record, path, digest):
    """Check an inherited input claim without reopening unused raw/model/candidate bytes."""
    # These frozen producers save canonical absolute keys. Demand that exact key;
    # do not resolve every input again for each of the 974 prepared outputs.
    require(
        record["inputs"].get(str(Path(path).resolve())) == digest,
        "named input SHA membership differs",
    )


def sealed(root, name, digest, stage, evidence):
    root = Path(root).resolve()
    crop.bind(root / name, digest, evidence)
    record = crop.read_json(root / name)
    require(
        record["stage"] == stage and record["status"] == "ok" and record.get("finished_at"),
        "successful sealed stage required",
    )
    paths = set()
    for key, row in record["outputs"].items():
        path = (root / key).resolve()
        require(
            path.is_relative_to(root)
            and path == Path(row["path"]).resolve()
            and path not in paths
            and path.stat().st_size == row["size_bytes"],
            "output role/path/size differs",
        )
        paths.add(path)
    require(
        {p.resolve() for p in root.rglob("*") if p.is_file()} == paths | {root / name},
        "sealed exact output membership differs",
    )
    return record


def json_output(root, record, name, evidence):
    path = (root / name).resolve()
    row = record["outputs"][name]
    require(Path(row["path"]).resolve() == path, "named JSON output differs")
    crop.bind(path, row["sha256"], evidence)
    return crop.read_json(path)


def workspace_code():
    paths = {Path(__file__).resolve(), *[p for p in PINS if p.suffix == ".py"]}
    for module in tuple(sys.modules.values()):
        value = getattr(module, "__file__", None)
        if value:
            path = Path(value).resolve()
            if path.suffix == ".py" and any(
                path.is_relative_to(REPO / d) for d in ("src", "labs", "work/three_body")
            ):
                paths.add(path)
    return {str(p): assets.sha(p) for p in sorted(paths)}


def verify_store_references(index, plan):
    """A numeric pair must refer to its exact registered crop/condition endpoints."""
    crops = {r["crop_id"] for r in plan["crops"]}
    features = {name + "__" + c for name in crops for c in CONDITIONS}
    pairs = {f"p{row['pair_index']:04d}_{c}" for row in plan["pairs"] for c in CONDITIONS}
    require(
        set(index["crops"]) == crops
        and set(index["features"]) == features
        and set(index["pairs"]) == pairs,
        "shared all-crop/condition/pair membership differs",
    )
    for name in crops:
        for condition in CONDITIONS:
            require(
                index["features"][name + "__" + condition]["crop_id"] == name,
                "shared feature points at different crop geometry",
            )
    for row in plan["pairs"]:
        for condition in CONDITIONS:
            ref = index["pairs"][f"p{row['pair_index']:04d}_{condition}"]
            require(
                ref["left_id"] == row["earlier_crop_id"] + "__" + condition
                and ref["right_id"] == row["later_crop_id"] + "__" + condition,
                "shared pair endpoint crop/condition reference differs",
            )


def metadata_contract(geometry, image, preparation, plan, image_plan, geom_summary, image_summary):
    """Small-JSON contract; image consumes sealed crops, not SL/calibration directly."""
    for record in (geometry, image):
        input_member(
            record, images.PREPARATION / "preparation.provenance.json", images.PREPARATION_SHA
        )
        input_member(record, crop.PLAN, crop.PLAN_SHA)
    for row in preparation["outputs"].values():
        input_member(image, row["path"], row["sha256"])
    for key in (
        "frames",
        "views",
        "crops",
        "requests",
        "pairs",
        "conditions",
        "maximum_forward_calls",
    ):
        require(image_plan[key] == plan[key], "image/crop plan differs: " + key)
    require(
        geom_summary["status"] == "ok"
        and geom_summary["frames"] == list(FRAMES)
        and geom_summary["views"] == list(VIEWS)
        and geom_summary["methods"] == list(METHODS)
        and geom_summary["horizon_stop"] == 76360
        and geom_summary["geometry_precedes_feature_join"] is True
        and geom_summary["crop_preparation_sha256"] == images.PREPARATION_SHA
        and geom_summary["comparison_sha256"] == crop.COMPARISON_SHA,
        "fixed successful same76360 geometry required",
    )
    for method, (path, digest) in crop.CANDIDATES.items():
        require(
            geom_summary["candidate_sha256"][method] == digest
            and plan["candidate_references"][method]["sha256"] == digest
            and Path(plan["candidate_references"][method]["path"]).resolve() == path.resolve(),
            "candidate origin differs",
        )
        input_member(geometry, path, digest)
        input_member(preparation, path, digest)
    shared = plan["shared_SL_calibration_sha256"]
    require(
        len(shared) == 7 and shared == geom_summary["shared_SL_calibration_sha256"],
        "same SL3/calibration4 required",
    )
    for path, digest in shared.items():
        for record in (geometry, preparation):
            input_member(record, path, digest)
    bindings = dict(
        preparation_sha256=images.PREPARATION_SHA,
        compatibility_sha256=COMPATIBILITY_SHA,
        source_plan_sha256=crop.PLAN_SHA,
        codec_source_sha256=PINS[Path(store.__file__).resolve()],
        original_model_runner_sha256=images.PINS[Path(images.old.__file__).resolve()],
        device="cpu",
        dtype="float32",
        torch_cpu_threads=2,
    )
    require(
        image_plan["bindings"] == bindings
        and image_summary["status"] == "ok"
        and image_summary["geometry_join_performed"] is False
        and image_summary["mask_pairs"] == 1934
        and image_summary["pair_condition_records"] == 3868
        and image_summary["shared_store_manifest_sha256"] == MANIFEST_SHA,
        "completed all1934/3868 independent image stage required",
    )
    return bindings


def load_inputs(evidence):
    for path, digest in PINS.items():
        crop.bind(path, digest, evidence)
    geometry = sealed(
        GEOMETRY,
        "geometry.provenance.json",
        GEOMETRY_SHA,
        "dino_minute_adjacent_geometry_preparation_v2",
        evidence,
    )
    image = sealed(IMAGE, "diagnosis.provenance.json", IMAGE_SHA, images.STAGE, evidence)
    prep_root = images.PREPARATION
    preparation = sealed(
        prep_root, "preparation.provenance.json", images.PREPARATION_SHA, crop.STAGE, evidence
    )
    for record in (geometry, image):
        input_member(record, prep_root / "preparation.provenance.json", images.PREPARATION_SHA)
        input_member(record, crop.PLAN, crop.PLAN_SHA)
    # Historical code used by either producer must still agree; no source swap
    # is permitted merely because a new join imports the same module name.
    for record in (geometry, image, preparation):
        for path, digest in record["inputs"].items():
            if Path(path).suffix == ".py" and Path(path).is_relative_to(REPO):
                crop.bind(path, digest, evidence)
    for path in (Path(images.__file__).resolve(), Path(store.__file__).resolve()):
        input_member(image, path, PINS[path])
    input_member(
        geometry,
        WORK / "prepare_dino_minute_geometry_v2.py",
        PINS[WORK / "prepare_dino_minute_geometry_v2.py"],
    )
    for record in (geometry, image):
        for row in record["outputs"].values():
            crop.bind(row["path"], row["sha256"], evidence)
    plan = json_output(prep_root, preparation, "plan.json", evidence)
    crops = json_output(prep_root, preparation, "crops.json", evidence)
    natives = json_output(prep_root, preparation, "native_evidence.json", evidence)
    prep_summary = json_output(prep_root, preparation, "summary.json", evidence)
    images.validate_preparation(plan, crops, natives, prep_summary)
    geom_summary = json_output(GEOMETRY, geometry, "summary.json", evidence)
    image_summary = json_output(IMAGE, image, "summary.json", evidence)
    image_plan = json_output(IMAGE, image, "plan.json", evidence)
    bindings = metadata_contract(
        geometry, image, preparation, plan, image_plan, geom_summary, image_summary
    )
    reader = store.StoreReader(
        IMAGE / "shared_store", expected_bindings=bindings, expected_manifest_sha256=MANIFEST_SHA
    )
    verify_store_references(reader.index, plan)
    for name, ref in reader.index["outputs"].items():
        row = image["outputs"]["shared_store/" + name]
        require(
            ref["sha256"] == row["sha256"] and ref["size_bytes"] == row["size_bytes"],
            "shared store/output role binding differs",
        )
    return dict(
        geometry=geometry,
        image=image,
        preparation=preparation,
        plan=plan,
        crops=crops,
        natives=natives,
        geom_summary=geom_summary,
        image_summary=image_summary,
        reader=reader,
    )


def run(output):
    output = Path(output).resolve()
    require(output.is_relative_to(WORK) and not output.exists(), "new join output required")
    crop.prior.no_model_imports()
    started, wall, evidence, outputs = assets.now(), time.perf_counter(), {}, {}
    output.mkdir(parents=True)

    def record(path):
        outputs[path.relative_to(output).as_posix()] = dict(
            path=str(path), size_bytes=path.stat().st_size, sha256=assets.sha(path)
        )

    def write(name, value):
        path = output / name
        assets.write_json(path, images.old.pack_correspondence(dict(value=value))[1]["value"])
        record(path)

    try:
        code = workspace_code()
        for path, digest in code.items():
            crop.bind(path, digest, evidence)
        write(
            "registration.json",
            dict(
                schema=STAGE,
                source_plan_sha256=crop.PLAN_SHA,
                geometry_sha256=GEOMETRY_SHA,
                image_sha256=IMAGE_SHA,
                metric="median_mutual_edge_cosine_distance",
                frames=list(FRAMES),
                views=list(VIEWS),
                methods=list(METHODS),
                conditions=list(CONDITIONS),
                requests=requests(),
                source_sha256=code,
            ),
        )
        for path, digest in {**code, str(crop.PLAN): crop.PLAN_SHA}.items():
            snapshot = output / "code_snapshot" / Path(path).relative_to(REPO)
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(Path(path).read_bytes())
            require(assets.sha(snapshot) == digest, "source snapshot differs")
            record(snapshot)
        bound = load_inputs(evidence)
        image_rows = json_output(IMAGE, bound["image"], "pair_records.json", evidence)
        lookup, metrics = image_metrics(bound["plan"], image_rows, bound["reader"].read_pair)
        require(
            dict(Counter(r["status"] for r in metrics))
            == bound["image_summary"]["pair_status_counts"],
            "all-image status accounting differs",
        )
        write("all_image_metrics.json", metrics)
        raw = json_output(GEOMETRY, bound["geometry"], "geometry_references.json", evidence)
        counts = {(r["frame_idx"], r["view"]): r["mask_count"] for r in bound["natives"]}
        references = validate_geometry(raw, bound["crops"], counts)
        planned = requests()
        arrays, rows, cases, paired = join_requests(planned, references, lookup)
        compare = method_comparison(rows)
        cross = cross_view_ranks(planned, arrays)
        require(
            len(rows) == 2124 and len(cases) == 354 and len(metrics) == 3868,
            "fixed complete join cardinality differs",
        )
        with (output / "joined_arrays.npz").open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        record(output / "joined_arrays.npz")
        for name, value in (
            ("source_rows.json", rows),
            ("requests.json", cases),
            ("background_comparison.json", paired),
            ("method_comparison.json", compare),
            ("cross_view_ranks.json", cross),
        ):
            write(name, value)
        summary = dict(
            schema=STAGE,
            status="ok",
            native_frames=60,
            adjacent_pairs=59,
            requests=354,
            source_rows=len(rows),
            paired_background_rows=len(paired),
            image_pair_conditions=len(metrics),
            methods=list(METHODS),
            views=list(VIEWS),
            conditions=list(CONDITIONS),
            horizon_stop=76360,
            metric="median_mutual_edge_cosine_distance",
            distance_slots=6372,
            permutation_slots=4248,
            finite_distances=int(arrays["finite_distance"].sum()),
            geometry_available_entries=int(arrays["geometry_available"].sum()),
            finite_permutation_costs=int(np.isfinite(arrays["permutation_cost"]).sum()),
            by_method_view_condition=[
                dict(
                    method=m,
                    view=v,
                    condition=c,
                    **group_stats(
                        [r for r in rows if (r["method"], r["view"], r["condition"]) == (m, v, c)]
                    ),
                )
                for m in METHODS
                for v in VIEWS
                for c in CONDITIONS
            ],
            by_method_view_condition_target_scope="each_method_own_finite_targets",
            by_method_view_condition_common_numeric_targets=[
                dict(
                    method=m,
                    view=v,
                    condition=c,
                    **group_stats(
                        [
                            r[m + "_common_targets"]
                            for r in compare
                            if (r["view"], r["condition"]) == (v, c)
                        ]
                    ),
                )
                for m in METHODS
                for v in VIEWS
                for c in CONDITIONS
            ],
            method_common_same_and_cross_rows=sum(
                r["common_targets_both_same_and_cross"] for r in compare
            ),
            common_numeric_elements_identical_anonymous_pair=sum(
                len(r["identical_anonymous_pair_target_slots"]) for r in compare
            ),
            common_numeric_elements_different_anonymous_pair=sum(
                len(r["different_anonymous_pair_target_slots"]) for r in compare
            ),
            identical_anonymous_pair_subset_same_and_cross_rows=sum(
                r["identical_pair_subset_same_and_cross"] for r in compare
            ),
            method_original_only_rows=sum(
                r["original_same_and_cross"] and not r["tail_same_and_cross"] for r in compare
            ),
            method_tail_only_rows=sum(
                r["tail_same_and_cross"] and not r["original_same_and_cross"] for r in compare
            ),
            cross_view_available=sum(r["comparison_available"] for r in cross),
            geometry_sha256=GEOMETRY_SHA,
            image_sha256=IMAGE_SHA,
            candidate_sha256=bound["geom_summary"]["candidate_sha256"],
            source_plan_sha256=crop.PLAN_SHA,
            raw_images_opened=0,
            model_forward_calls=0,
            new_geometry_evaluations=0,
            candidate_arrays_read=False,
            GPU=False,
            independent_ID_truth=False,
            ID_or_review_changed=False,
            temporal_holdout=False,
            limitations=[
                "All ranks depend on saved geometry; numeric slots are not independent "
                "biological IDs.",
                "Finite-target ranks retain exact ties; a singleton rank1 is not a "
                "same/cross comparison.",
                "One mutual edge suffices under the frozen metric; edge counts and coverage "
                "remain visible.",
                "All6 permutation costs require all3 endpoints distinct/eligible and all9 "
                "distances finite.",
                "Different cameras are compared by ranks only; their raw distances are not pooled.",
                "Method comparisons pair numeric slot indices, without biological alignment "
                "or optimized relabeling.",
                "Raw and masked conditions share tokens/time; these rows are not "
                "independent encounters.",
                "Sealed geometry/image outputs are rehashed; upstream raw videos, model "
                "assets and candidates are not reopened.",
            ],
        )
        require(workspace_code() == code, "late workspace import/source drift")
        crop.prior.no_model_imports()
        crop.prior.final_guard(evidence)
        for row in outputs.values():
            require(assets.sha(row["path"]) == row["sha256"], "join output changed before seal")
        summary["elapsed_seconds"] = time.perf_counter() - wall
        write("summary.json", summary)
        assets.write_json(
            output / "join.provenance.json",
            dict(
                stage=STAGE,
                status="ok",
                started_at=started,
                finished_at=assets.now(),
                inputs=evidence,
                outputs=outputs,
                environment=assets.runtime(),
            ),
        )
        return summary
    except BaseException as error:
        assets.write_json(
            output / "join.provenance.json",
            dict(
                stage=STAGE,
                status="failed",
                started_at=started,
                finished_at=assets.now(),
                inputs=evidence,
                outputs=outputs,
                error=dict(type=type(error).__name__, message=str(error)),
            ),
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
