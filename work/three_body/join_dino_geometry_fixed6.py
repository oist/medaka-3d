"""Fixed conditional join, never identity truth or a new tracking decision.

Only after reviewed execution: median mutual-edge cosine is connected to the
old149 finite entry region. All90 requests x2 conditions, 1080 permutation
slots and540 source rows remain. Rank means rank among finite targets only.
No candidate/GT/model/image decode, new threshold, relabeling or review change.
"""

from __future__ import annotations

import argparse
import copy
import json
from itertools import combinations, permutations
from pathlib import Path

import diagnose_dino_fixed_masks as images
import numpy as np

prep, assets = images.prep, images.assets
WORK = prep.WORK
STAGE = "dino_geometry_fixed6_join_v1"
IMAGE_ROOT = WORK / "dino_fixed71_image_correspondence_v1"
IMAGE_SHA = "6a975f9577b8af141e2da3660a91a4af79017a3a0b9fb88f0ce5ec9a18d87ae3"
PLAN = WORK / "DINO_GEOMETRY_JOIN_PLAN.md"
PLAN_SHA = "6d883a3334c2a335ffc76e7eb5b9f01294268ef3fb245ea8034a54906773e702"
PERMUTATIONS = np.asarray(tuple(permutations(range(3))), np.int64)
METHODS = ("directional", "bounded")
CONDITIONS = prep.CONDITIONS
IMAGE_SOURCE_SHA = "6d0d455e5c1c25d7dfba1f99666be7ec9e7442c41be7841c0084cabd3090e79a"
require = prep.require


def number(value):
    return np.nan if isinstance(value, dict) and value.get("nonfinite") == "NaN" else float(value)


def row_ranking(distance_row, *, same_slot):
    """Ranks are within finite targets, including the uninformative 1/1 case."""
    row = np.asarray(distance_row, np.float64)
    require(
        row.shape == (3,) and type(same_slot) is int and 0 <= same_slot < 3,
        "three targets and native slot required",
    )
    require(not np.isinf(row).any(), "distances are finite or NaN")
    finite = np.flatnonzero(np.isfinite(row))
    best = finite[row[finite] == row[finite].min()].tolist() if len(finite) else []
    same_present = bool(np.isfinite(row[same_slot]))
    cross = finite[finite != same_slot]
    comparable = same_present and len(cross) > 0
    return dict(
        rank_scope="finite_targets_only_not_identity_accuracy",
        finite_targets=finite.tolist(),
        finite_target_count=len(finite),
        best_targets=best,
        same_slot=same_slot,
        same_slot_present=same_present,
        cross_target_count=len(cross),
        same_slot_rank_min=1 + int((row[finite] < row[same_slot]).sum())
        if same_present
        else np.nan,
        same_slot_rank_max=int((row[finite] <= row[same_slot]).sum()) if same_present else np.nan,
        same_slot_margin=float(row[cross].min() - row[same_slot]) if comparable else np.nan,
        same_and_cross_comparable=bool(comparable),
        conditional_same_slot_is_minimum=same_slot in best if comparable else None,
        reason="same_and_cross_available"
        if comparable
        else "same_only_no_cross_comparison"
        if same_present
        else "same_missing_cross_only"
        if len(cross)
        else "no_finite_targets",
    )


def six_costs(matrix, original_all3_gate):
    values = np.asarray(matrix, np.float64)
    require(
        values.shape == (3, 3) and type(original_all3_gate) is bool,
        "3x3 and original bool gate required",
    )
    result = np.full(6, np.nan)
    if original_all3_gate and np.isfinite(values).all():
        result = np.asarray([values[np.arange(3), p].mean() for p in PERMUTATIONS])
    return result


def endpoint(reference, descriptor_context, mask_rows):
    """Verify the already sealed source/coherence/core gate; do not project again."""
    require(
        len(reference["slots"]) == 3 and len(descriptor_context["slot_reasons"]) == 3,
        "all three old slots required",
    )
    assigned, eligible, reasons = [], [], []
    for slot, row in enumerate(reference["slots"]):
        require(row["slot"] == slot, "old slot order changed")
        index, why = row["assigned_mask_row"], []
        require(
            type(index) is int and -1 <= index < len(mask_rows),
            "original mask row outside allN table",
        )
        if row["assignment_reason"] != "usable":
            why.append(row["assignment_reason"])
        body = row["body_coherence"]
        if body["requested_points"] != 33 or body["valid_points"] != 33:
            why.append("incomplete_33_projection")
        if len(body["best_mask_rows"]) != 1:
            why.append("coherence_no_unique_minimum")
        elif index >= 0 and index != body["best_mask_rows"][0]:
            why.append("assignment_coherence_disagree")
        if not np.isfinite(number(body["C_single"])):
            why.append("coherence_nonfinite")
        direct = all(row["observed_mouth_chest"]) and all(
            x >= 0 for x in row["view_sources_mouth_chest"]
        )
        require(row["direct_mouth_chest"] is direct, "saved direct source flag differs")
        require(
            row["geometry_eligible"] == (not why) and row["reasons"] == why,
            "original geometry reasons/gate differ",
        )
        if not why:
            require(direct and index >= 0, "usable geometry lacks direct mouth/chest or mask")
        core_valid = index >= 0 and mask_rows[index]["descriptor_valid"] is True
        if index >= 0 and not core_valid:
            why.append("assigned_core_invalid")
        eligible.append(row["geometry_eligible"] and core_valid)
        assigned.append(index)
        reasons.append(why)
    distinct = all(i >= 0 for i in assigned) and len(set(assigned)) == 3
    ready = bool(distinct and all(eligible))
    require(
        reference["assigned_mask_rows"] == assigned
        and reference["all3_distinct"] == distinct
        and reference["all3_geometry_eligible"]
        == bool(distinct and all(r["geometry_eligible"] for r in reference["slots"]))
        and reference["slot_geometry_eligible"]
        == [r["geometry_eligible"] for r in reference["slots"]],
        "original reference aggregate mismatch",
    )
    require(
        descriptor_context["slot_reasons"] == reasons
        and descriptor_context["all3_distinct"] == distinct
        and descriptor_context["scorable"] == ready,
        "original descriptor-context gate/reasons differ",
    )
    finite_descriptors = np.isfinite(
        np.asarray(descriptor_context["slot_descriptors"], np.float64)
    ).all(-1)
    require(
        np.array_equal(finite_descriptors, eligible), "original finite descriptor region differs"
    )
    return dict(assigned=assigned, eligible=eligible, reasons=reasons, ready=ready)


def conditional_matrix(request, references, image_pairs):
    """Use only original finite entries, with a unique anonymous pair binding."""
    method, early, late, view = (
        request[k] for k in ("method", "earlier_frame", "later_frame", "view")
    )
    left, right = references[method, early, view], references[method, late, view]
    old = np.asarray(request["distances"], np.float64)
    require(old.shape == (3, 3) and not np.isinf(old).any(), "old finite/NaN3x3 required")
    eligible = np.asarray(left["eligible"])[:, None] & np.asarray(right["eligible"])[None, :]
    require(
        np.array_equal(np.isfinite(old), eligible), "do not expand or contract old partial region"
    )
    require(
        request["earlier_ready"] == left["ready"]
        and request["later_ready"] == right["ready"]
        and request["scorable"] == (left["ready"] and right["ready"]),
        "original all3 gate differs",
    )
    require(
        request["earlier_slot_reasons"] == left["reasons"]
        and request["later_slot_reasons"] == right["reasons"],
        "original endpoint reasons differ",
    )
    distance, indices = np.full((2, 3, 3), np.nan), np.full((3, 3), -1, np.int64)
    entries = []
    for i in range(3):
        for j in range(3):
            old_reason = (
                []
                if eligible[i, j]
                else dict(earlier=left["reasons"][i], later=right["reasons"][j])
            )
            require(
                request["distance_entry_reasons"][i][j] == old_reason, "old missing reason changed"
            )
            for ci, condition in enumerate(CONDITIONS):
                entry = dict(
                    earlier_slot=i,
                    later_slot=j,
                    condition=condition,
                    original_available=bool(eligible[i, j]),
                    original_reasons=copy.deepcopy(old_reason),
                    mask_pair_index=-1,
                    metric=None,
                )
                if eligible[i, j]:
                    key = early, late, view, left["assigned"][i], right["assigned"][j], condition
                    require(key in image_pairs, "unique registered anonymous mask pair absent")
                    metric = image_pairs[key]
                    require(
                        indices[i, j] in (-1, metric["pair_index"]),
                        "conditions use different anonymous pair",
                    )
                    indices[i, j] = metric["pair_index"]
                    distance[ci, i, j] = metric["median_mutual_cosine_distance"]
                    entry.update(mask_pair_index=metric["pair_index"], metric=copy.deepcopy(metric))
                entries.append(entry)
    return dict(
        distance=distance,
        mask_pair_index=indices,
        entries=entries,
        original_all3_gate=bool(request["scorable"]),
        original_finite=eligible.copy(),
    )


def image_metrics(plan, rows, read_arrays):
    """Recompute the single registered metric before geometry labels are opened."""
    require(len(plan["pairs"]) == 700 and len(rows) == 1400, "fixed all700x2 images required")
    table, records = {}, []
    for row in rows:
        pair = plan["pairs"][row["pair_index"]]
        require(
            all(row[k] == v for k, v in pair.items()) and row["condition"] in CONDITIONS,
            "image row differs from fixed pair plan",
        )
        arrays = read_arrays(row)
        distance, mutual = arrays["distances"], arrays["mutual_nn"]
        require(
            distance.shape == mutual.shape == (256, 256)
            and distance.dtype == np.float64
            and mutual.dtype == bool,
            "all256x256 image distance/mutual arrays required",
        )
        selected = distance[mutual]
        require(np.isfinite(selected).all(), "mutual edge has nonfinite distance")
        median = float(np.quantile(selected, 0.5)) if len(selected) else np.nan
        saved = row["metadata"]["mutual_distance_distribution"]
        stored = number(saved["median"])
        require(
            (median == stored or (np.isnan(median) and np.isnan(stored)))
            and saved["count"] == len(selected),
            "registered median/edge count differs from full saved image arrays",
        )
        require(row["comparable"] == bool(len(selected)), "image comparability status inconsistent")
        key = tuple(
            row[k]
            for k in (
                "earlier_frame",
                "later_frame",
                "view",
                "earlier_mask_row",
                "later_mask_row",
                "condition",
            )
        )
        require(key not in table, "duplicate anonymous image pair")
        metric = dict(
            pair_index=row["pair_index"],
            condition=row["condition"],
            status=row["status"],
            comparable=row["comparable"],
            median_mutual_cosine_distance=median,
            mutual_edge_count=len(selected),
            earlier_eligible=row["metadata"]["left"]["eligible_count"],
            later_eligible=row["metadata"]["right"]["eligible_count"],
            earlier_valid=row["metadata"]["left"]["valid_count"],
            later_valid=row["metadata"]["right"]["valid_count"],
            earlier_coverage=number(row["metadata"]["left"]["matched_fraction"]),
            later_coverage=number(row["metadata"]["right"]["matched_fraction"]),
            forward_tie_fraction=number(row["metadata"]["left"]["tie_fraction"]),
            reverse_tie_fraction=number(row["metadata"]["right"]["tie_fraction"]),
        )
        table[key] = metric
        records.append({**pair, **metric})
    require(len(table) == 1400, "all image conditions retained")
    return table, records


def validate_geometry(raw_rows, contexts, mask_rows, requests, saved, crop_rows):
    """All original36 endpoint groups,90 requests and149 entries; no new geometry."""
    keys = {(m, n, v) for m in METHODS for n in prep.FRAMES for v in prep.VIEWS}
    refs = {(r["method"], r["frame_idx"], r["view"]): r for r in raw_rows}
    ctx = {(r["method"], r["frame_idx"], r["view"]): r for r in contexts}
    require(
        len(raw_rows) == len(contexts) == 36 and set(refs) == set(ctx) == keys,
        "all36 old endpoint groups required",
    )
    masks = prep.fixed_rows(mask_rows)
    crop_map = {(r["frame_idx"], r["view"], r["mask_row"]): r for r in crop_rows}
    for (native, view), row in masks.items():
        require(row["all_mask_count"] == len(row["masks"]), "allN old mask count differs")
        for i, mask in enumerate(row["masks"]):
            require(
                mask["mask_row"] == i and (native, view, i) in crop_map,
                "old mask order/presence differs",
            )
            crop = crop_map[native, view, i]
            require(
                all(mask[k] == crop[k] for k in ("mask_sha256", "core_sha256", "core_pixels")),
                "old geometry mask and learned image crop have different origins",
            )
            finite = bool(np.isfinite(np.asarray(mask["descriptor"], np.float64)).all())
            require(
                mask["descriptor_valid"] is finite,
                "old core validity disagrees with saved finite descriptor",
            )
    references = {
        key: endpoint(refs[key], ctx[key], masks[key[1], key[2]]["masks"]) for key in sorted(keys)
    }
    expected = [
        (m, a, b, v) for m in METHODS for a, b in combinations(prep.FRAMES, 2) for v in prep.VIEWS
    ]
    require(
        len(requests) == 90
        and [
            tuple(r[k] for k in ("method", "earlier_frame", "later_frame", "view"))
            for r in requests
        ]
        == expected,
        "old all90 order differs",
    )
    require(
        saved["distance"].shape == (90, 3, 3)
        and saved["distance"].dtype == np.float64
        and saved["permutation_costs"].shape == (90, 6)
        and np.array_equal(saved["permutations"], PERMUTATIONS),
        "old full3x3/6cost schema differs",
    )
    for index, row in enumerate(requests):
        require(
            np.array_equal(
                saved["distance"][index], np.asarray(row["distances"], np.float64), equal_nan=True
            )
            and np.array_equal(
                saved["permutation_costs"][index],
                np.asarray(row["permutation_costs"], np.float64),
                equal_nan=True,
            ),
            "old JSON/NPZ scores differ",
        )
        require(
            bool(saved["scorable"][index]) == row["scorable"]
            and (
                str(saved["methods"][index]),
                int(saved["earlier_frame"][index]),
                int(saved["later_frame"][index]),
                str(saved["views"][index]),
            )
            == expected[index],
            "old array/native alignment differs",
        )
    require(int(np.isfinite(saved["distance"]).sum()) == 149, "old149 finite region changed")
    return references


def join_requests(requests, references, lookup):
    matrices, indices, costs, row_records, entry_records, paired = [], [], [], [], [], []
    for request_index, request in enumerate(requests):
        joined = conditional_matrix(request, references, lookup)
        matrices.append(joined["distance"])
        indices.append(joined["mask_pair_index"])
        header = {k: request[k] for k in ("method", "earlier_frame", "later_frame", "view")}
        each_cost, each_rows = [], []
        for ci, condition in enumerate(CONDITIONS):
            matrix = joined["distance"][ci]
            each_cost.append(six_costs(matrix, joined["original_all3_gate"]))
            ranks = []
            for slot in range(3):
                row = dict(
                    **header,
                    request_index=request_index,
                    condition=condition,
                    source_slot=slot,
                    **row_ranking(matrix[slot], same_slot=slot),
                )
                row["targets"] = [
                    e
                    for e in joined["entries"]
                    if e["earlier_slot"] == slot and e["condition"] == condition
                ]
                row_records.append(row)
                ranks.append(row)
            each_rows.append(ranks)
        costs.append(each_cost)
        aggregate = []
        for condition, value in zip(CONDITIONS, each_cost, strict=True):
            finite = np.isfinite(value)
            aggregate.append(
                dict(
                    condition=condition,
                    scorable=bool(finite.all()),
                    best_permutation_indices=np.flatnonzero(value == value.min()).tolist()
                    if finite.all()
                    else [],
                    reason="original_all3_and_all_new_distances_finite"
                    if finite.all()
                    else "original_all3_failed_or_new_image_distance_missing",
                )
            )
        entry_records.append(
            dict(
                **header,
                request_index=request_index,
                original_request_reason=request["request_reason"],
                original_all3_gate=joined["original_all3_gate"],
                entries=joined["entries"],
                aggregate=aggregate,
            )
        )
        for slot in range(3):
            raw, masked = each_rows[0][slot], each_rows[1][slot]
            both = raw["same_and_cross_comparable"] and masked["same_and_cross_comparable"]
            deltas = []
            for early_metric, late_metric in zip(raw["targets"], masked["targets"], strict=True):
                a, b = early_metric["metric"], late_metric["metric"]
                row_delta = dict(target_slot=early_metric["later_slot"])
                for field in (
                    "median_mutual_cosine_distance",
                    "earlier_coverage",
                    "later_coverage",
                    "mutual_edge_count",
                ):
                    row_delta["masked_minus_raw_" + field] = (
                        b[field] - a[field] if a is not None and b is not None else np.nan
                    )
                deltas.append(row_delta)
            paired.append(
                dict(
                    **header,
                    request_index=request_index,
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
                    target_background_deltas=deltas,
                )
            )
    arrays = dict(
        distance=np.asarray(matrices),
        mask_pair_index=np.asarray(indices),
        permutation_cost=np.asarray(costs),
        permutations=PERMUTATIONS.copy(),
    )
    arrays["finite_distance"] = np.isfinite(arrays["distance"])
    for field in ("same_slot_margin", "same_slot_rank_min", "same_slot_rank_max"):
        arrays[field] = np.asarray([r[field] for r in row_records], np.float64).reshape(
            len(requests), 2, 3
        )
    return arrays, row_records, entry_records, paired


def summary(arrays, rows, paired):
    def group(selected):
        comparable = [r for r in selected if r["same_and_cross_comparable"]]
        return dict(
            rows=len(selected),
            no_finite_target=sum(r["finite_target_count"] == 0 for r in selected),
            same_only=sum(r["reason"] == "same_only_no_cross_comparison" for r in selected),
            cross_only=sum(r["reason"] == "same_missing_cross_only" for r in selected),
            same_and_cross=len(comparable),
            conditional_same_slot_minimum=sum(
                r["conditional_same_slot_is_minimum"] is True for r in comparable
            ),
            conditional_strict_positive_margin=sum(r["same_slot_margin"] > 0 for r in comparable),
            conditional_exact_zero_margin=sum(r["same_slot_margin"] == 0 for r in comparable),
            conditional_negative_margin=sum(r["same_slot_margin"] < 0 for r in comparable),
            singleton_rank_not_counted_as_success=True,
        )

    return dict(
        schema=STAGE,
        status="ok",
        requests=90,
        conditions=list(CONDITIONS),
        distance_slots=1620,
        finite_distances=int(arrays["finite_distance"].sum()),
        original_finite_entries=149,
        permutation_slots=1080,
        finite_permutation_costs=int(np.isfinite(arrays["permutation_cost"]).sum()),
        source_rows=540,
        row_rank_scope="finite_targets_only_not_identity_accuracy",
        metric="median_mutual_edge_cosine_distance",
        by_method_view_condition=[
            dict(
                method=m,
                view=v,
                condition=c,
                **group([r for r in rows if (r["method"], r["view"], r["condition"]) == (m, v, c)]),
            )
            for m in METHODS
            for v in prep.VIEWS
            for c in CONDITIONS
        ],
        by_condition={c: group([r for r in rows if r["condition"] == c]) for c in CONDITIONS},
        paired_source_rows=len(paired),
        comparable_in_both_conditions=sum(r["both_same_and_cross_comparable"] for r in paired),
        equal_best_target_set_among_comparable=sum(
            r["best_target_set_equal"] is True for r in paired
        ),
        same_and_cross_unique_native_view_source_rows=len(
            {
                (r["earlier_frame"], r["later_frame"], r["view"], r["source_slot"])
                for r in rows
                if r["condition"] == "raw" and r["same_and_cross_comparable"]
            }
        ),
        biological_ID_truth=False,
        ID_or_review_changed=False,
        original_missing_region_expanded=False,
        new_inference_or_geometry=False,
        limitations=[
            "Same/cross slots are old3D hypotheses, not independent biological labels.",
            "Rank is among finite targets only; rank1 with one target "
            "is not a successful comparison.",
            "Missing rows and all6 aggregate gates remain; no unmatched case is rescued.",
            "Mutual cosine median can be based on one edge; edge/eligible/valid counts "
            "and coverage are retained without thresholds.",
            "Repeated conditions and duplicate method/time/view/source rows "
            "are not independent encounters.",
            "Background changes in patch sets do not alone establish "
            "biological identity instability.",
        ],
    )


def run(output):
    output = Path(output).resolve()
    require(output.is_relative_to(WORK) and not output.exists(), "new fixed join output required")
    prep.no_model_imports()
    evidence = {}
    code = (
        Path(__file__).resolve(),
        Path(images.__file__).resolve(),
        Path(prep.__file__).resolve(),
        Path(images.partial.__file__).resolve(),
        Path(images.smoke.__file__).resolve(),
        Path(assets.__file__).resolve(),
    )
    for path in code:
        prep.bind(
            path,
            IMAGE_SOURCE_SHA if path == Path(images.__file__).resolve() else assets.sha(path),
            evidence,
        )
    prep.bind(PLAN, PLAN_SHA, evidence)
    image_seal = images.verified(
        IMAGE_ROOT, "diagnosis.provenance.json", images.STAGE, IMAGE_SHA, evidence
    )
    crop_seal = images.verified(
        images.PREPARATION,
        "preparation.provenance.json",
        prep.STAGE,
        images.PREPARATION_SHA,
        evidence,
    )
    old_seal = prep.verify_seal(
        prep.IDENTITY,
        "diagnosis.provenance.json",
        "identity_components_fixed6_v1",
        prep.IDENTITY_SHA,
        evidence,
    )
    for path in code[1:]:
        images.flat_input(image_seal, path, evidence)
    images.flat_input(image_seal, images.PREPARATION / "preparation.provenance.json", evidence)
    images.flat_input(crop_seal, prep.IDENTITY / "diagnosis.provenance.json", evidence)
    images.flat_input(crop_seal, prep.RAW / "review.provenance.json", evidence)
    prep.member(old_seal, prep.RAW / "review.provenance.json", evidence)
    environment = assets.runtime()
    require(image_seal["environment"] == environment, "same image-result software required")
    crop_summary = prep.read_json(
        prep.sealed_output(images.PREPARATION, crop_seal, "summary.json", evidence)
    )
    old_summary = prep.read_json(
        prep.sealed_output(prep.IDENTITY, old_seal, "summary.json", evidence)
    )
    shared = crop_summary["shared_SL_calibration_sha256"]
    require(
        len(shared) == 7 and shared == old_summary["shared_SL_calibration_sha256"],
        "sameSL3/calibration4 required",
    )
    for path, digest in shared.items():
        prep.bind(path, digest, evidence)
        prep.member(old_seal, path, evidence)
        images.flat_input(crop_seal, path, evidence)
        images.flat_input(image_seal, path, evidence)
    started = assets.now()
    output.mkdir()
    outputs = {}

    def record(path):
        outputs[path.name] = dict(
            path=str(path), size_bytes=path.stat().st_size, sha256=assets.sha(path)
        )

    def write(name, value):
        path = output / name
        assets.write_json(path, value)
        record(path)

    def dump(name, arrays):
        path = output / name
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        record(path)

    try:
        write(
            "registration.json",
            dict(
                schema=STAGE,
                started_at=started,
                plan_sha256=PLAN_SHA,
                image_seal_sha256=IMAGE_SHA,
                crop_seal_sha256=images.PREPARATION_SHA,
                old_identity_seal_sha256=prep.IDENTITY_SHA,
                source_sha256={str(p): evidence[str(p)] for p in code},
                metric="median_mutual_edge_cosine_distance",
                geometry_numeric_read_at_registration=False,
            ),
        )
        snapshot = output / "registered_plan.md"
        snapshot.write_bytes(PLAN.read_bytes())
        require(assets.sha(snapshot) == PLAN_SHA, "registered plan snapshot differs")
        record(snapshot)
        for path in code:
            snapshot = output / path.name
            snapshot.write_bytes(path.read_bytes())
            require(assets.sha(snapshot) == evidence[str(path)], "source snapshot drift")
            record(snapshot)
        crop_plan = prep.read_json(
            prep.sealed_output(images.PREPARATION, crop_seal, "plan.json", evidence)
        )
        crop_rows = prep.read_json(
            prep.sealed_output(images.PREPARATION, crop_seal, "crops.json", evidence)
        )
        images.validate_crop_plan(crop_plan, crop_rows)
        img_rows = prep.read_json(
            prep.sealed_output(IMAGE_ROOT, image_seal, "pair_records.json", evidence)
        )

        def read_arrays(row):
            path = prep.sealed_output(IMAGE_ROOT, image_seal, row["path"], evidence)
            require(evidence[str(path)] == row["sha256"], "image row NPZ binding mismatch")
            with np.load(path, allow_pickle=False) as z:
                result = {key: z[key].copy() for key in ("distances", "mutual_nn")}
            for key, value in result.items():
                require(
                    all(row["metadata"][key][k] == v for k, v in prep.signature(value).items()),
                    "saved full image array signature changed",
                )
            return result

        lookup, metrics = image_metrics(crop_plan, img_rows, read_arrays)
        _, encoded = images.pack_correspondence(dict(records=metrics))
        write("image_metrics_before_geometry.json", encoded)

        # All image metrics and the fixed definition are saved before opening
        # any numerical geometry/association/old partial-distance payload.
        def old_json(name):
            return prep.read_json(prep.sealed_output(prep.IDENTITY, old_seal, name, evidence))

        requests = old_json("pair_requests.json")
        saved = images.npz(prep.sealed_output(prep.IDENTITY, old_seal, "pair_costs.npz", evidence))
        references = validate_geometry(
            old_json("geometry_references_before_color.json"),
            old_json("descriptor_contexts.json"),
            old_json("all_mask_descriptors.json"),
            requests,
            saved,
            crop_rows,
        )
        arrays, rows, entries, paired = join_requests(requests, references, lookup)
        require(
            arrays["distance"].shape == (90, 2, 3, 3)
            and arrays["permutation_cost"].shape == (90, 2, 6)
            and len(rows) == 540
            and len(paired) == 270,
            "fixed full output denominators required",
        )
        dump("conditional_arrays.npz", arrays)
        for name, data in (
            ("source_rows.json", rows),
            ("entry_records.json", entries),
            ("paired_background_rows.json", paired),
            ("original_requests_preserved.json", requests),
        ):
            _, encoded = images.pack_correspondence(dict(records=data))
            write(name, encoded)
        result = summary(arrays, rows, paired)
        write("summary.json", result)
        report = output / "RESULT.md"
        report.write_text(
            "# DINOと旧幾何の条件付き結合\n\n全90要求・背景2条件・540行を保持。"
            "順位は有限target内であり、同slot/cross-slotは旧3D仮説の区分です。"
            "生物ID真値ではありません。\n\n"
            f"新距離有限 {result['finite_distances']}/1620、全6順列cost有限 "
            f"{result['finite_permutation_costs']}/1080。両条件でsame+crossを比較できる元行 "
            f"{result['comparable_in_both_conditions']}/270。\n\n"
            "median mutual cosine距離のみを順位に使用し、"
            "個数/coverageによる新gateは加えていません。"
            "rank1/候補1は比較成功へ数えず、未観測行や同点を保持しています。"
            "全targetのedge/eligible/valid個数、coverage、背景差はJSONへ保存しました。"
            "ID/reviewは変更していません。\n",
            encoding="utf-8",
        )
        record(report)
        prep.no_model_imports()
        require(assets.runtime() == environment, "runtime changed during join")
        prep.final_guard(evidence)
        for item in outputs.values():
            require(assets.sha(item["path"]) == item["sha256"], "join output changed before seal")
        assets.write_json(
            output / "join.provenance.json",
            dict(
                stage=STAGE,
                status="ok",
                started_at=started,
                finished_at=assets.now(),
                inputs=evidence,
                outputs=outputs,
                source_or_ID_changes=False,
                model_calls=0,
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
                    "finite_distances",
                    "finite_permutation_costs",
                    "comparable_in_both_conditions",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
