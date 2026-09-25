"""Saved-only all-mask votes and point dispersion; no image/model/3D/ID inference."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import io
import sys
import time
from collections import Counter
from pathlib import Path

import compare_temporal_point64_v2 as base
import numpy as np

WORK, REPO = base.WORK, base.REPO
STAGE = "temporal_mask_vote_diagnosis_v1"
ARMS = ("lk", "ct_first", "ct_final")
VIEWS = base.VIEWS
BASE_SHA = "2461f383d7c69223c98cb9371058373361a7fb30a409a37d617b60fc2db53790"
CORE_SHA = "3ec308ca7877338f8677c0703d6f180a9d5658c4858b9868451e1152b7d4e25b"
SCHEMA = WORK / "TEMPORAL_MASK_VOTE_SCHEMA.md"
SCHEMA_SHA = "cb7059a22c5381a7576fe722f8061792e04fe1c0ced2f45f476a66209f604952"
DATASETS = (
    dict(
        name="initial",
        first=66483,
        groups=13,
        queries=470,
        root="temporal_point64_comparison_v2",
        stage="temporal_point64_comparison_v2",
        seal="90b35bf8183367f432be4333f16816e58b34c526b387ea1799cce57683913e95",
        preparation="temporal_point64_preparation_v1",
        prep_seal="9771ee33cf36ba320cb2cf5b055c3c44b59f906d835ffac346fccb65241769f3",
        producer="compare_temporal_point64_v2.py",
        producer_sha=BASE_SHA,
    ),
    dict(
        name="known_gap",
        first=333519,
        groups=10,
        queries=334,
        root="temporal_known_gap64_comparison_v1",
        stage="temporal_known_gap64_comparison_v1",
        seal="bd0bae4df3da0d8f2f4a31f0e754a93e89460c6b72803d2054327655662e732e",
        preparation="temporal_known_gap64_preparation_v1",
        prep_seal="447a713a715ccb900883bcebff42ab629fbd55f422b36a0e633ad6ed7f544c8e",
        producer="compare_temporal_known_gap64.py",
        producer_sha="9db089624f62f2d55bd9f883f9ec6ca8408604dd3d61f1a084d38884837272af",
    ),
)
require, same = base.require, base.same


def paired_subsets(lk_reported, ct_reported):
    require(lk_reported.dtype == ct_reported.dtype == bool, "boolean reports required")
    require(lk_reported.shape == ct_reported.shape, "same query scope required")
    return dict(
        common=lk_reported & ct_reported,
        ct_only=ct_reported & ~lk_reported,
        lk_only=lk_reported & ~ct_reported,
        neither=~lk_reported & ~ct_reported,
    )


def subset_metrics(xy, initial, membership, selected, *, available, sampled, inside):
    """Counts retain nonfinite/unsampled points; spread uses finite points separately."""
    q = len(xy)
    require(xy.shape == initial.shape == (q, 2), "coordinate scope differs")
    require(np.isfinite(initial).all(), "initial query coordinates must be finite")
    require(
        membership.ndim == 2 and membership.shape[0] == q and membership.dtype == bool,
        "all-N boolean membership required",
    )
    require(
        all(a.shape == (q,) and a.dtype == bool for a in (selected, sampled, inside)),
        "selection/geometry flag schema differs",
    )
    require(type(available) is bool, "availability must be explicit bool")
    require(available or membership.shape[1] == 0, "unavailable masks have no columns")
    count, n = int(selected.sum()), membership.shape[1]
    selected_indices = np.flatnonzero(selected)
    degrees = membership.sum(axis=1)
    votes = membership[selected].sum(axis=0, dtype=np.int64)
    single_votes = membership[selected & (degrees == 1)].sum(axis=0, dtype=np.int64)
    maximum = int(votes.max()) if n else 0
    ties = votes == maximum
    positive_ties = ties & (maximum > 0)
    single_n = int(np.count_nonzero(selected & (degrees == 1)))
    single_max = int(single_votes.max()) if n else 0
    finite = np.isfinite(xy).all(axis=1)
    chosen = np.flatnonzero(selected & finite)
    points = xy[chosen].astype(np.float64)
    centroid = points.mean(axis=0) if len(points) else np.full(2, np.nan)
    pair_indices = np.asarray(
        [(a, b) for i, a in enumerate(chosen) for b in chosen[i + 1 :]], np.int64
    ).reshape(-1, 2)
    distance = np.linalg.norm(
        xy[pair_indices[:, 0]].astype(np.float64) - xy[pair_indices[:, 1]], axis=1
    )
    initial_distance = np.linalg.norm(
        initial[pair_indices[:, 0]].astype(np.float64) - initial[pair_indices[:, 1]], axis=1
    )
    delta = distance - initial_distance
    rms = (
        float(np.sqrt(np.mean(np.sum((points - centroid) ** 2, axis=1))))
        if len(points) >= 2
        else None
    )
    spread_reason = (
        "ok" if len(points) >= 2 else ("one_finite_point" if len(points) else "no_finite_points")
    )
    vote_reason = (
        "masks_unavailable"
        if not available
        else "no_current_masks"
        if not n
        else "no_selected_points"
        if not count
        else "zero_votes"
        if not maximum
        else "one_selected_point"
        if count == 1
        else "descriptive_votes"
    )
    metadata = dict(
        selected_count=count,
        finite_count=len(chosen),
        nonfinite_count=count - len(chosen),
        sampled_count=int(np.count_nonzero(selected & sampled)),
        continuous_inside_count=int(np.count_nonzero(selected & inside)),
        sampled_outside_continuous_count=int(np.count_nonzero(selected & sampled & ~inside)),
        mask_available=available,
        current_mask_count=n if available else None,
        union_count=int(np.count_nonzero(selected & (degrees > 0))) if available else None,
        single_membership_count=single_n if available else None,
        multi_membership_count=int(np.count_nonzero(selected & (degrees > 1)))
        if available
        else None,
        unassigned_count=int(np.count_nonzero(selected & (degrees == 0))) if available else None,
        unassigned_sampled_count=int(np.count_nonzero(selected & sampled & (degrees == 0)))
        if available
        else None,
        maximum_vote_count=maximum if available else None,
        maximum_vote_fraction=maximum / count if available and count else None,
        single_membership_fraction=single_n / count if available and count else None,
        single_maximum_share=single_max / single_n if available and single_n else None,
        single_share_status=(
            "descriptive_multi_point"
            if single_n >= 2
            else "one_single_member_point"
            if single_n == 1
            else "no_single_members"
        ),
        positive_argmax_count=int(positive_ties.sum()),
        vote_status=vote_reason,
        concentration_comparable=bool(
            available and np.count_nonzero(selected & (degrees > 0)) >= 2
        ),
        dispersion_status=spread_reason,
        rms_radius_px=rms,
        pair_count=len(distance),
        pair_distance_median_px=float(np.median(distance)) if len(distance) else None,
        pair_initial_distance_median_px=float(np.median(initial_distance))
        if len(distance)
        else None,
        pair_distance_change_median_px=float(np.median(delta)) if len(delta) else None,
    )
    arrays = dict(
        votes=votes,
        single_votes=single_votes,
        argmax_ties=ties,
        positive_argmax_ties=positive_ties,
        single_argmax_ties=(single_votes == single_max) & (single_max > 0),
        selected_query_indices=selected_indices,
        finite_query_indices=chosen,
        centroid_xy=centroid,
        pair_query_indices=pair_indices,
        pair_distance_px=distance,
        pair_initial_distance_px=initial_distance,
        pair_distance_change_px=delta,
    )
    return metadata, arrays


def verify_membership(saved, metrics, origin, groups, t, native):
    """Recount saved membership, not a new pixel membership test."""
    keys = {"available", "finite", "in_image", "sampled", "pixel_xy", "membership", "native_frame"}
    for prefix in ("all_query_group_mask_count", "reported_group_mask_count"):
        keys.update((prefix, prefix + "_argmax_ties", prefix + "_positive"))
    require(set(saved) == keys, "saved membership schema differs")
    require(
        saved["native_frame"].shape == ()
        and saved["native_frame"].dtype == np.int64
        and int(saved["native_frame"]) == native,
        "native membership offset differs",
    )
    require(
        saved["available"].shape == () and saved["available"].dtype == bool,
        "available scalar differs",
    )
    available = bool(saved["available"])
    m = saved["membership"]
    require(m.dtype == bool and m.ndim == 2 and m.shape[0] == len(origin), "all-N shape differs")
    require(available or m.shape[1] == 0, "None versus N0 differs")
    geometry = base.core.mask_memberships(metrics["xy"][t], None, (1200, 1200))
    for k in ("finite", "in_image", "sampled", "pixel_xy"):
        same(saved[k], geometry[k], "saved membership geometry " + k)
    require(not m[~saved["sampled"]].any(), "unsampled point carries a mask vote")
    for name, metric in (
        ("finite", "finite"),
        ("in_image", "continuous_inside"),
        ("sampled", "rounded_sampled"),
    ):
        same(saved[name], metrics[metric][t], "metric membership " + name)
    require(bool(metrics["mask_available"][t]) == available, "metric availability differs")
    require(
        int(metrics["mask_count"][t]) == (m.shape[1] if available else -1), "mask count differs"
    )
    same(m.any(axis=1), metrics["union_membership"][t], "union count")
    for prefix, selected in (
        ("all_query_group_mask_count", np.ones(len(origin), bool)),
        ("reported_group_mask_count", metrics["reported"][t]),
    ):
        counts = np.stack([m[(origin == g) & selected].sum(axis=0) for g in range(groups)]).astype(
            np.int64
        )
        same(counts, saved[prefix], prefix)
        ties = (
            counts == counts.max(axis=1, keepdims=True)
            if m.shape[1]
            else np.zeros(counts.shape, bool)
        )
        same(ties, saved[prefix + "_argmax_ties"], "raw ties")
        same(counts.any(axis=1), saved[prefix + "_positive"], "positive votes")
    return available


def members(record, root):
    result = {}
    for value in record["outputs"].values():
        path = Path(value["path"]).resolve()
        require(path.is_relative_to(root) and str(path) not in result, "member path conflict")
        result[str(path)] = value
    return result


def bound_path(root, membership, name, evidence):
    path = base.relative(root, name)
    require(str(path) in membership, "required named member missing")
    row = membership[str(path)]
    evidence.add(path, row["sha256"])
    require(path.stat().st_size == row["size_bytes"], "named member size differs")
    return path


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        require(len(z.files) == len(set(z.files)), "duplicate NPZ fields")
        result = {k: z[k].copy() for k in z.files}
    require(all(a.dtype.kind != "O" for a in result.values()), "object NPZ forbidden")
    return result


def load_dataset(spec, evidence):
    root, prep = WORK / spec["root"], WORK / spec["preparation"]
    seal, mem = base.verified(
        root, "comparison.provenance.json", spec["stage"], spec["seal"], evidence
    )
    summary = base.bound_json(root, mem, "summary.json")
    require(
        summary["schema"] == spec["stage"]
        and summary["status"] == "ok"
        and summary["frames"] == list(range(spec["first"], spec["first"] + 64))
        and summary["query_count"] == spec["queries"]
        and summary["model_forward_calls"] == 21
        and summary["model_initializations"] == 3
        and len(summary["views"]) == 3,
        "full fixed comparison required",
    )
    source = WORK / spec["producer"]
    evidence.add(source, spec["producer_sha"])
    require(
        seal["inputs"].get(str(source)) == spec["producer_sha"]
        and seal["inputs"].get(str(Path(base.core.__file__).resolve())) == CORE_SHA,
        "comparison source/core binding differs",
    )
    pseal = prep / "preparation.provenance.json"
    evidence.add(pseal, spec["prep_seal"])
    require(
        seal["inputs"].get(str(pseal)) == spec["prep_seal"], "comparison preparation edge differs"
    )
    prepared = base.read_json(pseal)
    require(
        prepared["status"] == "ok"
        and prepared["stage"] == spec["preparation"]
        and prepared["finished_at"],
        "successful fixed preparation required",
    )
    pmem = members(prepared, prep)
    rows_path = bound_path(prep, pmem, "query_rows.json", evidence)
    require(
        seal["inputs"].get(str(rows_path)) == evidence.paths[str(rows_path)],
        "query rows edge differs",
    )
    queries = base.read_json(rows_path)
    require(
        [r["view"] for r in queries] == list(VIEWS)
        and sum(len(r["rows"]) for r in queries) == spec["groups"],
        "all initial groups required",
    )
    return dict(
        spec=spec, root=root, prep=prep, members=mem, prep_members=pmem, seal=seal, queries=queries
    )


def load_view(context, view, evidence):
    spec, root = context["spec"], context["root"]
    row = context["queries"][VIEWS.index(view)]
    pp = bound_path(context["prep"], context["prep_members"], row["path"], evidence)
    require(
        evidence.paths[str(pp)] == row["sha256"]
        and context["seal"]["inputs"].get(str(pp)) == row["sha256"],
        "prepared query edge differs",
    )
    original = load_npz(pp)
    saved = load_npz(bound_path(root, context["members"], view + "/queries.npz", evidence))
    require(set(original) == set(saved) == set(row["array_signatures"]), "query fields differ")
    for k in original:
        same(original[k], saved[k], "saved prepared query " + k)
        require(
            base.signature(original[k]) == row["array_signatures"][k], "query signature differs"
        )
    q = len(saved["xy"])
    require(
        saved["xy"].shape == (q, 2)
        and saved["xy"].dtype == np.float32
        and np.isfinite(saved["xy"]).all()
        and q == row["query_count"]
        and int(saved["frame_idx"]) == spec["first"],
        "query native/shape differs",
    )
    expected_origin, expected_corner = [], []
    offset = 0
    for g, group in enumerate(row["rows"]):
        require(group["mask_index"] == g and group["query_start"] == offset, "group order differs")
        count = group["query_count"]
        expected_origin.extend([g] * count)
        expected_corner.extend(range(count))
        offset += count
        require(group["query_stop"] == offset, "group end differs")
    same(saved["mask_index"], np.asarray(expected_origin, np.int64), "initial group index")
    same(saved["corner_index"], np.asarray(expected_corner, np.int64), "initial corner index")
    metrics = {}
    for arm in ARMS:
        metrics[arm] = load_npz(
            bound_path(root, context["members"], f"{view}/{arm}/metrics.npz", evidence)
        )
        a = metrics[arm]
        require(
            a["xy"].shape == (64, q, 2)
            and a["xy"].dtype == np.float32
            and a["reported"].shape == (64, q)
            and a["reported"].dtype == bool,
            "metric shape differs",
        )
    return saved, row, metrics


def selections(metrics, t):
    result = [("own", "reported", arm, metrics[arm]["reported"][t].copy()) for arm in ARMS]
    for ct in ARMS[1:]:
        parts = paired_subsets(metrics["lk"]["reported"][t], metrics[ct]["reported"][t])
        result.extend(
            [
                (ct, "common", "lk", parts["common"]),
                (ct, "common", ct, parts["common"]),
                (ct, "ct_only", ct, parts["ct_only"]),
                (ct, "lk_only", "lk", parts["lk_only"]),
            ]
        )
    return result


def analyze_view(context, view, evidence, output, all_rows, vote_table):
    query, qrow, metrics = load_view(context, view, evidence)
    origin, initial = query["mask_index"], query["xy"]
    groups, spec = len(qrow["rows"]), context["spec"]
    for t in range(64):
        native = spec["first"] + t
        saved = {}
        for arm in ARMS:
            path = bound_path(
                context["root"],
                context["members"],
                f"{view}/{arm}/membership_{native}.npz",
                evidence,
            )
            saved[arm] = load_npz(path)
            verify_membership(saved[arm], metrics[arm], origin, groups, t, native)
        require(
            len({(bool(a["available"]), a["membership"].shape[1]) for a in saved.values()}) == 1,
            "methods do not share current mask columns",
        )
        rows, arrays = [], []
        for g in range(groups):
            for comparison, subset, arm, selected in selections(metrics, t):
                chosen = selected & (origin == g)
                a = saved[arm]
                scalar, detail = subset_metrics(
                    metrics[arm]["xy"][t],
                    initial,
                    a["membership"],
                    chosen,
                    available=bool(a["available"]),
                    sampled=a["sampled"],
                    inside=a["in_image"],
                )
                row = dict(
                    row_id=len(all_rows),
                    window=spec["name"],
                    view=view,
                    native_frame=native,
                    relative_frame=t,
                    initial_query_frame=(t == 0),
                    initial_mask_row=g,
                    initial_group_query_count=qrow["rows"][g]["query_count"],
                    initial_group_status=qrow["rows"][g]["status"],
                    comparison=comparison,
                    subset=subset,
                    arm=arm,
                    **scalar,
                )
                all_rows.append(row)
                rows.append(row)
                arrays.append(
                    dict(
                        votes=detail["votes"],
                        single_votes=detail["single_votes"],
                        argmax_ties=detail["argmax_ties"],
                        positive_argmax_ties=detail["positive_argmax_ties"],
                        single_argmax_ties=detail["single_argmax_ties"],
                        centroid_xy=detail["centroid_xy"],
                        selected=chosen.copy(),
                        finite_selected=chosen & np.isfinite(metrics[arm]["xy"][t]).all(axis=1),
                    )
                )
                for j in range(len(detail["votes"])):
                    vote_table.append(
                        dict(
                            row_id=row["row_id"],
                            current_mask_row=j,
                            vote_count=int(detail["votes"][j]),
                            single_vote_count=int(detail["single_votes"][j]),
                            raw_argmax=bool(detail["argmax_ties"][j]),
                            positive_argmax=bool(detail["positive_argmax_ties"][j]),
                        )
                    )
        block = {k: np.stack([a[k] for a in arrays]) for k in arrays[0]}
        block.update(
            row_id=np.asarray([r["row_id"] for r in rows], np.int64),
            native_frame=np.asarray(native, np.int64),
            initial_mask_index=origin.copy(),
            all_query_group_votes=np.stack([saved[a]["all_query_group_mask_count"] for a in ARMS]),
            reported_group_votes=np.stack([saved[a]["reported_group_mask_count"] for a in ARMS]),
            reported=np.stack([metrics[a]["reported"][t] for a in ARMS]),
        )
        for ct in ARMS[1:]:
            for k, a in paired_subsets(
                metrics["lk"]["reported"][t], metrics[ct]["reported"][t]
            ).items():
                block[ct + "__" + k] = a
        output.write(f"{spec['name']}/{view}/native_{native}.npz", block)
    return dict(
        window=spec["name"],
        view=view,
        initial_groups=groups,
        queries=len(initial),
        native_frames=64,
        membership_archives_verified=64 * 3,
    )


def csv_bytes(rows, fields=None):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else fields or [])
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")


def summarize(rows):
    groups = {}
    for row in rows:
        key = (row["window"], row["view"], row["comparison"], row["subset"], row["arm"])
        groups.setdefault(key, []).append(row)
    result = []
    for key, values in groups.items():
        tracking = [r for r in values if not r["initial_query_frame"]]
        statistics = {}
        for name in (
            "selected_count",
            "maximum_vote_fraction",
            "single_membership_fraction",
            "single_maximum_share",
            "rms_radius_px",
            "pair_distance_change_median_px",
        ):
            a = np.asarray([np.nan if r[name] is None else r[name] for r in tracking], np.float64)
            statistics[name] = base.finite_distribution(a)
        result.append(
            dict(
                zip(("window", "view", "comparison", "subset", "arm"), key, strict=True),
                all64_group_times=len(values),
                tracking63_group_times=len(tracking),
                selected_point_frames=sum(r["selected_count"] for r in tracking),
                vote_status_counts=dict(Counter(r["vote_status"] for r in tracking)),
                dispersion_status_counts=dict(Counter(r["dispersion_status"] for r in tracking)),
                concentration_comparable_group_times=sum(
                    r["concentration_comparable"] for r in tracking
                ),
                single_share_status_counts=dict(
                    Counter(r["single_share_status"] for r in tracking)
                ),
                descriptive_distributions=statistics,
            )
        )
    return result


def plots(rows, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"lk": "#1d4ed8", "ct_first": "#c2410c", "ct_final": "#15803d"}
    keys = sorted({(r["window"], r["view"], r["initial_mask_row"]) for r in rows})
    for window, view, group in keys:
        selected = [
            r
            for r in rows
            if (r["window"], r["view"], r["initial_mask_row"]) == (window, view, group)
        ]
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
        specs = [
            ("selected_count", "Reported query count", False),
            ("maximum_vote_fraction", "Largest current-mask vote / reported points", False),
            ("single_maximum_share", "Largest share among single-membership points", False),
            ("rms_radius_px", "Reported finite-point RMS radius (px)", False),
            ("maximum_vote_fraction", "CT-only: largest vote / selected points", True),
            ("rms_radius_px", "CT-only: finite-point RMS radius (px)", True),
        ]
        for ax, (metric, title, only) in zip(axes.flat, specs, strict=True):
            for arm in ARMS[1:] if only else ARMS:
                a = [
                    r
                    for r in selected
                    if r["arm"] == arm
                    and (
                        r["comparison"] == arm and r["subset"] == "ct_only"
                        if only
                        else r["comparison"] == "own"
                    )
                ]
                ax.plot(
                    [r["relative_frame"] for r in a],
                    [np.nan if r[metric] is None else r[metric] for r in a],
                    color=colors[arm],
                    label=arm,
                )
            ax.set_title(title, fontsize=10)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
            ax.axvline(0, color="gray", linestyle=":")
            ax.set_xlabel("Relative native frame (0 = query anchor)")
        fig.suptitle(
            f"{window} / {view} / initial mask row {group}\n"
            "Anonymous instance votes; overlap retained; no physical ID truth"
        )
        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=130)
        plt.close(fig)
        output.write(f"figures/{window}_{view}_group{group}.png", buffer.getvalue())


def run(destination):
    destination = Path(destination).resolve()
    require(
        destination.is_relative_to(WORK) and not destination.exists(),
        "new workspace output required",
    )
    evidence, output = base.Evidence(), base.Output(destination)
    started, clock = base.utc(), time.perf_counter()
    try:
        source = Path(__file__).resolve()
        for path, digest in (
            (source, None),
            (Path(base.__file__).resolve(), BASE_SHA),
            (Path(base.core.__file__).resolve(), CORE_SHA),
            (SCHEMA, SCHEMA_SHA),
        ):
            evidence.add(path, digest)
            output.write("code_snapshot/" + path.name, path.read_bytes())
        output.write(
            "plan.json",
            dict(
                schema=STAGE,
                schema_document_sha256=SCHEMA_SHA,
                datasets=DATASETS,
                arms=ARMS,
                expected_initial_groups=23,
                group_times_per_arm=1472,
                primary="ct_first vs lk",
                secondary="ct_final vs lk",
                subset_flags="algorithm-specific reported, no finite gate",
                mask_rows_are_per_native=True,
                no_physical_ID_truth=True,
                thresholds_added=0,
                unavailable_N0_zero_votes_single_point_distinct=True,
                pair_distances=(
                    "recomputed on same finite-selected query pairs; scalar medians saved"
                ),
                source_mask_scope="sealed saved membership; no raw/mask pixels reread",
            ),
        )
        contexts = [load_dataset(s, evidence) for s in DATASETS]
        rows, votes, view_results = [], [], []
        for context in contexts:
            for view in VIEWS:
                view_results.append(analyze_view(context, view, evidence, output, rows, votes))
                print(
                    base.json_bytes(dict(event="view_analyzed", **view_results[-1])).decode(),
                    flush=True,
                )
        require(
            sum(r["initial_groups"] for r in view_results) == 23 and len(rows) == 1472 * 11,
            "fixed all-group denominator differs",
        )
        output.write("group_time_rows.json", rows)
        output.write("group_time_rows.csv", csv_bytes(rows))
        output.write(
            "all_current_mask_votes.csv",
            csv_bytes(
                votes,
                fields=[
                    "row_id",
                    "current_mask_row",
                    "vote_count",
                    "single_vote_count",
                    "raw_argmax",
                    "positive_argmax",
                ],
            ),
        )
        plots(rows, output)
        summary = dict(
            schema=STAGE,
            status="ok",
            initial_groups=23,
            group_times_per_arm=1472,
            own_arm_rows=4416,
            paired_subset_rows=11776,
            total_rows=len(rows),
            full_mask_vote_rows=len(votes),
            views=view_results,
            groups=summarize(rows),
            source_membership_recount_exact=True,
            prepared_query_bytes_exact=True,
            model_calls=0,
            GPU=False,
            raw_image_reads=0,
            physical_ID_truth=False,
            ID_or_review_changed=False,
            new_thresholds=0,
            total_before_final_guard_seconds=time.perf_counter() - clock,
            environment=dict(
                python=sys.version,
                packages={
                    name: importlib.metadata.version(name)
                    for name in ("numpy", "opencv-python", "matplotlib")
                },
            ),
        )
        output.write("summary.json", summary)
        output.write(
            "RESULT.md",
            (
                b"# Saved temporal mask votes\n\n"
                b"All 23 initial groups and 64 times remain in the tables and figures. "
                b"CT-first is primary; CT-final is secondary. Frame0 is retained separately.\n\n"
                b"Votes describe current detector instances, including overlap and empty cases. "
                b"Mask row numbers are not temporal identities. "
                b"Reported flags have different meanings. "
                b"One point cannot establish dispersion or physical identity; "
                b"zero votes remain explicit. "
                b"No raw pixels, model, GPU, 3D, identity or review changes.\n"
            ),
        )
        require(
            not any(k == "torch" or k.startswith("cotracker") for k in sys.modules),
            "model imported",
        )
        evidence.final_guard()
        output.final_guard()
        seal = dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=base.utc(),
            inputs=evidence.paths,
            outputs=output.records.copy(),
        )
    except BaseException as exc:
        output.write(
            "diagnosis.provenance.json",
            dict(
                stage=STAGE,
                status="failed",
                started_at=started,
                finished_at=base.utc(),
                error_type=type(exc).__name__,
                error=str(exc),
                inputs=evidence.paths,
                outputs=output.records.copy(),
            ),
        )
        raise
    output.write("diagnosis.provenance.json", seal)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
