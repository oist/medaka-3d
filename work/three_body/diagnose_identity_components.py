"""Fixed old-six-frame geometry-conditioned temporal appearance diagnostic.

All 90 requests and 540 permutation slots remain, including unscorable cases.
Geometry references are written before any PNG decode or descriptor extraction.
No Tail candidate, raw video, new sampling, fitting, tracking, GPU or ID labels.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from itertools import combinations, permutations
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "labs/20260909_three_body")]

import appearance_diagnostic as appearance
import cv2
import diagnose_instance_mask_coherence_v2 as coh
import numpy as np
from prepare_sequence import array_sha256

checks, WORK, REPO = coh.checks, coh.WORK, coh.REPO
FRAMES, METHODS, VIEWS = coh.FRAMES, coh.METHODS, tuple(coh.VIEWS)
PAIRS = tuple(combinations(FRAMES, 2))
PERMUTATIONS = np.asarray(tuple(permutations(range(3))), np.int64)
STAGE = "identity_components_fixed6_v1"
COHERENCE = WORK / "bounded_instance_mask_coherence_6_v2"
COHERENCE_SHA = "c0311bdea7dccda1ff519ec698a547026dc0a9c4f9a7566e816b34424854dd60"
PINS = {
    WORK / "IDENTITY_NEXT_DECISION.md": (
        "abeaa2959e9316f0c36f076d2631b38bd2ae0dd0a958cbb07bce5ba4af64646b"
    ),
    Path(
        coh.__file__
    ).resolve(): "39aa2b6bbcd24200b4c1326ec923c06f3e331a20d4624687630bdbbb1dd487fa",
    Path(
        appearance.__file__
    ).resolve(): "17a03a828958bba2402710f59bd9071f1c83649268ca1b3f62863f880c2ff135",
}
TIE_ATOL = 1e-12


def verified(root, filename, stage, digest, hashes):
    coh.bind(root / filename, digest, hashes)
    before = dict(hashes)
    seal = checks.verify(root, filename, stage, hashes)
    checks.require(
        seal.get("finished_at") and all(hashes.get(k) == v for k, v in before.items()),
        "completed pinned evidence changed",
    )
    return seal


def output(root, seal, name, hashes):
    path = (root / name).resolve(strict=True)
    item = seal["outputs"].get(name)
    checks.require(item is not None and Path(item["path"]).resolve() == path, "output role/path")
    return coh.bind(path, item["sha256"], hashes)


def npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def load_inputs(hashes):
    raw = verified(
        coh.RAW, "review.provenance.json", "bounded_shape_raw_review_v1", coh.RAW_SHA, hashes
    )
    audit = verified(
        coh.AUDIT,
        "audit.provenance.json",
        "bounded_mouth_first_chunk_interim_audit_v1",
        coh.AUDIT_SHA,
        hashes,
    )
    proof = verified(COHERENCE, "diagnosis.provenance.json", coh.STAGE, COHERENCE_SHA, hashes)
    for seal in (raw, proof):
        coh.input_member(seal, coh.AUDIT / "audit.provenance.json", hashes)
    coh.input_member(proof, coh.RAW / "review.provenance.json", hashes)
    summaries = [
        checks.read(output(root, seal, "summary.json", hashes))
        for root, seal in ((coh.RAW, raw), (coh.AUDIT, audit), (COHERENCE, proof))
    ]
    checks.require(
        all(s["status"] == "ok" and s["horizon_stop"] == 68060 for s in summaries),
        "same68060 success required",
    )
    checks.require(
        summaries[0]["frames"] == list(FRAMES)
        and summaries[2]["frames"] == list(FRAMES)
        and summaries[2]["methods"] == list(METHODS)
        and summaries[2]["body_view_records"] == 108,
        "fixed six/108 scope",
    )
    shared = summaries[1]["shared_input_binding"]["inputs"]["seed"]["SL_and_calibration_sha256"]
    checks.require(len(shared) == 7, "same SL3/calibration4 required")
    for path, digest in shared.items():
        coh.bind(path, digest, hashes)
        for seal in (raw, audit, proof):
            coh.input_member(seal, path, hashes)
    calibration = next(
        Path(p).parent for p in shared if Path(p).name == "calibration_manifest.json"
    )
    manifest = checks.read(calibration / "calibration_manifest.json")
    checks.require(
        {str((calibration / x["path"]).resolve()) for x in manifest["model_files"]}
        | {str(calibration / "calibration_manifest.json")}
        == {p for p in shared if Path(p).suffix == ".json"},
        "actual calibration file set",
    )
    rig = coh.load_rig(calibration)
    traces = {}
    for method in METHODS:
        path = output(coh.AUDIT, audit, f"{method}_same68060_candidate.npz", hashes)
        for seal in (raw, proof):
            coh.input_member(seal, path, hashes)
        values = npz(path)
        checks.require(
            len(values) == 21 and np.array_equal(values["frame_idx"], np.arange(66400, 68060)),
            "all21 same1660 candidate",
        )
        traces[method] = {
            k: values[k][np.asarray(FRAMES) - 66400].copy()
            for k in ("pose_xyz", "observed", "sources", "identity_review")
        }
        geometry = summaries[1]["methods"][method]["all_native_source_geometry"]
        checks.require(
            geometry["ray_gate_mm"] == 2
            and geometry["native_pixel_gate"] == 12
            and geometry["all_sources_and_observed_flags_pass"] is True
            and geometry["source_rows_checked"] == int((values["sources"] >= 0).sum())
            and geometry["observed_nodes_checked"] == int(values["observed"].sum()),
            "strict saved-source proof",
        )
        run = (
            "directional_one_minute_66400-76360_v1"
            if method == "directional"
            else "bounded_mouth_one_minute_66400-76360_v1"
        )
        meta = coh.input_member(
            audit, WORK / run / "checkpoints/through_068060/metadata.json", hashes
        )
        coh.validate_rig_context(checks.read(meta)["context"], rig)
    records = checks.read(output(COHERENCE, proof, "records.json", hashes))
    keys = [(r["frame_idx"], r["method"], r["view"], r["slot"]) for r in records]
    expected = {(n, m, v, s) for n in FRAMES for m in METHODS for v in VIEWS for s in range(3)}
    checks.require(len(keys) == 108 and set(keys) == expected, "all108 unique coherence records")
    evidence = checks.read(output(coh.RAW, raw, "native_observation_evidence.json", hashes))
    checks.require(
        len(evidence) == 18
        and {(e["frame_idx"], e["view"]) for e in evidence}
        == {(n, v) for n in FRAMES for v in VIEWS},
        "all18 raw evidence",
    )
    return dict(
        raw=raw,
        proof=proof,
        traces=traces,
        rig=rig,
        shared=shared,
        records=dict(zip(keys, records, strict=True)),
        evidence={(e["frame_idx"], e["view"]): e for e in evidence},
    )


def geometry_reference(masks, available, projected, observed, sources, metrics):
    """No images/descriptors accepted: geometry decisions cannot depend on color."""
    direct = observed[:, :2].all(-1) & (sources[:, :2] >= 0).all(-1)
    assigned, assignment_reasons = appearance.assign_instances(
        masks if available else None, projected[:, :2].copy(), direct.copy()
    )
    eligible = np.zeros(3, bool)
    rows = []
    for slot, body in enumerate(metrics):
        reasons = []
        if assignment_reasons[slot] != "usable":
            reasons.append(str(assignment_reasons[slot]))
        if body["requested_points"] != 33 or body["valid_points"] != 33:
            reasons.append("incomplete_33_projection")
        best = body["best_mask_rows"]
        if len(best) != 1:
            reasons.append("coherence_no_unique_minimum")
        elif assigned[slot] >= 0 and int(assigned[slot]) != best[0]:
            reasons.append("assignment_coherence_disagree")
        if not np.isfinite(body["C_single"]):
            reasons.append("coherence_nonfinite")
        eligible[slot] = not reasons
        rows.append(
            dict(
                slot=slot,
                assigned_mask_row=int(assigned[slot]),
                assignment_reason=str(assignment_reasons[slot]),
                direct_mouth_chest=bool(direct[slot]),
                observed_mouth_chest=observed[slot, :2].tolist(),
                view_sources_mouth_chest=sources[slot, :2].tolist(),
                geometry_eligible=bool(eligible[slot]),
                reasons=reasons,
                body_coherence=body,
            )
        )
    distinct = len(set(assigned[assigned >= 0].tolist())) == 3 and (assigned >= 0).all()
    return dict(
        slots=rows,
        assigned_mask_rows=assigned.tolist(),
        slot_geometry_eligible=eligible.tolist(),
        all3_distinct=bool(distinct),
        all3_geometry_eligible=bool(eligible.all() and distinct),
    )


def slot_descriptors(reference, all_features):
    values = np.full((3, 6), np.nan)
    reasons = []
    for slot, row in enumerate(reference["slots"]):
        why = list(row["reasons"])
        index = row["assigned_mask_row"]
        core_valid = index >= 0 and np.isfinite(all_features[index]).all()
        if index >= 0 and not core_valid:
            why.append("assigned_core_invalid")
        if row["geometry_eligible"] and core_valid:
            values[slot] = all_features[index]
        reasons.append(why)
    scorable = reference["all3_geometry_eligible"] and np.isfinite(values).all()
    return values, reasons, bool(scorable)


def pair_costs(left, right, left_ready, right_ready):
    checks.require(left.shape == right.shape == (3, 6), "three fixed six-component descriptors")
    good = np.isfinite(left).all(-1)[:, None] & np.isfinite(right).all(-1)[None, :]
    distances = np.full((3, 3), np.nan)
    for i, j in np.argwhere(good):
        distances[i, j] = np.linalg.norm(left[i] - right[j])
    costs = np.full(6, np.nan)
    scorable = bool(left_ready and right_ready)
    best = []
    gap = None
    if scorable:
        checks.require(good.all(), "scorable pair lacks a descriptor")
        costs = np.asarray([distances[np.arange(3), p].sum() / 3 for p in PERMUTATIONS])
        best = np.flatnonzero(np.isclose(costs, costs.min(), rtol=0, atol=TIE_ATOL)).tolist()
        ordered = np.sort(costs)
        gap = float(ordered[1] - ordered[0])
    return dict(
        distances=distances,
        permutation_costs=costs,
        scorable=scorable,
        best_permutation_indices=best,
        second_minus_first=gap,
    )


def cross_view_ranks(requests):
    rows = []
    for method in METHODS:
        for early, late in PAIRS:
            by_view = {
                r["view"]: r
                for r in requests
                if r["method"] == method
                and r["earlier_frame"] == early
                and r["later_frame"] == late
            }
            top = by_view[VIEWS[0]]
            record = dict(
                method=method,
                earlier_frame=early,
                later_frame=late,
                top_scorable=top["scorable"],
                top_best_permutation_indices=top["best_permutation_indices"],
                sides={},
            )
            for view in VIEWS[1:]:
                side = by_view[view]
                ranked = []
                if top["scorable"] and side["scorable"]:
                    values = np.asarray(side["permutation_costs"])
                    for p in top["best_permutation_indices"]:
                        less = (values < values[p]) & ~np.isclose(
                            values, values[p], rtol=0, atol=TIE_ATOL
                        )
                        ranked.append(
                            dict(
                                permutation_index=p,
                                competition_rank=1 + int(less.sum()),
                                side_best=p in side["best_permutation_indices"],
                                side_cost=float(values[p]),
                            )
                        )
                record["sides"][view] = dict(
                    scorable=side["scorable"],
                    comparison_available=bool(top["scorable"] and side["scorable"]),
                    top_minimum_ranks=ranked,
                )
            rows.append(record)
    return rows


def run(destination):
    destination = Path(destination).resolve()
    checks.require(destination.is_relative_to(WORK) and not destination.exists(), "new work output")
    code = coh.source_code()
    hashes = dict(code)
    outputs = {}
    for path, digest in PINS.items():
        coh.bind(path, digest, hashes)
    destination.mkdir(parents=True)
    with coh.provenance(STAGE, destination, name="diagnosis") as record:

        def emit(name, value=None):
            path = destination / name
            if value is not None:
                path.write_text(
                    json.dumps(coh.encode_nonfinite(value), indent=2, allow_nan=False),
                    encoding="utf-8",
                )
            outputs[str(path)] = checks.sha(path)
            record.output(name, path, sha256=outputs[str(path)])

        emit(
            "plan.json",
            dict(
                schema=STAGE,
                frames=FRAMES,
                methods=METHODS,
                views=VIEWS,
                pairs=PAIRS,
                requested_pair_views=90,
                permutation_slots=540,
                permutations=PERMUTATIONS,
                descriptor_features=appearance.FEATURES,
                core_rule="unchanged describe_core 5x5 ellipse erosion / minimum12",
                tie_atol=TIE_ATOL,
                coherence_ties="exact equal minima, unchanged",
                aggregate_gate="both frames all3 distinct geometry-valid references and core-valid",
                partial_distances=(
                    "individual eligible slot endpoints retained even if whole request unscorable"
                ),
                no_resampling=True,
                no_threshold_change=True,
                Tail_excluded=True,
                posthoc_fixed_frames=True,
                biological_ID_truth=False,
            ),
        )
        emit(
            "schema.json",
            dict(
                schema=STAGE,
                geometry="36 frame/method/view groups, each with all3 slot records (108 total)",
                descriptors="18 NPZ files, each retains allN mask rows as float64[N,6]",
                pair_order="method, earlier/later pair, view; fixed90 requests",
                distance="float64[90,3,3]; eligible endpoints only; NaN with JSON reasons",
                costs="float64[90,6]; all6 NaN unless both all3 geometry/core gates pass",
                permutation_direction="earlier slot i maps to later slot permutation[i]",
                nonfinite_JSON="NaN string; NPZ retains IEEE NaN",
                cross_view="30 method/time pairs; every tied top minimum; missing remains missing",
            ),
        )
        decision = destination / "registered_decision.md"
        decision.write_bytes((WORK / "IDENTITY_NEXT_DECISION.md").read_bytes())
        checks.require(
            checks.sha(decision) == PINS[WORK / "IDENTITY_NEXT_DECISION.md"],
            "registered decision changed",
        )
        emit("registered_decision.md")
        data = load_inputs(hashes)
        observations = {}
        geometry = {}
        geometry_rows = []
        # Complete and serialize every geometry reference before decoding color.
        for fi, native in enumerate(FRAMES):
            for vi, view in enumerate(VIEWS):
                obs_path = output(coh.RAW, data["raw"], f"observation_{native}_{view}.npz", hashes)
                coh.input_member(data["proof"], obs_path, hashes)
                obs = npz(obs_path)
                evidence = data["evidence"][native, view]
                masks = obs["masks"]
                available = bool(obs["mask_available"])
                checks.require(
                    masks.dtype == bool and masks.ndim == 3 and masks.shape[1:] == (1200, 1200),
                    "native allN masks",
                )
                checks.require(
                    (
                        available
                        and array_sha256(masks) == evidence["mask_sha256"]
                        and len(masks) == evidence["mask_count"]
                    )
                    or (not available and len(masks) == 0 and evidence["mask_sha256"] is None),
                    "native mask evidence",
                )
                points = obs["detections"]
                signature = dict(
                    dtype=points.dtype.str, shape=list(points.shape), sha256=array_sha256(points)
                )
                checks.require(signature == evidence["detection_signature"], "SL signature")
                saved = npz(
                    output(COHERENCE, data["proof"], f"coherence_{native}_{view}.npz", hashes)
                )
                checks.require(
                    int(saved["frame_idx"]) == native
                    and tuple(saved["methods"]) == METHODS
                    and saved["outside_distance"].shape == (len(masks) + 1, 198),
                    "coherence native/allN schema",
                )
                for mi, method in enumerate(METHODS):
                    scene = data["traces"][method]["pose_xyz"][fi]
                    projected = data["rig"].camera(view).project(scene)
                    checks.require(
                        checks.equal_bytes(projected, obs[method + "_projection"]),
                        "old endpoint projection",
                    )
                    bodies = [coh.project_body(pose, data["rig"].camera(view)) for pose in scene]
                    checks.require(
                        checks.equal_bytes(
                            np.stack([b["points_xyz"] for b in bodies]), saved["points_xyz"][mi]
                        )
                        and checks.equal_bytes(
                            np.stack([b["projected_uv"] for b in bodies]), saved["projected_uv"][mi]
                        ),
                        "fixed33 coherence geometry differs",
                    )
                    metrics = []
                    for slot in range(3):
                        body = coh.coherence_metrics(saved, np.arange(33) + (mi * 3 + slot) * 33)
                        prior = data["records"][native, method, view, slot]["metrics"]["body"]
                        checks.require(
                            coh.encode_nonfinite(body) == prior, "all108 coherence records differ"
                        )
                        metrics.append(body)
                    ref = geometry_reference(
                        masks,
                        available,
                        projected,
                        data["traces"][method]["observed"][fi],
                        data["traces"][method]["sources"][fi, :, vi],
                        metrics,
                    )
                    ref.update(frame_idx=native, method=method, view=view, mask_count=len(masks))
                    geometry[method, native, view] = ref
                    geometry_rows.append(ref)
                observations[native, view] = obs
        emit("geometry_references_before_color.json", geometry_rows)
        geometry_sha = outputs[str(destination / "geometry_references_before_color.json")]
        feature_tables = {}
        descriptor_rows = []
        for native in FRAMES:
            for view in VIEWS:
                obs = observations[native, view]
                masks = obs["masks"]
                png = output(coh.RAW, data["raw"], f"raw_{native}_{view}.png", hashes)
                image = cv2.imread(str(png), cv2.IMREAD_COLOR)
                checks.require(
                    image is not None
                    and image.dtype == np.uint8
                    and image.shape == (1200, 1200, 3)
                    and array_sha256(image) == data["evidence"][native, view]["bgr_sha256"],
                    "native PNG BGR",
                )
                features = np.full((len(masks), 6), np.nan)
                core_counts = []
                core_sha = []
                mask_rows = []
                for index, mask in enumerate(masks):
                    descriptor, core = appearance.describe_core(image, mask)
                    if descriptor is not None:
                        features[index] = descriptor
                    core_counts.append(int(core.sum()))
                    core_sha.append(array_sha256(core))
                    mask_rows.append(
                        dict(
                            mask_row=index,
                            mask_sha256=array_sha256(mask),
                            core_sha256=core_sha[-1],
                            core_pixels=core_counts[-1],
                            descriptor_valid=descriptor is not None,
                            reason="valid" if descriptor is not None else "core_below12",
                            descriptor=features[index],
                        )
                    )
                feature_tables[native, view] = features
                np.savez_compressed(
                    destination / f"descriptors_{native}_{view}.npz",
                    frame_idx=np.asarray(native, np.int64),
                    mask_rows=np.arange(len(masks)),
                    descriptors=features,
                    core_pixel_count=np.asarray(core_counts, np.int64),
                    core_sha256=np.asarray(core_sha, dtype="S64"),
                    mask_available=obs["mask_available"],
                )
                emit(f"descriptors_{native}_{view}.npz")
                descriptor_rows.append(
                    dict(frame_idx=native, view=view, all_mask_count=len(masks), masks=mask_rows)
                )
        emit("all_mask_descriptors.json", descriptor_rows)
        contexts = {}
        context_rows = []
        for key, reference in geometry.items():
            method, native, view = key
            features, reasons, scorable = slot_descriptors(reference, feature_tables[native, view])
            contexts[key] = (features, reasons, scorable)
            context_rows.append(
                dict(
                    method=method,
                    frame_idx=native,
                    view=view,
                    slot_descriptors=features,
                    slot_reasons=reasons,
                    all3_distinct=reference["all3_distinct"],
                    scorable=scorable,
                )
            )
        emit("descriptor_contexts.json", context_rows)
        requests = []
        distances = []
        costs = []
        for method in METHODS:
            for early, late in PAIRS:
                for view in VIEWS:
                    a, ar, ag = contexts[method, early, view]
                    b, br, bg = contexts[method, late, view]
                    answer = pair_costs(a, b, ag, bg)
                    requests.append(
                        dict(
                            method=method,
                            earlier_frame=early,
                            later_frame=late,
                            view=view,
                            earlier_slot_reasons=ar,
                            later_slot_reasons=br,
                            earlier_ready=ag,
                            later_ready=bg,
                            distance_entry_reasons=[
                                [
                                    []
                                    if np.isfinite(answer["distances"][i, j])
                                    else dict(earlier=ar[i], later=br[j])
                                    for j in range(3)
                                ]
                                for i in range(3)
                            ],
                            request_reason="scorable"
                            if answer["scorable"]
                            else "one_or_both_all3_geometry_core_gates_failed",
                            **answer,
                        )
                    )
                    distances.append(answer["distances"])
                    costs.append(answer["permutation_costs"])
        checks.require(len(requests) == 90 and len(costs) * 6 == 540, "all fixed requests retained")
        emit("pair_requests.json", requests)
        np.savez_compressed(
            destination / "pair_costs.npz",
            distance=np.stack(distances),
            permutation_costs=np.stack(costs),
            permutations=PERMUTATIONS,
            scorable=np.asarray([r["scorable"] for r in requests]),
            earlier_frame=np.asarray([r["earlier_frame"] for r in requests]),
            later_frame=np.asarray([r["later_frame"] for r in requests]),
            methods=np.asarray([r["method"] for r in requests]),
            views=np.asarray([r["view"] for r in requests]),
        )
        emit("pair_costs.npz")
        cross = cross_view_ranks(requests)
        emit("cross_view_ranks.json", cross)
        summary = dict(
            schema=STAGE,
            status="ok",
            frames=FRAMES,
            methods=METHODS,
            views=VIEWS,
            requested_pair_views=90,
            permutation_cost_slots=540,
            scorable_requests=sum(r["scorable"] for r in requests),
            finite_distance_elements=int(np.isfinite(distances).sum()),
            distance_slots=810,
            geometry_body_view_records=108,
            geometry_before_color_sha256=geometry_sha,
            all_mask_descriptors=sum(len(r["masks"]) for r in descriptor_rows),
            valid_mask_descriptors=sum(
                m["descriptor_valid"] for r in descriptor_rows for m in r["masks"]
            ),
            by_method_view=[
                dict(
                    method=m,
                    view=v,
                    requested=15,
                    scorable=sum(
                        r["scorable"] for r in requests if r["method"] == m and r["view"] == v
                    ),
                )
                for m in METHODS
                for v in VIEWS
            ],
            geometry_reason_counts=dict(
                Counter(
                    reason
                    for r in geometry_rows
                    for slot in r["slots"]
                    for reason in slot["reasons"]
                )
            ),
            shared_SL_calibration_sha256=data["shared"],
            raw_png_decodes=18,
            raw_video_decodes=0,
            GPU=False,
            frame_reselection=False,
            threshold_changes=False,
            new_ID_labels=False,
            biological_ID_truth=False,
            limitations=[
                "Conditional appearance under saved geometry, not male identity accuracy.",
                "All masks stay anonymous; merged/split/reflected/duplicate masks are not IDs.",
                "Partial descriptor distances remain even without any valid six-permutation score.",
                "Geometry costs are invariant to relabeling the same simultaneous body set; "
                "temporal color evidence is separate.",
                "Fixed posthoc failure cases, not an encounter benchmark; Tail excluded.",
            ],
        )
        emit("summary.json", summary)
        table = "\n".join(
            f"| {r['method']} | {r['view']} | {r['scorable']} / 15 |"
            for r in summary["by_method_view"]
        )
        (destination / "RESULT.md").write_text(
            "# 固定6時刻の条件付き外見診断\n\n"
            f"全90要求・540順列cost枠を保持。採点可能 {summary['scorable_requests']}/90、"
            f"部分距離は {summary['finite_distance_elements']}/810 要素。\n\n"
            "| 方式 | 視点 | 採点可能pair |\n|---|---|---:|\n" + table + "\n\n"
            "全108身体・視点の幾何参照を色抽出前に保存した。直接口胸支持と同一mask包含、全潜在競合、33点coherence唯一最小、"
            "3slot別mask、元5x5coreの有効性が両時刻で成立した場合だけ全6costを評価した。"
            "成立しない場合も全N mask記述子・理由・部分3x3距離を残し、"
            "NaNをゼロ誤差や一致へ変換しない。\n\n"
            "上面の全同順位最小を側面で評価可能な時だけ順位照合する。視点の重み付け・多数決・ID受理は行わない。"
            "この固定時刻は既観察の形状失敗から選ばれた診断であり、オスのID真値や確率ではない。"
            "支持や距離の閾値を緩めず、不能なpairを時刻差替えで救済しない。\n",
            encoding="utf-8",
        )
        emit("RESULT.md")
        for path, digest in code.items():
            snap = destination / "code_snapshot" / Path(path).relative_to(REPO)
            snap.parent.mkdir(parents=True, exist_ok=True)
            snap.write_bytes(Path(path).read_bytes())
            checks.require(checks.sha(snap) == digest, "source snapshot changed")
            emit(str(snap.relative_to(destination)))
        checks.require(code == coh.source_code(), "loaded source set changed")
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
        for path, digest in {**hashes, **outputs}.items():
            checks.require(checks.sha(path) == digest, "source/input/output changed")
        record.param("all_source_input_snapshot_output_SHA_unchanged", True)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
