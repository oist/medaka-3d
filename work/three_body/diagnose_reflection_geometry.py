"""Post-hoc geometry/source/mask overlap in the two sealed 200-frame outputs.

Only saved direct/Fresnel results are compared. No old trajectory, human labels,
fitting, rendering, raw-image decoding or CUDA calls are used. Close geometry or
a shared detector instance is a possible duplicated explanation, not fish truth.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_reflection_temporal as audit  # noqa: E402
from measure_scene0_body_proximity import polyline_closest  # noqa: E402

from medaka3d.provenance import provenance  # noqa: E402

ROOT = HERE.parents[1]
RUN = HERE / "reflection_temporal_200_v1"
AUDIT = HERE / "reflection_temporal_200_independent_audit_v1"
PAIRS = tuple(itertools.combinations(range(3), 2))
ARMS = audit.ARMS
VIEWS = audit.producer.VIEWS
producer = audit.producer
require = producer.require
sha256 = audit.sha256


def bilinear_membership(masks, pixels):
    """Return M,F,N mask values and F,N known flags at native pixel centers.

    Outside the image is unknown, not observed background. A value >=0.5 is a
    fixed post-hoc membership rule only; it is not a tracker support criterion.
    """
    masks, pixels = np.asarray(masks), np.asarray(pixels)
    require(
        masks.ndim == 3 and masks.dtype == bool and pixels.shape == (3, 3, 2),
        "mask/native projection schema differs",
    )
    height, width = masks.shape[1:]
    known = np.isfinite(pixels).all(-1) & (pixels[..., 0] >= 0) & (pixels[..., 0] <= width - 1)
    known &= (pixels[..., 1] >= 0) & (pixels[..., 1] <= height - 1)
    safe = np.where(known[..., None], pixels, 0)
    lo = np.floor(safe).astype(int)
    x0, y0 = lo[..., 0], lo[..., 1]
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    dx, dy = safe[..., 0] - x0, safe[..., 1] - y0
    values = (
        masks[:, y0, x0] * (1 - dx) * (1 - dy)
        + masks[:, y0, x1] * dx * (1 - dy)
        + masks[:, y1, x0] * (1 - dx) * dy
        + masks[:, y1, x1] * dx * dy
    )
    values[:, ~known] = np.nan
    return values, known


def summarize_series(values, frames):
    values = np.asarray(values)
    return dict(
        minimum=float(values.min()),
        median=float(np.median(values)),
        maximum=float(values.max()),
        minimum_frame=int(frames[np.argmin(values)]),
        maximum_frame=int(frames[np.argmax(values)]),
    )


def geometry(data):
    pose, source = data["pose_xyz"], data["sources"]
    lengths = np.linalg.norm(np.diff(pose, axis=2), axis=-1)
    total = lengths.sum(-1)
    pectoral, node_distance, segment_distance = [], [], []
    for t in range(len(pose)):
        pectoral.append([np.linalg.norm(pose[t, a, 1] - pose[t, b, 1]) for a, b in PAIRS])
        node_distance.append(
            [
                np.linalg.norm(pose[t, a, :, None] - pose[t, b, None, :], axis=-1).min()
                for a, b in PAIRS
            ]
        )
        segment_distance.append(
            [polyline_closest(pose[t, a], pose[t, b])["distance_mm"] for a, b in PAIRS]
        )
    pectoral = np.asarray(pectoral)
    normalized = np.stack(
        [
            pectoral[:, pair] / ((total[:, a] + total[:, b]) / 2)
            for pair, (a, b) in enumerate(PAIRS)
        ],
        axis=1,
    )
    same_node = np.zeros((len(pose), 3, 3, 3), bool)  # T,V,pair,node
    shared_rows = np.zeros((len(pose), 3, 3), np.int64)
    mixed = np.zeros((len(pose), 3, 3), bool)  # T,F,V
    rows_testable = np.zeros_like(mixed)
    for t in range(len(pose)):
        for view in range(3):
            rowsets = []
            for fish in range(3):
                rows = source[t, fish, view]
                rowsets.append(set(rows[rows >= 0].tolist()))
                rows_testable[t, fish, view] = (rows >= 0).sum() >= 2
                mixed[t, fish, view] = len(rowsets[-1]) > 1
            for pair, (a, b) in enumerate(PAIRS):
                same_node[t, view, pair] = (source[t, a, view] >= 0) & (
                    source[t, a, view] == source[t, b, view]
                )
                shared_rows[t, view, pair] = len(rowsets[a] & rowsets[b])
    require(not same_node.any(), "same current view/node row is used by two fish")
    return dict(
        segment_lengths_mm=lengths,
        body_polyline_length_mm=total,
        minimum_node_z_mm=pose[..., 2].min(-1),
        nodes_below_nominal_z0=pose[..., 2] < 0,
        nodes_below_engineering_zminus6=pose[..., 2] < -6,
        pectoral_pair_distance_mm=pectoral,
        pectoral_distance_over_mean_body_length=normalized,
        nearest_node_pair_distance_mm=np.asarray(node_distance),
        polyline_pair_distance_mm=np.asarray(segment_distance),
        same_view_node_row_shared=same_node,
        cross_node_shared_instance_rows=shared_rows,
        within_body_mixed_source_rows=mixed,
        source_grouping_testable=rows_testable,
    )


def bind_inputs():
    bindings = {
        str(Path(__file__).resolve()): sha256(__file__),
        str(HERE / "measure_scene0_body_proximity.py"): sha256(
            HERE / "measure_scene0_body_proximity.py"
        ),
    }
    seal = producer.verify_outputs(
        AUDIT, "audit.provenance.json", "independent_reflection_temporal_output_audit", bindings
    )
    producer.output_path(seal, "summary.json", AUDIT / "summary.json")
    checked = audit.read_json(AUDIT / "summary.json")
    require(
        checked["passed"] is True and checked["frame_range"] == [79236, 79436],
        "successful matching independent audit required",
    )
    for item in seal["inputs"].values():
        if Path(item["path"]).suffix == ".py":
            producer.bind(item["path"], item["sha256"], bindings)
    producer.bind(RUN / "experiment.provenance.json", checked["producer_seal_sha256"], bindings)
    experiment = audit.read_json(RUN / "experiment.provenance.json")
    require(
        experiment["stage"] == "reflection_temporal_comparison" and experiment["status"] == "ok",
        "producer experiment incomplete",
    )
    for arm in ARMS:
        path = producer.output_path(experiment, arm + "_candidate", RUN / f"{arm}_candidate.npz")
        producer.bind(path, checked["candidate_sha256"][arm], bindings)
    plan_path = producer.output_path(experiment, "plan", RUN / "plan.json")
    producer.bind(plan_path, experiment["outputs"]["plan"]["sha256"], bindings)
    plan = audit.read_json(plan_path)

    # Bind only the independent masks and calibration actually used below.
    def inference_file(path):
        path = Path(path).resolve()
        entry = experiment["inputs"][str(path)]
        require(Path(entry["path"]).resolve() == path, "inference input key/path mismatch")
        producer.bind(path, entry["sha256"], bindings)
        return path

    manifest_path = inference_file(Path(plan["sequence"]) / "manifest.json")
    manifest = audit.read_json(manifest_path)
    require(sha256(manifest_path) == plan["source_manifest_sha256"], "source manifest differs")
    mask_data = {}
    for view in VIEWS:
        path = inference_file(Path(plan["sequence"]) / manifest["masks"][view]["path"])
        require(sha256(path) == manifest["masks"][view]["sha256"], "detector mask file differs")
        mask_data[view] = audit.arrays(path)
        require(
            np.array_equal(mask_data[view]["frame_idx"], manifest["frames"]),
            "detector mask native frame index differs",
        )
    calibration = Path(plan["calibration_root"])
    for filename in (
        "calibration_manifest.json",
        *(f"cam{i}_bundle_refractive.json" for i in range(3)),
    ):
        inference_file(calibration / filename)
    rig = producer.load_rig(calibration)
    require(tuple(rig.views) == VIEWS, "calibration view order differs")
    return checked, bindings, plan, manifest, mask_data, rig


def mask_context(candidates, masks, rig, full_first_frame, selected):
    frames = candidates["direct"]["frame_idx"]
    projected = {
        arm: np.stack([rig.camera(view).project(data["pose_xyz"]) for view in VIEWS], axis=2)
        for arm, data in candidates.items()
    }
    outputs = {
        arm: dict(
            projected_native_pixels=projected[arm],
            mask_pectoral_shared_rows=np.zeros((len(frames), 3, 3), np.int64),
            mask_two_node_shared_rows=np.zeros((len(frames), 3, 3), np.int64),
            known_projected_nodes=np.zeros((len(frames), 3, 3, 3), bool),
        )
        for arm in ARMS
    }
    details = {arm: {} for arm in ARMS}
    for t, frame in enumerate(frames):
        for vi, view in enumerate(VIEWS):
            data, index = masks[view], int(frame - full_first_frame)
            decoded = producer.decode_frame_masks(data, index)
            require(
                producer.array_sha256(decoded) == data["frame_mask_sha256"][index].decode(),
                "decoded detector mask SHA differs",
            )
            available = bool(data["view_available"][index])
            for arm in ARMS:
                values, known = bilinear_membership(decoded, projected[arm][t, :, vi])
                known &= available
                outputs[arm]["known_projected_nodes"][t, :, vi] = known
                hit = (values >= 0.5) & known[None]
                chest, two = hit[:, :, 1], hit.sum(-1) >= 2
                for pair, (a, b) in enumerate(PAIRS):
                    outputs[arm]["mask_pectoral_shared_rows"][t, vi, pair] = np.sum(
                        chest[:, a] & chest[:, b]
                    )
                    outputs[arm]["mask_two_node_shared_rows"][t, vi, pair] = np.sum(
                        two[:, a] & two[:, b]
                    )
                if selected[t]:
                    details[arm].setdefault(str(frame), {})[view] = dict(
                        available=available,
                        instance_count=len(decoded),
                        pectoral_mask_rows=[
                            np.flatnonzero(chest[:, fish]).tolist() for fish in range(3)
                        ],
                        at_least_two_nodes_mask_rows=[
                            np.flatnonzero(two[:, fish]).tolist() for fish in range(3)
                        ],
                    )
    return outputs, details


def write_csv(path, rows, record):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    record.output(path.name, path, sha256=sha256(path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    require(output.is_relative_to(ROOT) and not output.exists(), "new workspace output required")
    checked, bindings, _plan, manifest, masks, rig = bind_inputs()
    output.mkdir(parents=True, exist_ok=False)
    with provenance(
        "posthoc_reflection_temporal_geometry_diagnosis", output, name="diagnosis"
    ) as record:
        started = time.perf_counter()
        for path, digest in bindings.items():
            record.input(path, path, sha256=digest)
        producer.snapshot({p: d for p, d in bindings.items() if p.endswith(".py")}, output, record)
        candidates = {arm: audit.arrays(RUN / f"{arm}_candidate.npz") for arm in ARMS}
        frames = candidates["direct"]["frame_idx"]
        audit.exact(frames, np.arange(79236, 79436, dtype=np.int64), "native 200 frames")
        audit.exact(frames, candidates["fresnel"]["frame_idx"], "shared native frames")
        selected = (
            (np.abs(frames - 79318) <= 16) | (np.abs(frames - 79372) <= 16) | (frames == 79405)
        )
        measures = {arm: geometry(data) for arm, data in candidates.items()}
        mask_values, mask_details = mask_context(
            candidates, masks, rig, int(manifest["frames"][0]), selected
        )
        pair_rows, fish_rows, selected_details, summaries = [], [], {}, {}
        for arm in ARMS:
            data, values = candidates[arm], measures[arm]
            values.update(mask_values[arm])
            audit.save_npz(
                output / f"{arm}_geometry.npz",
                record,
                frame_idx=frames,
                pairs=np.asarray(PAIRS, np.int64),
                **values,
            )
            pair_summary = {}
            for pair, (a, b) in enumerate(PAIRS):
                distance = values["pectoral_pair_distance_mm"][:, pair]
                ratio = values["pectoral_distance_over_mean_body_length"][:, pair]
                pair_summary[f"S{a}_S{b}"] = dict(
                    pectoral_mm=summarize_series(distance, frames),
                    pectoral_over_mean_body_length=summarize_series(ratio, frames),
                    closest_polylines_mm=summarize_series(
                        values["polyline_pair_distance_mm"][:, pair], frames
                    ),
                    chest_below_mm={
                        str(threshold): int((distance < threshold).sum()) for threshold in (2, 4, 8)
                    },
                    chest_below_4mm_ranges=audit.runs(distance < 4, frames),
                    same_view_node_row_duplicates=int(
                        values["same_view_node_row_shared"][:, :, pair].sum()
                    ),
                    cross_node_shared_SLEAP_row_events=int(
                        values["cross_node_shared_instance_rows"][:, :, pair].sum()
                    ),
                    D2_shared_pectoral_mask_events=int(
                        values["mask_pectoral_shared_rows"][:, :, pair].sum()
                    ),
                    D2_shared_two_node_mask_events=int(
                        values["mask_two_node_shared_rows"][:, :, pair].sum()
                    ),
                )
            summaries[arm] = dict(
                pairs=pair_summary,
                fish={
                    f"S{fish}": dict(
                        body_length_mm=summarize_series(
                            values["body_polyline_length_mm"][:, fish], frames
                        ),
                        mouth_pectoral_mm=summarize_series(
                            values["segment_lengths_mm"][:, fish, 0], frames
                        ),
                        pectoral_tail_mm=summarize_series(
                            values["segment_lengths_mm"][:, fish, 1], frames
                        ),
                        min_z_mm=float(values["minimum_node_z_mm"][:, fish].min()),
                        nodes_below_z0=int(values["nodes_below_nominal_z0"][:, fish].sum()),
                        nodes_below_zminus6=int(
                            values["nodes_below_engineering_zminus6"][:, fish].sum()
                        ),
                        SLEAP_mixed_row_body_views=int(
                            values["within_body_mixed_source_rows"][:, fish].sum()
                        ),
                        SLEAP_grouping_testable_body_views=int(
                            values["source_grouping_testable"][:, fish].sum()
                        ),
                    )
                    for fish in range(3)
                },
            )
            for t, frame in enumerate(frames):
                for pair, (a, b) in enumerate(PAIRS):
                    pair_rows.append(
                        dict(
                            arm=arm,
                            frame_idx=int(frame),
                            pair=f"S{a}_S{b}",
                            selected=bool(selected[t]),
                            pectoral_mm=float(values["pectoral_pair_distance_mm"][t, pair]),
                            normalized_pectoral=float(
                                values["pectoral_distance_over_mean_body_length"][t, pair]
                            ),
                            nearest_nodes_mm=float(
                                values["nearest_node_pair_distance_mm"][t, pair]
                            ),
                            polyline_mm=float(values["polyline_pair_distance_mm"][t, pair]),
                            direct_pectoral_a=bool(data["observed"][t, a, 1]),
                            direct_pectoral_b=bool(data["observed"][t, b, 1]),
                            **{
                                f"{view}_{name}": int(values[name][t, vi, pair])
                                for vi, view in enumerate(VIEWS)
                                for name in (
                                    "cross_node_shared_instance_rows",
                                    "mask_pectoral_shared_rows",
                                    "mask_two_node_shared_rows",
                                )
                            },
                        )
                    )
                for fish in range(3):
                    fish_rows.append(
                        dict(
                            arm=arm,
                            frame_idx=int(frame),
                            slot=fish,
                            selected=bool(selected[t]),
                            body_length_mm=float(values["body_polyline_length_mm"][t, fish]),
                            mouth_pectoral_mm=float(values["segment_lengths_mm"][t, fish, 0]),
                            pectoral_tail_mm=float(values["segment_lengths_mm"][t, fish, 1]),
                            min_z_mm=float(values["minimum_node_z_mm"][t, fish]),
                            observed_mouth=bool(data["observed"][t, fish, 0]),
                            observed_pectoral=bool(data["observed"][t, fish, 1]),
                            observed_tail=bool(data["observed"][t, fish, 2]),
                            identity_review=bool(data["identity_review"][t, fish]),
                            source_mixed_views=int(
                                values["within_body_mixed_source_rows"][t, fish].sum()
                            ),
                        )
                    )
                if selected[t]:
                    selected_details.setdefault(str(frame), {})[arm] = dict(
                        pose_xyz=data["pose_xyz"][t].tolist(),
                        observed=data["observed"][t].tolist(),
                        sources=data["sources"][t].tolist(),
                        identity_review=data["identity_review"][t].tolist(),
                        pectoral_pair_distance_mm=values["pectoral_pair_distance_mm"][t].tolist(),
                        body_length_mm=values["body_polyline_length_mm"][t].tolist(),
                        mask_rows=mask_details[arm][str(frame)],
                    )
        write_csv(output / "all_pair_frames.csv", pair_rows, record)
        write_csv(output / "all_fish_frames.csv", fish_rows, record)
        write_csv(
            output / "fixed_pair_frames.csv", [row for row in pair_rows if row["selected"]], record
        )
        producer.save_json(output / "fixed_frame_details.json", selected_details, record)
        result = dict(
            schema="reflection_temporal_geometry_diagnosis_v1",
            passed=True,
            frame_range=[79236, 79436],
            frames=200,
            arms=list(ARMS),
            pairs=PAIRS,
            candidate_sha256=checked["candidate_sha256"],
            qualification=checked["qualification"],
            audit_seal_sha256=sha256(AUDIT / "audit.provenance.json"),
            selected_ranges=[[79302, 79335], [79356, 79389], [79405, 79406]],
            selected_frames=frames[selected].tolist(),
            statistics=summaries,
            metrics=dict(
                distance="Euclidean millimeters; pectorals and two straight body segments",
                normalized_distance="pectoral distance / mean two-segment body length",
                source_events="distinct shared SLEAP rows per frame/view/pair; cross-node only",
                D2_events="distinct shared D2 rows per frame/view/pair; latent projections",
                D2_membership="bilinear native mask >=0.5; outside/missing unknown",
                threshold_scope="2/4/8mm are post-hoc descriptions, not gates or collision tests",
                nominal_z0="nominal box plane; no surveyed optical-floor claim",
            ),
            no_same_node_source_reuse=True,
            historical_coordinates_loaded=False,
            GT=False,
            raw_decoding=False,
            refit=False,
            rerender=False,
            GPU=False,
            identity_claim="Proximity/shared masks are hypotheses, not biological identity truth",
            elapsed_seconds=time.perf_counter() - started,
        )
        producer.save_json(output / "summary.json", result, record)
        lines = [
            "# 反射モデルの3D接近・観測共有診断",
            "",
            "全200フレームのDirect/Fresnel保存結果だけを比較した。近い姿勢や同じD2領域への投影は、同じ実魚を二重に説明している可能性の指標であり、人手なしに確定しない。",
            "",
            "| 方式 | 組 | 胸距離の最小 mm (frame) | 胸距離中央値 mm | "
            "4mm未満 fr | 体軸の最小距離 mm |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for arm in ARMS:
            for pair, value in summaries[arm]["pairs"].items():
                d = value["pectoral_mm"]
                lines.append(
                    f"| {arm} | {pair} | {d['minimum']:.6f} ({d['minimum_frame']}) | "
                    f"{d['median']:.3f} | {value['chest_below_mm']['4']} | "
                    f"{value['closest_polylines_mm']['minimum']:.6f} |"
                )
        lines += [
            "",
            "全方式で同一view・同一nodeのSLEAP行の二重使用は0。"
            "別nodeが同じ2D instance行を共有する状態と、"
            "排他的な行を使いながら3D姿勢が近づく状態は別に数えた。",
            "",
            "## 固定時点の胸部距離と支持",
            "",
            "| frame | 方式 | S0-S1 mm | S0-S2 mm | S1-S2 mm | 胸部直接支持 S0/S1/S2 |",
            "|---|---|---:|---:|---:|---|",
        ]
        for frame in (79302, 79318, 79334, 79356, 79372, 79388, 79405):
            for arm in ARMS:
                row = selected_details[str(frame)][arm]
                distances = row["pectoral_pair_distance_mm"]
                observed = [int(value[1]) for value in row["observed"]]
                lines.append(
                    f"| {frame} | {arm} | {distances[0]:.3f} | {distances[1]:.3f} | "
                    f"{distances[2]:.3f} | {observed} |"
                )
        lines += [
            "",
            "固定範囲は最初の胸支持喪失79318の前後16frame、79372の前後16frame、79405。全200の数値はCSV/NPZ、該当範囲の全pose/source/D2行はJSONへ保存した。",
            "",
            "床zは名目0と既存工学下限-6mmを区別する。体長は3nodesの折れ線長で実際の体表長ではない。D2にはmerged/split/反射があり、共有する1maskを1魚と断定できない。",
            "",
            "最終backtrace上の差はオンラインの因果分岐時刻を証明しない。検出行の排他が守られていても匿名slotの生物学的同一性は保証されない。今回の結果で反射重み、床位置、受理条件は変更しない。",
        ]
        path = output / "RESULT.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        record.output(path.name, path, sha256=sha256(path))
        producer.guard(bindings, {}, record)
    print(json.dumps(dict(status="ok", output=str(output))))


if __name__ == "__main__":
    main()
