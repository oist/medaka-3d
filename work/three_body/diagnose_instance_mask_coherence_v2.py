"""Fixed six-frame, two-method body/individual-mask geometry diagnostic.

No video decoding, fit, tracking, GPU or identity assignment. Distances refer to
native pixel centres, not a continuous or surveyed fish silhouette. Missing
observations/invalid projections are NaN; absent opposite pixel sets are infinity.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "labs/20260909_three_body")]

import numpy as np
import scipy
import summarize_chest_candidate_replay as checks
from continuous_tracker import _canonical
from scipy.ndimage import distance_transform_edt

from medaka3d.calib import load_rig
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS

WORK = REPO / "work/three_body"
RAW = WORK / "bounded_shape_raw_review_v1"
AUDIT = WORK / "bounded_mouth_first_chunk_interim_audit_v1"
RAW_SHA = "d5fa256848a0298fc0d5f5cbca2e7b7a0d55867b5874f9159d5859c56ddcbd70"
AUDIT_SHA = "08a7440cb5c851c4a97d98d1d67f665f6762a89c524c2ba87bfa7c0245656f04"
STAGE = "bounded_instance_mask_coherence_6_v2"
FRAMES = (66524, 66600, 66628, 66703, 66800, 66908)
METHODS = ("directional", "bounded")
PARTS = {"body": np.arange(33), "mouth_chest": np.arange(17), "chest_tail": np.arange(16, 33)}


def encode_nonfinite(value):
    """JSON distinguishes IEEE missing NaN from empty-set infinities explicitly."""
    if isinstance(value, np.ndarray):
        return encode_nonfinite(value.tolist())
    if isinstance(value, np.generic):
        return encode_nonfinite(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return "NaN" if np.isnan(value) else ("+inf" if value > 0 else "-inf")
    if isinstance(value, dict):
        return {str(k): encode_nonfinite(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode_nonfinite(v) for v in value]
    return value


def project_body(pose_xyz, camera):
    pose = np.asarray(pose_xyz)
    checks.require(pose.shape == (3, 3) and pose.dtype.kind == "f", "one float three-node pose")
    t = np.linspace(0.0, 1.0, 17)[:, None]
    first = (1 - t) * pose[0] + t * pose[1]
    second = (1 - t) * pose[1] + t * pose[2]
    points = np.concatenate((first, second[1:]))
    # Preserve saved endpoint bits, including any invalid endpoint values.
    points[[0, 16, 32]] = pose
    uv = np.asarray(camera.project(points.copy()))
    checks.require(uv.shape == (33, 2) and uv.dtype.kind == "f", "projection shape/dtype")
    return dict(points_xyz=points, projected_uv=uv.copy())


def sample_mask_fields(masks, available, uv):
    masks, uv = np.asarray(masks), np.asarray(uv)
    checks.require(
        masks.dtype == bool and masks.ndim == 3 and min(masks.shape[1:]) > 0,
        "boolean N,H,W masks required",
    )
    checks.require(type(available) is bool and (available or len(masks) == 0), "availability")
    checks.require(uv.ndim == 2 and uv.shape[1] == 2 and uv.dtype.kind == "f", "float K,2 uv")
    height, width = masks.shape[1:]
    finite = np.isfinite(uv).all(-1)
    rounded = np.full(uv.shape, np.nan)
    rounded[finite] = np.rint(uv[finite])
    valid = finite & (rounded[:, 0] >= 0) & (rounded[:, 0] < width)
    valid &= (rounded[:, 1] >= 0) & (rounded[:, 1] < height)
    xy = np.full(uv.shape, -1, np.int64)
    xy[valid] = rounded[valid].astype(np.int64)
    shape = (len(masks) + 1, len(uv))
    membership = np.full(shape, -1, np.int8)
    outside, margin = np.full(shape, np.nan), np.full(shape, np.nan)
    status = np.full(len(masks) + 1, "missing", dtype="U7")
    if available:
        union = np.any(masks, axis=0)
        for row, mask in enumerate((*masks, union)):
            inside = mask[xy[valid, 1], xy[valid, 0]]
            membership[row, valid] = inside.astype(np.int8)
            if not mask.any():
                status[row] = "empty"
                outside[row, valid], margin[row, valid] = np.inf, 0.0
            elif mask.all():
                status[row] = "full"
                outside[row, valid], margin[row, valid] = 0.0, np.inf
            else:
                status[row] = "mixed"
                outside[row, valid] = distance_transform_edt(~mask)[xy[valid, 1], xy[valid, 0]]
                margin[row, valid] = distance_transform_edt(mask)[xy[valid, 1], xy[valid, 0]]
    return dict(
        valid=valid,
        finite=finite,
        pixel_xy=xy,
        membership=membership,
        outside_distance=outside,
        inside_margin=margin,
        signed_distance=outside - margin,
        mask_status=status,
    )


def coherence_metrics(fields, indices):
    indices = np.asarray(indices)
    checks.require(
        indices.ndim == 1
        and indices.dtype.kind in "iu"
        and len(indices) > 0
        and np.all(indices < len(fields["valid"]))
        and np.all(indices >= 0)
        and len(np.unique(indices)) == len(indices),
        "unique in-range point indices",
    )
    valid = indices[fields["valid"][indices]]
    count = len(valid)
    n = len(fields["mask_status"]) - 1
    missing = fields["mask_status"][-1] == "missing"

    def metrics(row):
        if not count or missing:
            return dict(
                inside_count=None,
                inside_fraction=np.nan,
                mean_outside_distance=np.nan,
                max_outside_distance=np.nan,
            )
        member = fields["membership"][row, valid]
        distances = fields["outside_distance"][row, valid]
        return dict(
            inside_count=int((member == 1).sum()),
            inside_fraction=float((member == 1).mean()),
            mean_outside_distance=float(distances.mean()),
            max_outside_distance=float(distances.max()),
        )

    rows, union = [metrics(row) for row in range(n)], metrics(n)
    means = np.array([r["mean_outside_distance"] for r in rows])
    single = float(means.min()) if n and count and not missing else np.nan
    union_cost = union["mean_outside_distance"]
    ties = np.flatnonzero(means == single).tolist() if not np.isnan(single) else []
    delta = single - union_cost if np.isfinite(single) and np.isfinite(union_cost) else np.nan
    if np.isfinite(delta):
        checks.require(delta >= -1e-12, "individual/union same-point inequality violated")
    status = (
        "missing"
        if missing
        else "no_valid_points"
        if not count
        else "no_instances"
        if not n
        else "empty_union"
        if not np.isfinite(union_cost)
        else "finite"
    )
    return dict(
        requested_points=len(indices),
        valid_points=count,
        projection_status="complete" if count == len(indices) else "partial" if count else "none",
        observation_status="missing"
        if missing
        else "available_no_instances"
        if not n
        else "available",
        per_mask=rows,
        union=union,
        C_single=single,
        C_union=union_cost,
        delta=delta,
        best_mask_rows=ties,
        coherence_status=status,
    )


def bind(path, digest, hashes):
    path = Path(path).resolve(strict=True)
    checks.require(str(path) not in hashes or hashes[str(path)] == digest, "conflicting SHA")
    checks.require(checks.sha(path) == digest, "bound input changed: " + str(path))
    hashes[str(path)] = digest
    return path


def input_member(seal, path, hashes):
    path = Path(path).resolve(strict=True)
    item = seal["inputs"].get(str(path))
    checks.require(item is not None and Path(item["path"]).resolve() == path, "unbound input")
    return bind(path, item["sha256"], hashes)


def verified(root, filename, stage, expected, hashes):
    bind(root / filename, expected, hashes)
    seal = checks.verify(root, filename, stage, hashes)
    checks.require(hashes[str(root / filename)] == expected, "seal changed during verification")
    return seal


def source_code():
    paths = {Path(__file__).resolve()}
    scopes = (WORK, REPO / "src", REPO / "labs/20260909_three_body")
    for module in tuple(sys.modules.values()):
        file = getattr(module, "__file__", None)
        if file:
            path = Path(file).resolve()
            if path.suffix == ".py" and any(path.is_relative_to(s) for s in scopes):
                paths.add(path)
    return {str(p): checks.sha(p) for p in paths}


def validate_rig_context(context, rig):
    checks.require(
        context["cameras"] == json.loads(_canonical([dataclasses.asdict(c) for c in rig.cameras]))
        and context["views"] == list(rig.views)
        and context["world_units"] == rig.world_units == "mm",
        "actual rig context differs",
    )


def load_inputs(hashes, code):
    # Preserve the failed revision and its unsuccessful, pre-measurement result.
    bind(
        WORK / "diagnose_instance_mask_coherence.py",
        "210577a99b49b7b21894e1d947e5bd96c4b30346bed57fa1a8225a9a305103e6",
        hashes,
    )
    bind(
        WORK / "bounded_instance_mask_coherence_6_v1/diagnosis.provenance.json",
        "fe8c328e2d60925adebeb0518ed439f1a02b73f64516a19d9eb9fff856d2a789",
        hashes,
    )
    code[str(WORK / "diagnose_instance_mask_coherence.py")] = hashes[
        str(WORK / "diagnose_instance_mask_coherence.py")
    ]
    raw = verified(RAW, "review.provenance.json", "bounded_shape_raw_review_v1", RAW_SHA, hashes)
    input_member(raw, AUDIT / "audit.provenance.json", hashes)
    audit = verified(
        AUDIT,
        "audit.provenance.json",
        "bounded_mouth_first_chunk_interim_audit_v1",
        AUDIT_SHA,
        hashes,
    )
    for root in (RAW, AUDIT):
        checks.require(str(root / "summary.json") in hashes, "unsealed summary")
    raw_summary, summary = checks.read(RAW / "summary.json"), checks.read(AUDIT / "summary.json")
    checks.require(
        raw_summary["status"] == summary["status"] == "ok"
        and raw_summary["frames"] == list(FRAMES)
        and raw_summary["horizon_stop"] == summary["horizon_stop"] == 68060
        and raw_summary["native_BGR_and_mask_SHA_checks"] == 18,
        "fixed six-frame same-horizon input differs",
    )
    producer = WORK / "render_bounded_shape_raw_review.py"
    input_member(raw, producer, hashes)
    code[str(producer)] = hashes[str(producer)]
    for path, digest in code.items():
        for seal in (raw, audit):
            if path in seal["inputs"]:
                checks.require(
                    seal["inputs"][path]["sha256"] == digest, "historical/current code differs"
                )
    for path in (REPO / "src/medaka3d/calib.py", REPO / "src/medaka3d/types.py"):
        input_member(raw, path, hashes)
        input_member(audit, path, hashes)
    shared = summary["shared_input_binding"]["inputs"]["seed"]["SL_and_calibration_sha256"]
    checks.require(
        len(shared) == 7
        and sum(Path(p).suffix == ".npz" for p in shared) == 3
        and sum(Path(p).suffix == ".json" for p in shared) == 4,
        "same SL3/calibration4 required",
    )
    for path, digest in shared.items():
        input_member(raw, path, hashes)
        input_member(audit, path, hashes)
        checks.require(hashes[str(Path(path).resolve())] == digest, "registered geometry differs")
    calibration = next(
        Path(p).parent for p in shared if Path(p).name == "calibration_manifest.json"
    )
    manifest = checks.read(calibration / "calibration_manifest.json")
    calibration_files = {str((calibration / x["path"]).resolve()) for x in manifest["model_files"]}
    calibration_files.add(str(calibration / "calibration_manifest.json"))
    checks.require(
        calibration_files == {p for p in shared if Path(p).suffix == ".json"},
        "rig loader file set differs from registered calibration4",
    )
    rig = load_rig(calibration)
    traces = {}
    for name in METHODS:
        path = AUDIT / f"{name}_same68060_candidate.npz"
        input_member(raw, path, hashes)
        checks.require(str(path) in hashes, "unsealed candidate")
        with np.load(path, allow_pickle=False) as z:
            checks.require(
                len(z.files) == 21 and np.array_equal(z["frame_idx"], np.arange(66400, 68060)),
                "same1660 trace required",
            )
            traces[name] = z["pose_xyz"][np.array(FRAMES) - 66400].copy()
        run_name = (
            "directional_one_minute_66400-76360_v1"
            if name == "directional"
            else "bounded_mouth_one_minute_66400-76360_v1"
        )
        meta_path = input_member(
            audit, WORK / run_name / "checkpoints/through_068060/metadata.json", hashes
        )
        context = checks.read(meta_path)["context"]
        validate_rig_context(context, rig)
    return traces, rig


def plan():
    return dict(
        schema=STAGE,
        frames=FRAMES,
        methods=METHODS,
        views=VIEWS,
        body_view_records=108,
        native_shape=[1200, 1200],
        horizon_stop=68060,
        samples_per_segment=17,
        total_body_points=33,
        endpoint_indices=[0, 16, 32],
        parts={k: v.tolist() for k, v in PARTS.items()},
        sample_rule="3D linear segments, fixed t=0..1/16..1; then refractive camera.project",
        pixel_rule="np.rint ties-to-even; rounded in-image pixel centres only",
        distance_rule="exact native pixel-centre EDT, no outside-image padding/erosion/dilation",
        opposite_empty_set="empty: signed +inf; full: signed -inf; invalid/missing: NaN",
        aggregation="same valid points for every mask and union; partial status is retained",
        comparison="C_single=min_j mean outside_j; C_union=mean outside_union; all exact ties",
        case_selection="post-observation fixed six diagnostic frames; not held-out evaluation",
        masks_are_anonymous=True,
        ID_assignment=False,
        new_threshold=False,
        raw_decode=False,
        fit=False,
        GPU=False,
        adopted=False,
    )


def schema():
    return dict(
        view_npz="18 files; point axis order method(2),slot(3),bodypoint(33), flattened C-order",
        fields={
            "points_xyz": "(2,3,33,3)float64",
            "projected_uv": "(2,3,33,2)float64",
            "membership": "(D2N+1,198)int8: -1 unknown, 0 outside, 1 inside; union last",
            "outside_distance/inside_margin/signed_distance": "(D2N+1,198)float64 native pixels",
            "finite/valid": "(198,)bool",
            "pixel_xy": "(198,2)int64; invalid -1",
            "mask_status": "(D2N+1,)Unicode: missing/empty/full/mixed",
        },
        json=(
            "108 body-view records, each body/mouth_chest/chest_tail metrics; "
            "IEEE encoded as strings"
        ),
        ties="all exact equal minimum rows; local D2 row only; no temporal/cross-view identity",
        invalid="projection requested/valid count and complete/partial/none always retained",
    )


def aggregate_records(rows):
    """All rows accounted for; means explicitly limited to complete finite cases."""
    groups = []
    for method in METHODS:
        for view in VIEWS:
            selected = [r for r in rows if r["method"] == method and r["view"] == view]
            for part in PARTS:
                values = [r["metrics"][part] for r in selected]
                eligible = [
                    m
                    for m in values
                    if m["projection_status"] == "complete" and m["coherence_status"] == "finite"
                ]
                groups.append(
                    dict(
                        method=method,
                        view=view,
                        part=part,
                        requested_body_cases=len(values),
                        finite_complete_cases=len(eligible),
                        other_cases=len(values) - len(eligible),
                        projection_status=dict(Counter(m["projection_status"] for m in values)),
                        coherence_status=dict(Counter(m["coherence_status"] for m in values)),
                        mean_over_finite_complete={
                            key: float(np.mean([m[key] for m in eligible])) if eligible else np.nan
                            for key in ("C_single", "C_union", "delta")
                        },
                    )
                )
    return groups


def run(output):
    output = Path(output).resolve()
    checks.require(output.is_relative_to(WORK) and not output.exists(), "new work output required")
    code = source_code()
    hashes, outputs = dict(code), {}
    output.mkdir(parents=True)
    with provenance(STAGE, output, name="diagnosis") as record:

        def emit(name, value=None):
            path = output / name
            if value is not None:
                path.write_text(
                    json.dumps(encode_nonfinite(value), indent=2, allow_nan=False), encoding="utf-8"
                )
            digest = checks.sha(path)
            outputs[str(path)] = digest
            record.output(name, path, sha256=digest)

        emit("plan.json", plan())
        emit("schema.json", schema())
        traces, rig = load_inputs(hashes, code)
        rows = []
        for fi, native in enumerate(FRAMES):
            for view in VIEWS:
                path = RAW / f"observation_{native}_{view}.npz"
                checks.require(str(path) in hashes, "unsealed observation")
                with np.load(path, allow_pickle=False) as z:
                    masks, available = z["masks"].copy(), z["mask_available"].copy()
                    checks.require(
                        masks.shape[1:] == (1200, 1200)
                        and available.shape == ()
                        and available.dtype == bool,
                        "full native mask schema",
                    )
                    projected, points = [], []
                    for name in METHODS:
                        endpoint = rig.camera(view).project(traces[name][fi])
                        checks.require(
                            checks.equal_bytes(endpoint, z[name + "_projection"]),
                            "saved endpoint projection differs",
                        )
                        bodies = [project_body(pose, rig.camera(view)) for pose in traces[name][fi]]
                        points.append([b["points_xyz"] for b in bodies])
                        projected.append([b["projected_uv"] for b in bodies])
                points, projected = np.asarray(points), np.asarray(projected)
                fields = sample_mask_fields(masks, bool(available), projected.reshape(-1, 2))
                filename = f"coherence_{native}_{view}.npz"
                np.savez_compressed(
                    output / filename,
                    frame_idx=np.asarray(native, np.int64),
                    methods=np.array(METHODS),
                    points_xyz=points,
                    projected_uv=projected,
                    mask_available=available,
                    **fields,
                )
                emit(filename)
                for mi, name in enumerate(METHODS):
                    for slot in range(3):
                        offset = (mi * 3 + slot) * 33
                        rows.append(
                            dict(
                                frame_idx=native,
                                method=name,
                                slot=slot,
                                view=view,
                                mask_count=len(masks),
                                output=filename,
                                metrics={
                                    part: coherence_metrics(fields, ids + offset)
                                    for part, ids in PARTS.items()
                                },
                            )
                        )
        checks.require(len(rows) == 108, "fixed108 records required")
        emit("records.json", rows)
        emit(
            "summary.json",
            dict(
                status="ok",
                schema=STAGE,
                frames=FRAMES,
                methods=METHODS,
                body_view_records=len(rows),
                view_files=18,
                horizon_stop=68060,
                projection_status=dict(
                    Counter(r["metrics"]["body"]["projection_status"] for r in rows)
                ),
                coherence_status=dict(
                    Counter(r["metrics"]["body"]["coherence_status"] for r in rows)
                ),
                by_method_view_part=aggregate_records(rows),
                runtime=dict(
                    python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__
                ),
                raw_video_decodes=0,
                GPU=False,
                fits=0,
                independent_ID_truth=False,
                previous_failed_revision=dict(
                    path=str(WORK / "bounded_instance_mask_coherence_6_v1"),
                    failure="before measurement: camera lists differed from saved array signatures",
                    repair="frozen continuous_tracker._canonical dtype/shape/SHA representation",
                    original_source_preserved=True,
                ),
                new_threshold=False,
                adopted=False,
            ),
        )
        (output / "RESULT.md").write_text(
            "# 個別マスクと身体区間の固定6時刻診断\n\n"
            "同68060終端の2方式・6時刻・3匿名slot・3視点、全108件を保存した。"
            "全nativeマスクへ3D口胸尾の固定33点を屈折投影し、個別とunionの距離を同じ有効点で比較する。"
            "全件と両部位の値、最良mask全tie、欠測・空集合・投影不正はrecords.json/NPZに残す。\n\n"
            "unionだけ良好なら複数maskをまたぐ形状の候補だが、検出分裂でも生じる。"
            "融合・反射・重複・欠落、細尾未検出、2直線の身体近似、画素丸めの影響を分離できない。"
            "同一mask内の誤った口位置・二体混成・向き・個体IDは判定しない。"
            "失敗観察後の固定6枚であり独立精度評価ではない。閾値や追跡目的へ採用していない。\n\n"
            "v1は計測前に校正contextの表現照合で失敗した。v2は元trackerの配列署名形式を使用し、"
            "距離・投影・集計の式は不変。v1失敗記録と元ソースを保持して入力SHAへ結合した。\n",
            encoding="utf-8",
        )
        emit("RESULT.md")
        for path, digest in code.items():
            dest = output / "code_snapshot" / Path(path).relative_to(REPO)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(Path(path).read_bytes())
            checks.require(checks.sha(dest) == digest, "source snapshot changed")
            emit(str(dest.relative_to(output)))
        for path, digest in hashes.items():
            checks.require(checks.sha(path) == digest, "input/source changed")
            record.input(path, path, sha256=digest)
        for path, digest in outputs.items():
            checks.require(checks.sha(path) == digest, "output changed")
    print(json.dumps(dict(status="ok", body_view_records=108)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
