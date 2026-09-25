"""Bounded CPU synthetic timing/parity report; no data, checkpoint, tracker or GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import instance_body_score as dense
import numpy as np
import scipy
import sparse_instance_body_score as sparse


def synthetic_frame(semiaxes, candidates):
    """Three 1200² camera slots, three ellipse masks each; declared synthetic sizes."""
    yy, xx = np.ogrid[:1200, :1200]
    scenes = []
    for view in range(3):
        masks, rows = [], []
        a, b = semiaxes
        for fish, (cx, cy) in enumerate(((230, 310), (600, 610), (960, 890))):
            theta = (view - 1) * 0.18 + fish * 0.14
            c, s = np.cos(theta), np.sin(theta)
            u, v = c * (xx - cx) + s * (yy - cy), -s * (xx - cx) + c * (yy - cy)
            masks.append((u / a) ** 2 + (v / b) ** 2 <= 1)
            offset = np.array([0.8 * a, 0.0, -0.8 * a])
            rows.append(np.stack([cx + c * offset, cy + s * offset], axis=-1))
        masks, rows = np.array(masks), np.array(rows)
        pose = np.concatenate([rows, np.ones((3, 3, 1))], axis=-1)
        bank = np.repeat(pose[None], candidates, axis=0)
        rng = np.random.default_rng(671 + view + candidates)
        bank[1, 2, :2] = bank[0, 0, :2]
        bank[2, 2] = bank[0, 0]
        bank[3, 0, 0] = np.nan
        bank[4:] += rng.normal(0, 18, size=bank[4:].shape)
        scenes.append((masks, rows, bank))
    return scenes


def execute(scenes, backend):
    prepare, score = (
        (dense.prepare_instance_observation, dense.score_instance_bodies)
        if backend == "dense"
        else (sparse.prepare_sparse_instance_observation, sparse.score_sparse_instance_bodies)
    )
    preparation, scoring, payload_bytes, results, observations = 0.0, 0.0, 0, [], []
    for masks, rows, bank in scenes:
        started = perf_counter()
        observation = prepare(masks, rows)
        preparation += perf_counter() - started
        started = perf_counter()
        result = score(bank, lambda x: x[..., :2], observation)
        scoring += perf_counter() - started
        if backend == "dense":
            payload_bytes += observation.distance_fields.nbytes
        else:
            fields = observation.sparse_fields
            payload_bytes += sum(x.nbytes for x in fields.centers_yx)
            payload_bytes += sum(t.data.nbytes + t.indices.nbytes for t in fields.trees)
        results.append(result)
        observations.append(
            dict(
                eligible=observation.eligible,
                reason=observation.reason,
                evidence=observation.evidence,
                rows=observation.sleap_row_by_instance.tolist(),
            )
        )
    return dict(
        preparation_seconds=preparation,
        scoring_seconds=scoring,
        total_seconds=preparation + scoring,
        known_distance_array_bytes=payload_bytes,
        results=results,
        observations=observations,
    )


def compare_results(a, b):
    detail = dict(
        all_arrays_byte_equal=True,
        all_gate_and_scalar_fields_equal=True,
        compared_arrays=0,
        compared_array_elements=0,
        differing_array_elements=0,
        max_abs_finite_difference=0.0,
        nonfinite_or_discrete_mismatches=0,
    )
    detail["all_gate_and_scalar_fields_equal"] = a["observations"] == b["observations"]
    for da, db in zip(a["results"], b["results"], strict=True):
        if set(da) != set(db):
            raise ValueError("score result membership changed")
        for key, x in da.items():
            y = db[key]
            if not isinstance(x, np.ndarray):
                detail["all_gate_and_scalar_fields_equal"] &= x == y
                continue
            if x.shape != y.shape or x.dtype != y.dtype:
                raise ValueError("score array shape or dtype changed: " + key)
            detail["compared_arrays"] += 1
            detail["compared_array_elements"] += x.size
            detail["all_arrays_byte_equal"] &= x.tobytes() == y.tobytes()
            equal = x == y
            if x.dtype.kind == "f":
                equal |= np.isnan(x) & np.isnan(y)
                finite = np.isfinite(x) & np.isfinite(y)
                if finite.any():
                    detail["max_abs_finite_difference"] = max(
                        detail["max_abs_finite_difference"],
                        float(np.max(np.abs(x[finite] - y[finite]))),
                    )
                detail["nonfinite_or_discrete_mismatches"] += int(
                    np.count_nonzero(~equal & ~finite)
                )
            else:
                detail["nonfinite_or_discrete_mismatches"] += int(np.count_nonzero(~equal))
            detail["differing_array_elements"] += int(np.count_nonzero(~equal))
    return detail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh output path required")
    report = dict(
        schema="sparse_instance_synthetic_cpu_benchmark_v1",
        real_data_used=False,
        tracker_run=False,
        GPU_used=False,
        live_integration=False,
        image_shape_hw=[1200, 1200],
        views=3,
        instances_per_view=3,
        candidates=[14, 16],
        repeats=3,
        tree_query_workers=1,
        benchmark_is_not_isolated_from_other_machine_workloads=True,
        known_distance_array_bytes_exclude_tree_nodes_python_objects_and_temporaries=True,
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        sources={
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), Path(dense.__file__), Path(sparse.__file__))
        },
        scenarios=[],
    )
    for semiaxes in ((50, 8), (80, 14), (120, 24)):
        for candidates in (14, 16):
            scenes = synthetic_frame(semiaxes, candidates)
            # One small pair of warmups per scenario, using the same declared synthetic frame.
            execute(scenes[:1], "dense")
            execute(scenes[:1], "sparse")
            repetitions = []
            for repeat in range(3):
                values = {}
                for backend in ("dense", "sparse") if repeat % 2 == 0 else ("sparse", "dense"):
                    values[backend] = execute(scenes, backend)
                parity = compare_results(values["dense"], values["sparse"])
                repetitions.append(
                    dict(
                        parity=parity,
                        **{
                            backend: {
                                k: v
                                for k, v in result.items()
                                if k not in ("results", "observations")
                            }
                            for backend, result in values.items()
                        },
                    )
                )
            medians = {
                backend: {
                    key: float(np.median([r[backend][key] for r in repetitions]))
                    for key in ("preparation_seconds", "scoring_seconds", "total_seconds")
                }
                for backend in ("dense", "sparse")
            }
            report["scenarios"].append(
                dict(
                    ellipse_semiaxes_px=list(semiaxes),
                    candidates=candidates,
                    foreground_pixels_by_view=[m.sum(axis=(1, 2)).tolist() for m, _, _ in scenes],
                    repetitions=repetitions,
                    medians=medians,
                    total_speedup_ratio=medians["dense"]["total_seconds"]
                    / medians["sparse"]["total_seconds"],
                    preparation_speedup_ratio=medians["dense"]["preparation_seconds"]
                    / medians["sparse"]["preparation_seconds"],
                )
            )
    report["all_comparisons_bit_exact"] = all(
        r["parity"]["all_arrays_byte_equal"] and r["parity"]["all_gate_and_scalar_fields_equal"]
        for scenario in report["scenarios"]
        for r in scenario["repetitions"]
    )
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            dict(
                output=str(args.output),
                all_comparisons_bit_exact=report["all_comparisons_bit_exact"],
                scenarios=[
                    {
                        k: s[k]
                        for k in (
                            "ellipse_semiaxes_px",
                            "candidates",
                            "medians",
                            "total_speedup_ratio",
                        )
                    }
                    for s in report["scenarios"]
                ],
            ),
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
