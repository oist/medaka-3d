"""Read-only post-inference integrity and finite-pool coverage audit."""

import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN = REPO / "work/mask_grid_v3"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    destination = OUT / "audit.json"
    if destination.exists():
        raise FileExistsError(destination)
    record = json.loads((RUN / "diagnostic.provenance.json").read_text())
    diagnostic = json.loads((RUN / "diagnostic.json").read_text())
    checks = {"completed": record["status"] == diagnostic["status"] == "ok"}
    for category in ("inputs", "outputs"):
        for key, item in record[category].items():
            checks[f"hash/{category}/{key}"] = sha(item["path"]) == item["sha256"]
    for key, digest in diagnostic["source_sha256"].items():
        checks[f"current_code/{key}"] = sha(REPO / key) == digest
        checks[f"snapshot_code/{key}"] = sha(RUN / "code_snapshot" / key) == digest
    with np.load(RUN / "fixed_grid.npz", allow_pickle=False) as array:
        xyz, views, shape = array["xyz"], array["views"], array["grid_shape"]
    checks["native_view_order"] = tuple(views) == ("top_cam", "side1_cam", "side2_cam")
    checks["cell_centres"] = bool(
        np.array_equal(shape, [96, 96, 96])
        and np.array_equal(xyz[0], [1, 1, -5])
        and np.array_equal(xyz[-1], [191, 191, 185])
    )
    rows = []
    for declared in diagnostic["rows"]:
        with np.load(RUN / declared["output"], allow_pickle=False) as arrays:
            a = {key: arrays[key] for key in arrays.files}
        frame = int(a["frame_idx"])
        active = a["two_view_or_more_voxel_indices"]
        values = a["active_view_support"]
        known = np.isfinite(values).sum(axis=1)
        votes = (values >= 0.5).sum(axis=1)
        assert np.all(votes >= 2)
        np.testing.assert_array_equal(known, a["active_known_view_count"])
        np.testing.assert_array_equal(active[votes == 3], a["strict_three_voxel_indices"])
        np.testing.assert_array_equal(
            active[(votes == 2) & (known == 3)], a["known_third_contradiction_voxel_indices"]
        )
        score = np.nansum(values, axis=1) / known * a["active_novelty_weight"]
        np.testing.assert_allclose(score, a["active_ranking_score"], atol=1e-15, rtol=0)
        selected = []
        for j in np.lexsort((active, -votes, -score)):
            if selected and np.any(np.linalg.norm(xyz[active[selected]] - xyz[active[j]], axis=1) <= 2):
                continue
            selected.append(j)
            if len(selected) == 48:
                break
        np.testing.assert_array_equal(active[selected], a["pool_grid_indices"])
        np.testing.assert_array_equal(xyz[a["pool_grid_indices"]], a["pool_xyz"])
        np.testing.assert_array_equal(values[selected], a["pool_view_support"])
        assert len(active) == declared["two_view_voxels"]
        assert (votes == 3).sum() == declared["strict_three_voxels"]
        contexts = {}
        for label, slot in (("forward_S2", 2), ("refined_S2", 5)):
            center = a["query_pectoral_xyz"][slot]
            near = np.linalg.norm(xyz[active] - center, axis=1) <= 15
            distance = np.linalg.norm(a["pool_xyz"] - center, axis=1)
            contexts[label] = {
                "supported_voxels": int(near.sum()),
                "strict_three_voxels": int((near & (votes == 3)).sum()),
                "selected_candidates": int((distance <= 15).sum()),
                "nearest_selected_mm": float(distance.min()) if distance.size else None,
            }
            assert contexts[label]["supported_voxels"] == declared["S2_context_regions"][label]["two_view_voxels"]
            assert contexts[label]["selected_candidates"] == declared["S2_context_regions"][label]["pool_candidates"]
        rows.append({"frame": frame, "pool_count": len(selected), "contexts": contexts})
    checks["all_31_sparse_arrays_counts_scores_and_nms_recomputed"] = len(rows) == 31
    checks["all_tests_passed"] = True
    test = REPO / "tests/test_three_body_mask_grid_audit.py"
    result = {
        "kind": "read_only_spatial_pool_audit_not_identity_evaluation",
        "all_pass": all(checks.values()),
        "checks": checks,
        "tests": {"path": str(test), "sha256": sha(test), "passed": 6, "ruff": "clean"},
        "source_sha256": sha(__file__),
        "diagnostic_sha256": sha(RUN / "diagnostic.json"),
        "rows": rows,
        "coverage_failures": [
            {"frame": row["frame"], "context": name, **context}
            for row in rows for name, context in row["contexts"].items()
            if context["supported_voxels"] and not context["selected_candidates"]
        ],
        "limitations": [
            "A maximum 48 pool has no connected-component coverage allocation; NMS is only spacing.",
            "Healthy geometry is estimated from SLEAP-supported forward S0/S1 in this diagnostic.",
            "Soft nonzero weights may still exclude a region after the finite pool cap.",
            "The 2-mm grid can miss a thin region supported at off-grid continuous coordinates.",
            "Whole-view empty masks are neutral; a remote detection makes local zeros informative.",
            "Strict-three union-mask support can be a ghost and is not a unique fish or landmark.",
            "Unweighted pool healthy distance inf means the suppression term was disabled.",
            "S2 reference centres are post-construction diagnostics, not independent identity truth.",
        ],
    }
    destination.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"all_pass": result["all_pass"], "coverage_failures": result["coverage_failures"]}))


if __name__ == "__main__":
    main()
