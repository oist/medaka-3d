"""Audit immutable postpass outputs and their exact raw permutations."""
import hashlib
import json
from pathlib import Path

import numpy as np

base = Path(__file__).resolve().parent
repo = base.parents[1]
audits = {}
for window in ("swap_a", "swap_b", "occlusion", "temporal_control"):
    run = base / f"reconnect_fixed_validation_{window}"
    raw_run = base / f"recovery_v2_validation_{window}"
    record = json.loads((run / "benchmark.provenance.json").read_text())
    assert record["status"] == "ok"
    code = {}
    for source, expected in record["params"]["code_hashes"].items():
        path = Path(source)
        relative = path.relative_to(repo)
        snapshot = run / "code_snapshot" / relative
        matches = hashlib.sha256(snapshot.read_bytes()).hexdigest() == expected
        assert matches
        code[relative.as_posix()] = {
            "snapshot_matches": matches,
            "current_matches": hashlib.sha256(path.read_bytes()).hexdigest() == expected,
            "sha256": expected,
        }
    for key, output in record["outputs"].items():
        if "sha256" in output:
            assert hashlib.sha256(Path(output["path"]).read_bytes()).hexdigest() == output["sha256"]
    with np.load(run / "candidate.npz", allow_pickle=False) as post:
        with np.load(raw_run / "candidate.npz", allow_pickle=False) as raw:
            np.testing.assert_array_equal(post["frame_idx"], raw["frame_idx"])
            rows = np.arange(len(raw["frame_idx"]))[:, None]
            perm = post["permutation"]
            np.testing.assert_array_equal(np.sort(perm, axis=1), np.tile(np.arange(3), (len(rows), 1)))
            for field in ("pose_xyz", "xyz", "observed", "node_source"):
                np.testing.assert_array_equal(post[field], raw[field][rows, perm])
            histogram = raw["appearance"].astype(float)
            mass = histogram.sum(-1, keepdims=True)
            normalized = np.divide(histogram, mass, out=np.zeros_like(histogram), where=mass > 0)
            np.testing.assert_array_equal(post["appearance"], normalized[rows, perm])
            quality = (raw["appearance_quality"].astype(float)
                       * (mass[..., 0] > 0) * raw["observed"][:, :, :2].all(-1)[..., None])
            np.testing.assert_array_equal(post["appearance_quality"], quality[rows, perm])
            assert not np.any(raw["geometry_review"][rows, perm] & ~post["geometry_review"])
            np.testing.assert_array_equal(post["accepted_xyz"],
                                          np.where(post["review"][..., None], np.nan, post["xyz"]))
            audits[window] = {
                "code": code,
                "raw_geometry_source_exactly_permuted": True,
                "appearance_matches_declared_normalization_and_support_mask": True,
                "geometry_review_never_removed": True,
                "candidate_accepted_mask_consistent": True,
                "output_hashes_valid": True,
                "last_permutation": perm[-1].tolist(),
            }
out = base / "fixed_postpass_audit.json"
if out.exists():
    raise FileExistsError(out)
out.write_text(json.dumps(audits, indent=2, allow_nan=False))
print(json.dumps({key: {k: v for k, v in value.items() if k != "code"}
                  for key, value in audits.items()}, indent=2))
