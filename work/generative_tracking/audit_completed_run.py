"""Verify frozen benchmark artifacts and emit compact, independently checked metrics."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


folder = Path(sys.argv[1])
provenance = json.loads((folder / "benchmark.provenance.json").read_text())
comparison = json.loads((folder / "comparison.json").read_text())
assert provenance["status"] == "ok"
checks = {}
for source, expected in provenance["params"]["code_hashes"].items():
    source = Path(source)
    checks[source.name] = {
        "snapshot_matches": digest(folder / "code_snapshot" / source.name) == expected,
        "current_matches": digest(source) == expected,
    }
assert all(all(item.values()) for item in checks.values())
with np.load(folder / "candidate.npz", allow_pickle=False) as arrays:
    frames, pose = arrays["frame_idx"], arrays["pose_xyz"]
    xyz, accepted = arrays["xyz"], arrays["accepted_xyz"]
    observed, review = arrays["observed"], arrays["review"]
    assert np.array_equal(frames, np.arange(*comparison["frames"]))
    np.testing.assert_array_equal(accepted, np.where(review[..., None], np.nan, xyz))
    np.testing.assert_array_equal(np.isfinite(xyz).all(-1), observed[:, :, 1])
    candidate = comparison["candidate"]
    assert candidate["motion"]["observed_animal_frames"] == np.isfinite(xyz).all(-1).sum()
    assert candidate["accepted_observed_animal_frames"] == np.isfinite(accepted).all(-1).sum()
    assert candidate["review_animal_frames"] == review.sum()
    seed = np.asarray(provenance["params"]["seed"]["pose_xyz"])
    initial_lengths = np.linalg.norm(np.diff(seed, axis=1), axis=-1)
    lengths = np.linalg.norm(np.diff(pose, axis=2), axis=-1)
    for slot, item in enumerate(candidate["skeleton"]["slots"]):
        for node, name in enumerate(("mouth", "pectoral", "tail")):
            assert item["observed_nodes"][name]["count"] == observed[:, slot, node].sum()
        for segment, name in enumerate(("mouth-pectoral", "pectoral-tail")):
            values = item["segments"][name]
            np.testing.assert_allclose(values["initial_mm"], initial_lengths[slot, segment])
            np.testing.assert_allclose(values["maximum_mm"], lengths[:, slot, segment].max())
    audit = {
        "run": str(folder),
        "candidate_sha256": digest(folder / "candidate.npz"),
        "code_checks": checks,
        "accepted_mask_consistent": True,
        "summary_counts_consistent": True,
        "skeleton_lengths_independently_verified": True,
        "observed_pectoral_by_slot": observed[:, :, 1].sum(axis=0).tolist(),
        "accepted_pectoral_by_slot": np.isfinite(accepted).all(-1).sum(axis=0).tolist(),
        "review_by_slot": review.sum(axis=0).tolist(),
        "maximum_segment_mm": lengths.max(axis=0).tolist(),
        "maximum_segment_to_seed_ratio": (lengths / initial_lengths[None]).max(axis=0).tolist(),
        "tracking_seconds": comparison["tracking_seconds"],
        "female_raw": candidate["female_identity_raw"],
        "female_accepted": candidate["female_identity_accepted"],
        "diagnostics": candidate["diagnostics"],
        "baseline_arms": comparison["baseline_arms"],
    }
output = folder.parent / (folder.name + "_audit.json")
if output.exists():
    raise FileExistsError(f"refusing to overwrite audit: {output}")
output.write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
print(json.dumps({k: v for k, v in audit.items() if k != "baseline_arms"}, indent=2))
