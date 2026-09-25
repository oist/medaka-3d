"""Check completed new-cycle outputs without changing them or recomputing inference."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

repo = Path(__file__).resolve().parents[2]
run = Path(sys.argv[1])
provenance = json.loads((run / "benchmark.provenance.json").read_text())
summary = json.loads((run / "comparison.json").read_text())
assert provenance["status"] == "ok" and summary["technical_completion"]
checks = {}
for path, expected in provenance["params"]["code_hashes"].items():
    path = Path(path)
    relative = path.relative_to(repo)
    snapshot = run / "code_snapshot" / relative
    checks[relative.as_posix()] = {
        "snapshot_matches": hashlib.sha256(snapshot.read_bytes()).hexdigest() == expected,
        "current_matches": hashlib.sha256(path.read_bytes()).hexdigest() == expected,
    }
assert all(check["snapshot_matches"] for check in checks.values())
with np.load(run / "candidate.npz", allow_pickle=False) as arrays:
    frames, xyz, accepted = arrays["frame_idx"], arrays["xyz"], arrays["accepted_xyz"]
    observed, pose, review = arrays["observed"], arrays["pose_xyz"], arrays["review"]
    np.testing.assert_array_equal(frames, np.arange(*summary["frames"]))
    np.testing.assert_array_equal(accepted, np.where(review[..., None], np.nan, xyz))
    np.testing.assert_array_equal(np.isfinite(xyz).all(-1), observed[:, :, 1])
    np.testing.assert_array_equal(review, arrays["geometry_review"] | arrays["identity_review"])
    assert arrays["appearance"].shape == (len(frames), 3, 3, 24)
    assert arrays["appearance_quality"].shape == (len(frames), 3, 3)
    assert summary["candidate"]["motion"]["observed_animal_frames"] == observed[:, :, 1].sum()
    assert summary["candidate"]["accepted_observed_animal_frames"] == np.isfinite(accepted).all(-1).sum()
    assert summary["candidate"]["review_animal_frames"] == review.sum()
    seed = np.asarray(provenance["params"]["seed"]["pose_xyz"])
    lengths = np.linalg.norm(np.diff(pose, axis=2), axis=-1)
    initial = np.linalg.norm(np.diff(seed, axis=1), axis=-1)
    assert summary["candidate"]["diagnostics"]["initializations"] == 1
    audit = {
        "run": str(run), "candidate_sha256": hashlib.sha256((run / "candidate.npz").read_bytes()).hexdigest(),
        "code_checks": checks, "arrays_and_summary_consistent": True,
        "n_frames": len(frames), "tracking_seconds": summary["tracking_seconds"],
        "tracking_fps": summary["tracking_fps"],
        "observed_pectoral_by_slot": observed[:, :, 1].sum(0).tolist(),
        "accepted_pectoral_by_slot": np.isfinite(accepted).all(-1).sum(0).tolist(),
        "review_by_slot": review.sum(0).tolist(),
        "last_observed_nodes": observed[-1].tolist(), "last_review": review[-1].tolist(),
        "maximum_segment_to_seed_ratio": (lengths / initial[None]).max(0).tolist(),
        "candidate": summary["candidate"],
    }
out = run.parent / (run.name + "_audit.json")
if out.exists():
    raise FileExistsError(out)
out.write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
print(json.dumps({k:v for k,v in audit.items() if k not in ("candidate", "code_checks")}, indent=2))
