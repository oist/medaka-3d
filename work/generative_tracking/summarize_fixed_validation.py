"""Read completed fixed runs; do not change any inference outputs."""
import hashlib
import json
from pathlib import Path

import numpy as np

base = Path(__file__).resolve().parent
rows = {}
for window in ("swap_a", "swap_b", "occlusion", "temporal_control"):
    methods = {}
    for method, prefix in (("raw", "recovery_v2"), ("postpass", "reconnect_fixed")):
        run = base / f"{prefix}_validation_{window}"
        summary = json.loads((run / "comparison.json").read_text())
        provenance = json.loads((run / "benchmark.provenance.json").read_text())
        c = summary["candidate"]
        with np.load(run / "candidate.npz", allow_pickle=False) as arrays:
            counts = np.isfinite(arrays["accepted_xyz"]).all(-1).sum(0).tolist()
            review = arrays["review"].sum(0).tolist()
            identity = arrays["identity_review"].sum(0).tolist()
        methods[method] = {
            "frames": summary["frames"],
            "seconds": summary.get("tracking_seconds", summary.get("postpass_seconds")),
            "observed": c["motion"]["observed_animal_frames"],
            "total": c["motion"]["total_animal_frames"],
            "accepted": c["accepted_observed_animal_frames"],
            "accepted_by_slot": counts,
            "review_by_slot": review,
            "identity_review_by_slot": identity,
            "female_raw": c["female_identity_raw"],
            "female_accepted": c["female_identity_accepted"],
            "availability": c["availability_assessment"],
            "nominal_tank": c["nominal_tank"],
            "diagnostics": c["diagnostics"],
            "snapshot_hashes": provenance["params"]["code_hashes"],
        }
        if method == "raw":
            baseline = {}
            for arm, result in sorted(summary["baseline_arms"].items(), key=lambda x: float(x[0])):
                baseline[arm] = {
                    "observed": result["motion"]["observed_animal_frames"],
                    "female": result["female_identity"],
                }
            methods["baseline"] = baseline
    rows[window] = methods
out = base / "fixed_validation_summary.json"
if out.exists():
    raise FileExistsError(out)
out.write_text(json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8")
for window, row in rows.items():
    print(window)
    for method in ("raw", "postpass"):
        c = row[method]
        print(method, "observed", c["observed"], "accepted", c["accepted"], c["accepted_by_slot"],
              "female_raw", c["female_raw"]["interval_counts"],
              "female_accepted", c["female_accepted"]["interval_counts"],
              "physical_feasible", c["diagnostics"].get("physically_feasible_global_path"),
              "floor", c["nominal_tank"])
    print("baseline", {a: [v["observed"], v["female"]["interval_counts"]] for a, v in row["baseline"].items()})
