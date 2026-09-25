"""Run deterministic review proposals on the two completed v1 development arms."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))
from review_plan import build_review_plan  # noqa: E402


def sha(path):
    return hashlib.file_digest(Path(path).open("rb"), "sha256").hexdigest()


out = ROOT / "work/three_body/review_plan_development_v1"
out.mkdir(exist_ok=False)
code = [Path(__file__), LAB / "review_plan.py", LAB / "review_events.py"]
paths = [ROOT / "work/three_body" / name / "candidate.npz" for name in (
    "v1_transient_detectron2", "v1_interval_detectron2")]
before = {str(p): sha(p) for p in code + paths}
reports = {}
for path in paths:
    with np.load(path, allow_pickle=False) as data:
        report = build_review_plan({k: data[k] for k in data.files})
    reports[path.parent.name] = report
    (out / f"{path.parent.name}.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
after = {str(p): sha(p) for p in code + paths}
if before != after:
    raise RuntimeError("candidate or code changed during proposal audit")
summary = dict(
    status="ok", code_and_input_sha256=before, inputs_unchanged=True,
    output_sha256={p.name: sha(p) for p in out.glob("*.json")},
    counts={name: {key: report[key] for key in (
        "proposed_scene_count", "incomplete_scene_count", "risk_counts")}
        for name, report in reports.items()},
    human_labels_used=False, ids_corrected=False,
    limitations=["Existing completed development candidates only; no new tracking.",
                 "Proposals select inspection times, never identity answers.",
                 "No user JSON, review pack or human correction is created here."],
)
(out / "audit.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
print(json.dumps(summary["counts"], indent=2))
