"""Seal the CPU sizing evidence after its note and source hashes are reviewed."""

import hashlib
import json
from pathlib import Path

from medaka3d.provenance import provenance

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work/three_body/long_run_feasibility_v3"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


data = json.loads((OUT / "summary.json").read_text())
assert data["code_sha256_before"] == data["code_sha256_after"]
with provenance("long_run_cpu_feasibility_evidence", OUT, name="audit") as record:
    record.param("tracking_executed", False)
    record.param("cuda_allocations_requested", False)
    record.param("sizing_is_not_rss_or_full_long_run", True)
    record.param("live_progress_is_a_saved_snapshot_only", True)
    for relative, expected in data["code_sha256_before"].items():
        path = ROOT / relative
        assert sha256(path) == expected
        record.input(relative, path, sha256=expected)
    for views in data["packed_masks"].values():
        for info in views.values():
            path = ROOT / info["path"]
            assert sha256(path) == info["sha256"]
            record.input(info["path"], path, sha256=info["sha256"])
    for view, info in data["sleap_full_archives"].items():
        path = ROOT / (
            "work/generative_tracking_inputs/work/c01_s16_predictions/bottomup/"
            f"20251225_K_1_2_FCMC__{view}.npz")
        assert sha256(path) == info["sha256"]
        record.input("sleap:" + view, path, sha256=info["sha256"])
    for name in ("summary.json", "audit.py", "minute_progress_snapshot.jsonl", "FEASIBILITY.md"):
        path = OUT / name
        record.output(name, path, sha256=sha256(path))
    timing = ROOT / "work/three_body/graph_accelerated_transient_200_v1/summary.json"
    assert sha256(timing) == data["accelerated_200_timing"]["sha256"]
    record.input("accelerated_200_summary", timing, sha256=sha256(timing))
    record.input("seal_script", Path(__file__), sha256=sha256(Path(__file__)))
print("CPU-only feasibility evidence sealed; no tracker or GPU inference executed.")
