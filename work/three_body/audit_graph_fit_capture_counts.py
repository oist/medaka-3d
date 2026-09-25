"""Reconstruct LRU cache events from saved proposal counts, without another GPU run."""

import argparse
import hashlib
import json
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    summary_path = args.run / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary["all_arrays_exact"] or summary["initialization_count"] != 1:
        raise ValueError("audit requires the successful single-initialization equivalence run")
    for source in ("tracker.py", "observations.py", "accelerated_tracker.py", "graph_fit.py"):
        path = repo / "labs/20260909_three_body" / source
        if summary["code_hashes"][str(path.relative_to(repo))] != digest(path):
            raise ValueError(f"source changed before cache audit: {source}")
    candidate_path = args.run / "candidate.npz"
    if digest(candidate_path) != summary["candidate_sha256"]:
        raise ValueError("candidate changed")
    with np.load(candidate_path, allow_pickle=False) as data:
        frames, batches = data["frame_idx"], data["hypotheses"]
    config = summary["tracker_config"]
    pixels = None if config["mode"] == "nodes" else sum(
        config[key] for key in ("positive_samples", "near_samples", "global_samples"))
    cache, events = OrderedDict(), []
    for frame, batch in zip(frames, batches, strict=True):
        key = (int(batch), pixels)
        hit = key in cache
        evicted = None
        if hit:
            cache.pop(key)
        elif len(cache) >= 4:
            evicted, _ = cache.popitem(last=False)
        cache[key] = True
        events.append(dict(frame=int(frame), batch=int(batch), pixels=pixels, hit=hit,
                           capture=not hit, evicted=evicted))
    if [str(key) for key in cache] != summary["cached_shapes"]:
        raise ValueError("reconstructed final cache disagrees with recorded runtime cache")
    unique = sorted(set(map(int, batches)))
    captures = sum(row["capture"] for row in events)
    report = dict(
        status="ok", method="deterministic reconstruction, not runtime instrumentation",
        assumptions=[
            "The frozen tracker calls _fit exactly once per native frame.",
            "Every record at a frame stores the same len(proposals) as hypotheses.",
            "The frozen sampler emits the configured fixed N for every view and frame.",
            "The fixed config and dtype do not change during this single run.",
            "The wrapper starts with an empty OrderedDict and uses exactly four LRU entries.",
        ],
        frame_count=len(frames), maximum_entries=4, unique_batches=unique,
        batch_histogram=dict(sorted(Counter(map(int, batches)).items())),
        reconstructed_hits=len(frames) - captures, reconstructed_misses=captures,
        reconstructed_graph_captures=captures,
        reconstructed_evictions=sum(row["evicted"] is not None for row in events),
        recaptures_after_eviction=captures - len(unique),
        hypothetical_unbounded_cache_captures=len(unique),
        replay_calls=len(frames) * (config["opt_steps"] + 1),
        capture_plus_three_warmup_objective_calls=captures * 4,
        measured_peak_torch_allocated_bytes=summary["cuda_peak_allocated_bytes"],
        maximum_reserved_bytes=None,
        memory_limitation=("Peak reserved memory and capture durations were not instrumented. "
                           "They cannot be reconstructed from these outputs. Peak allocated "
                           "is this process's torch allocator, not the concurrent D2 process."),
        final_cache_matches_recorded=True, events=events,
        inputs={str(summary_path): digest(summary_path),
                str(candidate_path): digest(candidate_path)},
        script_sha256=digest(__file__),
    )
    output = args.run / "cache_audit.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("events", "inputs", "assumptions")}, indent=2))


if __name__ == "__main__":
    main()
