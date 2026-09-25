"""Profile the unchanged fixed accelerated command once, then verify its output.

The profiler encloses the entire original main, including preparation and the
post-inference comparison. Reported tracker.track cumulative time isolates the
continuous inference call. No tracker method or numerical operation is patched.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import shutil
import sys
import time
from pathlib import Path

import run_graph_accelerated as runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite profiling experiment")
    source_hash = runner.sha256(__file__)
    command = sys.argv.copy()
    profiler = cProfile.Profile()
    start = time.perf_counter()
    try:
        profiler.runcall(runner.main)
    finally:
        elapsed = time.perf_counter() - start
        if args.output.is_dir():
            profiler.dump_stats(str(args.output / "whole_command.pstats"))
    if runner.sha256(__file__) != source_hash:
        raise RuntimeError("profiling driver changed during execution")
    summary_path = args.output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (not summary["all_arrays_exact"] or len(summary["comparisons"]) != 19
            or summary["initialization_count"] != 1 or summary["frame_count"] != 200):
        raise ValueError("profiled run failed full-output equivalence or continuous input checks")
    stats = pstats.Stats(profiler)
    rows = []
    for (file, line, name), statistics in stats.stats.items():
        primitive, calls, self_seconds, cumulative, callers = statistics
        rows.append(dict(file=file, line=line, name=name, primitive_calls=primitive, calls=calls,
                         self_seconds=self_seconds, cumulative_seconds=cumulative,
                         callers=[dict(file=k[0], line=k[1], name=k[2], statistics=list(value))
                                  for k, value in callers.items()]))
    for order, name in ((pstats.SortKey.CUMULATIVE, "top_cumulative"),
                        (pstats.SortKey.TIME, "top_self")):
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).sort_stats(order).print_stats(65)
        (args.output / f"{name}.txt").write_text(stream.getvalue(), encoding="utf-8")
    selected = (
        ("tracker.py", "track"), ("accelerated_tracker.py", "_fit"),
        ("tracker.py", "_proposals"), ("tracker.py", "_different_identity"),
        ("observations.py", "sample_pixels"), ("observations.py", "__iter__"),
        ("motion.py", "from_frames"), ("motion.py", "_tracked_features"),
        ("motion.py", "_local_warp"), ("node_geometry.py", "supported_nodes"),
        ("node_geometry.py", "assignment_plans"), ("node_geometry.py", "exclusive"),
        ("node_geometry.py", "node_pools"), ("node_geometry.py", "recovery_poses"),
        ("graph_fit.py", "__init__"), ("graph_fit.py", "configure"),
        ("graph_fit.py", "evaluate"), ("graph_fit.py", "_validate_pose"),
        ("graph_fit.py", "_validate_inputs"), ("graph_fit.py", "objective"),
    )
    selected_rows = []
    for file, name in selected:
        matches = [row for row in rows if Path(row["file"]).name == file and row["name"] == name]
        if len(matches) > 1:
            raise ValueError(f"ambiguous profile selector {file}/{name}")
        selected_rows.extend(matches)
    driver_snapshot = args.output / "code_snapshot" / "work/three_body" / Path(__file__).name
    driver_snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, driver_snapshot)
    report = dict(
        status="ok", actual_command=command, profiler="cProfile.Profile",
        scope="unchanged runner.main: preparation, continuous track, post-inference comparison",
        elapsed_profiled_command_seconds=elapsed,
        tracking_seconds=summary["tracking_seconds"],
        profile_total_calls=stats.total_calls, profile_primitive_calls=stats.prim_calls,
        profile_total_self_seconds=stats.total_tt,
        all_19_arrays_exact=True, initialization_count=1, frame_count=200,
        source_unchanged=summary["source_unchanged"],
        candidate_sha256=summary["candidate_sha256"],
        summary_sha256=runner.sha256(summary_path),
        profiler_driver_sha256=source_hash,
        pstats_sha256=runner.sha256(args.output / "whole_command.pstats"),
        selected_functions=selected_rows,
        top_cumulative=sorted(rows, key=lambda row: row["cumulative_seconds"], reverse=True)[:50],
        top_self=sorted(rows, key=lambda row: row["self_seconds"], reverse=True)[:50],
        limitations=[
            "cProfile instrumentation changes Python overhead; timings are not unprofiled latency.",
            "Detector staging may overlap; do not infer a speed ratio against earlier runs.",
            "CUDA kernels execute asynchronously: launch calls and later host synchronizations "
            "receive CPU wall time, not a GPU-kernel execution breakdown.",
            "Native OpenCV/NumPy execution is attributed to exposed C-call boundaries; internal "
            "native worker-thread activity and GPU memory bandwidth are not separately measured.",
            "Nested cumulative times overlap and must not be added as independent contributions.",
            "Preparation and post-inference evaluation are outside tracker.track, but inside "
            "the whole-command profile. This command performs no new identity evaluation.",
        ],
    )
    with (args.output / "profile_summary.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    with (args.output / "profile_all_functions.json").open("x", encoding="utf-8") as stream:
        json.dump(rows, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(status="ok", all_19_arrays_exact=True,
                          tracking_seconds=summary["tracking_seconds"],
                          profiled_command_seconds=elapsed,
                          selected=[{key: value for key, value in row.items() if key != "callers"}
                                    for row in selected_rows])), flush=True)


if __name__ == "__main__":
    main()
