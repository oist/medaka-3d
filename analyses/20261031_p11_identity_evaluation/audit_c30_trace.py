"""Observe the unchanged C30 v4 function and verify replay against stored output."""

import inspect
import sys

import numpy as np


def trace_stitch(module, xyz, ok_idx, degraded, times, *, window, margin_mm):
    """Return original stitch outputs plus diagnostics; never change decisions."""
    source, first_line = inspect.getsourcelines(module.stitch)

    def unique_line(prefix):
        found = [first_line + i for i, line in enumerate(source) if line.strip().startswith(prefix)]
        if len(found) != 1:
            raise ValueError(f"C30 trace source anchor is not unique: {prefix}")
        return found[0]

    median_line = unique_line("if best == 0:")
    zone_line = unique_line("if clean_p > margin_mm and clean_v > margin_mm:")
    boundary, zones, velocity_calls = {}, {}, {}
    stitch_code, velocity_code = module.stitch.__code__, module._velocity_costs.__code__

    def trace(frame, event, arg):
        if frame.f_code not in (stitch_code, velocity_code):
            return None
        values = frame.f_locals
        if frame.f_code is velocity_code and event == "return":
            t0, tb = values.get("t0"), values.get("tb")
            velocity_calls[int(values["target_frame"])] = {
                "velocity_called": True,
                "velocity_costs_available": arg is not None,
                "configured_velocity_window_frames": int(values["velocity_window"]),
                "velocity_history_start_frame": tb,
                "velocity_history_end_frame": t0,
                "velocity_history_span_seconds": (
                    float(times[t0] - times[tb]) if t0 is not None and tb is not None else None
                ),
                "velocity_unavailable_reason": (
                    None
                    if arg is not None
                    else "history_too_short"
                    if t0 is not None
                    and tb is not None
                    and t0 - tb < values["velocity_window"] // 2
                    else "nonfinite_history_or_target"
                ),
                "velocity_proposed_permutation": (
                    list(module.PERMS[int(arg.argmin())]) if arg is not None else None
                ),
            }
        if frame.f_code is stitch_code and event == "line":
            if frame.f_lineno == median_line:
                boundary[int(values["frame"])] = {
                    "median_best_is_identity": values["best"] == 0,
                    "median_margin_mm": float(values["margin"]),
                    "median_proposed_permutation": list(module.PERMS[values["best"]]),
                    "median_history_start_frame": int(values["hist_frames"][0]),
                    "median_history_end_frame": int(values["hist_frames"][-1]),
                }
            elif frame.f_lineno == zone_line:
                t0, t1, tb = (int(values[key]) for key in ("t0", "t1", "tb"))
                zones[(int(values["first"]), int(values["last"]))] = {
                    "before_frame": t0,
                    "after_frame": t1,
                    "velocity_called": True,
                    "velocity_costs_available": True,
                    "velocity_history_start_frame": tb,
                    "velocity_history_end_frame": t0,
                    "velocity_history_span_seconds": float(times[t0] - times[tb]),
                    "velocity_proposed_permutation": list(module.PERMS[values["best_v"]]),
                    "median_best_is_identity": values["best_p"] == 0,
                    "decisively_kept_identity": (
                        values["clean_p"] > margin_mm and values["clean_v"] > margin_mm
                    ),
                }
        return trace

    previous_trace = sys.gettrace()
    sys.settrace(trace)
    try:
        perm, joins, refusals = module.stitch(
            xyz, ok_idx, window, margin_mm, module.GAP_FRAMES, degraded=degraded
        )
    finally:
        sys.settrace(previous_trace)
    boundary_joins = {j["frame"]: j for j in joins if "zone" not in j}
    boundary_refusals = {r["frame"]: r for r in refusals if "zone" not in r}
    records = []
    for k in module.candidate_boundaries(xyz, ok_idx, module.GAP_FRAMES):
        after, before = int(ok_idx[k]), int(ok_idx[k - 1])
        detail = boundary.get(after, {})
        joined, refused = boundary_joins.get(after), boundary_refusals.get(after)
        decision = "adopted" if joined else "refused" if refused else "kept_identity"
        record = {
            "kind": "boundary",
            "before_frame": before,
            "after_frame": after,
            "decision": decision,
            "evidence": joined["evidence"] if joined else None,
            "reason": refused["reason"] if refused else None,
            "velocity_called": False,
            "velocity_costs_available": False,
            "configured_velocity_window_frames": module.VELOCITY_WINDOW,
            "velocity_history_start_frame": None,
            "velocity_history_end_frame": None,
            "velocity_history_span_seconds": None,
            **detail,
            **velocity_calls.get(after, {}),
        }
        record["velocity_adopted"] = record["evidence"] == "velocity"
        records.append(record)
    zone_joins = {tuple(j["zone"]): j for j in joins if "zone" in j}
    zone_refusals = {tuple(r["zone"]): r for r in refusals if "zone" in r}
    for zone in module.turbulent_zones(xyz, ok_idx, degraded):
        key = tuple(zone)
        joined, refused = zone_joins.get(key), zone_refusals.get(key)
        raw = joined or refused or {}
        detail = zones.get(key, {})
        records.append(
            {
                "kind": "zone",
                "zone_start_frame": int(zone[0]),
                "zone_last_frame": int(zone[1]),
                "before_frame": raw.get("pre_frame"),
                "after_frame": raw.get("post_frame"),
                "decision": "adopted" if joined else "refused" if refused else "kept_identity",
                "reason": refused["reason"] if refused else None,
                "evidence": "bridge" if joined else None,
                "velocity_called": False,
                "velocity_costs_available": False,
                "configured_velocity_window_frames": module.VELOCITY_WINDOW,
                "velocity_adopted": bool(joined),
                "velocity_history_span_seconds": None,
                **detail,
            }
        )
    for i, record in enumerate(records):
        record["c30_decision_id"] = i
        record["physical_identity_correctness"] = "not_adjudicated"
    return perm, joins, refusals, records


def require_identical_replay(actual, saved_perm, saved_arm):
    perm, joins, refusals, _records = actual
    if not np.array_equal(perm, saved_perm):
        raise ValueError("Instrumented C30 permutation differs from the saved run")
    if joins != saved_arm["join_list"] or refusals != saved_arm["refusal_list"]:
        raise ValueError("Instrumented C30 decisions differ from the saved run")
