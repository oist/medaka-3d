"""Suggest raw-image review times from a completed tracker's uncertainty inventory.

These are model-selected inspection scenes, not labels, repairs or an independent
test set. Probe selection uses observation flags and native frame indices. The
shared inventory additionally validates poses and summarizes their motion; no
image, loss ranking or identity truth is used to select a probe.
"""

from __future__ import annotations

import numpy as np
from review_events import extract_review_events


def build_review_plan(result: dict) -> dict:
    """Choose supported probes around each whole scene without hiding failures.

    Entry is the last all-mouth-observed frame before the first risk. After is
    the first such frame after the last risk; exit is the last later such frame
    inside the original context. Do not move outside context or across gaps to
    manufacture a completed proposal. Existing review flags do not suppress
    candidate probes; they are reported explicitly, including unresolved entry.
    """
    inventory = extract_review_events(result)
    frames = np.asarray(result["frame_idx"])
    mouths = np.asarray(result["observed"])[:, :, 0]
    flagged = np.asarray(result["identity_review"])
    all_mouths = mouths.all(axis=1)
    scenes = []
    for scene in inventory["scenes"]:
        lo, hi = scene["context_start_frame"], scene["context_stop_frame_exclusive"]
        risk_start, risk_stop = scene["risk_start_frame"], scene["risk_stop_frame_exclusive"]
        in_context = (frames >= lo) & (frames < hi)
        before = np.flatnonzero(in_context & (frames < risk_start) & all_mouths)
        after = np.flatnonzero(in_context & (frames >= risk_stop) & all_mouths)
        indices = [int(before[-1]) if len(before) else None,
                   int(after[0]) if len(after) else None,
                   int(after[-1]) if len(after) >= 2 else None]
        issues = []
        if indices[0] is None:
            issues.append("no all-three-mouth-supported entry before risk inside context")
        if indices[1] is None:
            issues.append("no all-three-mouth-supported return after risk inside context")
        if indices[2] is None:
            issues.append("no distinct later all-three-mouth-supported exit inside context")
        probes = [int(frames[i]) if i is not None else None for i in indices]
        reviews = [flagged[i].tolist() if i is not None else None for i in indices]
        entry_reviewed = bool(flagged[indices[0]].any()) if indices[0] is not None else None
        scenes.append({
            **scene,
            "status": "proposed_for_human_review" if not issues else "needs_probe_selection",
            "proposed_anchor_frames": probes,
            "anchor_identity_review_by_slot": reviews,
            "entry_contains_unresolved_machine_id": entry_reviewed,
            "all_three_native_mouths_directly_observed_at_proposed_probes": [
                bool(all_mouths[i]) if i is not None else None for i in indices
            ],
            "selection_issues": issues,
            "human_confirmed_links": 0,
            "ids_repaired": False,
            "biological_names_assigned": False,
        })
    return {
        "schema": "three_body_raw_review_proposal_v1",
        "scope": "model-selected uncertainty inspection; not identity truth or correction",
        "source_blocks": inventory["available_blocks"],
        "risk_counts": inventory["counts"],
        "proposed_scene_count": sum(not s["selection_issues"] for s in scenes),
        "incomplete_scene_count": sum(bool(s["selection_issues"]) for s in scenes),
        "scenes": scenes,
        "limitations": [
            "All risk scenes remain in the denominator, including unavailable probes.",
            "Supported mouths may still be erroneous; humans inspect raw images independently.",
            "Sticky review is retained; an unresolved entry does not establish a global ID.",
            "No anonymous names, mouth coordinates or continuity answers are prefilled.",
            "Three probes cannot establish dense continuity or locate a returning swap.",
            "Scene selection uses this model's failures and is not an independent test set.",
            "Unflagged swaps and the actual manual review burden are not measured.",
        ],
    }
