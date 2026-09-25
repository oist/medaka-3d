"""Bind the final native82720 pose to its saved fitted proposal and source scene."""

import hashlib
import io
import json
from pathlib import Path

import diagnose_body_row_82720_fit as old
import numpy as np

WORK = Path(__file__).resolve().parent
RUN = WORK / "nodewise_body_row_ten_second_continuation_v1"
COMPARISON = WORK / "nodewise_body_row_ten_second_comparison_v1"
TRACKING_SHA = "d6c1dc36ef1c79901fa47ce536585f3b7e0a4e35984fdcd7e12f765ccb0a0985"
COMPARISON_SHA = "4ba0f03c0d8887f73b2a8b72bd57785be0560c3ccfdd1cf377447da64615ed53"


def run():
    output = WORK / "NODEWISE_82720_SELECTED_SCENE_V1.json"
    old.require(not output.exists(), "fresh output required")
    inputs = {}

    def capture(path, digest=None):
        path = Path(path).resolve()
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        old.require(digest is None or digest == sha, "input hash mismatch")
        inputs[str(path)] = dict(sha256=sha, size_bytes=len(data))
        return data

    def member(root, seal, name, decode_json=True):
        record = seal["outputs"][name]
        path = (root / name).resolve()
        old.require(str(path) == record["path"], "member path mismatch")
        data = capture(path, record["sha256"])
        old.require(len(data) == record["size_bytes"], "member size mismatch")
        return json.loads(data) if decode_json else data

    capture(__file__)
    capture(old.__file__)
    tracking = json.loads(capture(RUN / "tracking.provenance.json", TRACKING_SHA))
    comparison = json.loads(capture(COMPARISON / "comparison.provenance.json", COMPARISON_SHA))
    old.require(tracking["status"] == comparison["status"] == "ok", "completed producers")
    summary = member(RUN, tracking, "summary.json")
    comparison_summary = member(COMPARISON, comparison, "summary.json")
    old.require(
        comparison_summary["candidate_references"]["nodewise_body_row"]["seal_sha256"]
        == TRACKING_SHA
        and comparison_summary["terminal_horizon"] == 83000,
        "comparison lineage or terminal mismatch",
    )
    records = []
    for prefix in ("body_row_diagnostics", "diagnostics"):
        role = f"{prefix}/f82720"
        meta = member(RUN, tracking, role + ".json")
        old.require(meta["archive"] == tracking["outputs"][role + ".npz"], "archive edge")
        with np.load(
            io.BytesIO(member(RUN, tracking, role + ".npz", False)), allow_pickle=False
        ) as z:
            records.append(old.restore(meta["tree"], {k: z[k] for k in z.files}))
    before, after = records
    old.require(before["frame_idx"] == after["frame_idx"] == 82720, "native frame mismatch")
    signatures = member(COMPARISON, comparison, "review_samples_signatures.json")
    with np.load(
        io.BytesIO(member(COMPARISON, comparison, "review_samples.npz", False)), allow_pickle=False
    ) as z:
        sample = {k: z[k] for k in z.files if k.startswith("nodewise_body_row_")}
    old.require(all(old.signature(v) == signatures[k] for k, v in sample.items()), "sample bytes")
    row = int(np.flatnonzero(sample["nodewise_body_row_frame_idx"] == 82720)[0])
    pose = sample["nodewise_body_row_pose_xyz"][row]
    kind = str(sample["nodewise_body_row_proposal_kind"][row])
    matches = [
        i
        for i, p in enumerate(after["candidate_pose_xyz"])
        if np.array_equal(p, pose) and after["candidate_kind"][i] == kind
    ]
    old.require(len(matches) == 1, "selected fitted proposal must be unique")
    selected = matches[0]
    added_position = np.flatnonzero(before["new_proposal_indices"] == selected)
    old.require(len(added_position) == 1, "final proposal must be one admitted body scene")
    scene = before["scenes"][int(added_position[0])]
    old.require(
        np.array_equal(scene["proposed_sources"], after["candidate_plan"][selected]),
        "scene and fitted plan differ",
    )
    retained, _ = old.retained_candidates(
        after["candidate_pose_xyz"],
        after["candidate_parent"],
        after["candidate_permutation"],
        after["adjusted_path_cost"],
        summary["tracker_context"]["tracker_config"]["beam_size"],
    )
    old.require(selected in retained, "selected proposal was not retained")
    bodies = [before["pool"][i] for i in scene["pool_indices"]]
    result = dict(
        status="ok",
        frame_idx=82720,
        inputs=inputs,
        final_selected_proposal_index=selected,
        final_kind=kind,
        scene_pool_indices=scene["pool_indices"],
        supplemental_nodewise_bodies=[
            b["pool_index"] for b in bodies if b.get("supplemental_nodewise_body", False)
        ],
        triangulated_nodes_before_fit=scene["triangulated_nodes"],
        stored_direct_support_after_fit=sample["nodewise_body_row_observed"][row],
        prefit_pose=scene["pose_xyz"],
        fitted_pose=pose,
        fit_displacement_by_slot_node_mm=np.linalg.norm(pose - scene["pose_xyz"], axis=-1),
        selected_adjusted_local_cost=after["adjusted_local_cost"][selected],
        selected_adjusted_path_cost=after["adjusted_path_cost"][selected],
        retained_proposal_indices=retained,
        candidate_generation_or_fit_performed=False,
        physical_ID_truth=False,
        original_candidate_or_state_archives_read=False,
    )
    output.write_text(
        json.dumps(old.safe(result), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: old.safe(result[k])
                for k in (
                    "final_selected_proposal_index",
                    "scene_pool_indices",
                    "supplemental_nodewise_bodies",
                    "fit_displacement_by_slot_node_mm",
                    "retained_proposal_indices",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
