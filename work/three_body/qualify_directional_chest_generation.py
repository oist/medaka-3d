"""Compare the continuous branch's chest generator to all 50 fixed diagnostic frames."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import fit_chest_recovery_candidates as base
import numpy as np
from directional_chest_tracker import chest_proposals
from node_geometry import RayObservations
from tracker import Proposal

from medaka3d.calib import load_rig
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS

checks = base.evidence_tools
REPO = Path(__file__).resolve().parents[2]


def main(replay, recovery, output):
    code = base.local_code()
    hashes = dict(code)
    run = checks.verify(
        replay, "replay.provenance.json", "chest_deficit_original_beam_replay_v1", hashes
    )
    checks.verify(
        recovery, "probe.provenance.json", "chest_recovery_counterfactual_proposals", hashes
    )
    registration = checks.read(replay / "registration/plan.json")
    binding = registration["original_binding"]
    calibration = Path(binding["inputs"]["seed"]["input_root"]) / base.CALIBRATION_DIR
    seed = Path(binding["inputs"]["seed"]["registration_path"]).parent / "initial_pose_xyz.npy"
    for p in [seed, *[p for p in calibration.iterdir() if p.is_file()]]:
        digest = run["inputs"][str(p)]["sha256"]
        checks.require(checks.sha(p) == digest, "original seed/calibration differs")
        hashes[str(p)] = digest
    tracker = SimpleNamespace(
        rig=load_rig(calibration),
        lengths=np.linalg.norm(np.diff(np.load(seed, allow_pickle=False), axis=1), axis=-1),
        config=base.TrackerConfig(
            **{**registration["original_context"]["tracker_config"], "device": "cpu"}
        ),
        lower=np.array([0.0, 0.0, -6.0]),
        upper=np.array([192.0, 192.0, 186.0]),
    )
    rows = []
    for expected in checks.read(recovery / "frames.json"):
        native = expected["frame_idx"]
        folder = replay / "diagnostics" / f"frame_{native:06d}"
        a = base.load_arrays(folder / "arrays.npz", hashes)
        kinds = checks.read(folder / "metadata.json")["proposal_kind"]
        proposals = [
            Proposal(
                a["proposal_pose"][i],
                a["proposal_predicted"][i],
                a["proposal_plan"][i],
                int(a["proposal_parent"][i]),
                kinds[i],
                tuple(a["proposal_permutation"][i].tolist()),
            )
            for i in range(len(kinds))
        ]
        states = [
            SimpleNamespace(
                missing=a["parent_missing"][0], node_missing=a["parent_node_missing"][0]
            )
        ]
        rays = RayObservations.from_points(tracker.rig, {v: a[v + "_points"] for v in VIEWS})
        extra, diagnostic = chest_proposals(proposals, states, rays, tracker)
        checks.require(
            diagnostic["enabled"] == expected["enabled"]
            and len(extra) == expected["returned_candidates"]
            and diagnostic["excluded_by_budget"] == 0,
            "admission differs from local experiment",
        )
        if extra:
            old = base.load_arrays(recovery / f"frame_{native:06d}.npz", hashes)
            for name, actual in (
                ("pose_xyz", np.stack([p.pose for p in extra])),
                ("plan", np.stack([p.plan for p in extra])),
            ):
                checks.require(
                    checks.equal_bytes(actual, old[name]), "continuous generator differs: " + name
                )
        rows.append(dict(frame_idx=native, **diagnostic, generated_pose_plan_exact=True))
    checks.require(
        len(rows) == 50 and sum(r["admitted"] for r in rows) == 31, "fixed coverage incomplete"
    )
    output.mkdir(parents=True, exist_ok=False)
    with provenance(
        "directional_chest_generation_qualification", output, name="qualification"
    ) as record:
        result = dict(
            schema="directional_chest_generation_qualification_v1",
            status="ok",
            frames=50,
            admitted_candidates=31,
            generated_pose_plan_all_byte_exact=True,
            CUDA_used=False,
            temporal_run=False,
            independent_ID_truth=False,
            rows=rows,
        )
        path = output / "summary.json"
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        record.output(path.name, path, sha256=checks.sha(path))
        checks.require(code == base.local_code(), "source set changed")
        for path, digest in hashes.items():
            checks.require(checks.sha(path) == digest, "bound input changed")
            record.input(path, path, sha256=digest)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", type=Path, required=True)
    p.add_argument("--recovery", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    output = args.output.resolve()
    checks.require(
        output.is_relative_to(REPO / "work") and not output.exists(),
        "new workspace output required",
    )
    main(args.replay.resolve(), args.recovery.resolve(), output)
