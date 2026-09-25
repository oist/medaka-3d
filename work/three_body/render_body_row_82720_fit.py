"""Render one saved pre-fit/post-fit/final comparison; no fitting or support audit."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import render_body_row_continuation as frozen

common, drawing = frozen.common, frozen.drawing
proof, require, WORK = frozen.proof, frozen.require, frozen.WORK
STAGE = "body_row_82720_fit_visual_v1"
DIAG = WORK / "BODY_ROW_82720_FIT_DIAGNOSIS_V1.json"
DIAG_SHA = "adba218510ea7c5c12d904aac402a9673bbc551dc3ad6bd6e74d4ebc68c28831"
PINS = {
    "render_body_row_continuation.py": (
        "ff09bff96392c3445c61dbb24324897a4b37d6c91ecd1fa81e23a056a9d05af2"
    ),
    "render_partial_body_bank_comparison.py": (
        "7658faac53fa0cb19ccfb6b1323144e26ef4c4ef8d4030fb4182d988af3d684e"
    ),
}


def run(output):
    output = Path(output).resolve()
    require(output.is_relative_to(WORK) and not output.exists(), "fresh workspace output")
    out, evidence = proof.Output(output), proof.Evidence()
    for name, digest in PINS.items():
        evidence.add(WORK / name, digest)
    evidence.add(Path(__file__))
    evidence.add(DIAG, DIAG_SHA)
    data = proof.read_json(DIAG)
    require(data["status"] == "ok" and data["frame_idx"] == 82720, "saved native diagnosis")
    scene = [s for s in data["scenes"] if s["candidate_index"] == 19]
    require(len(scene) == 1 and scene[0]["admitted"], "one admitted candidate19 required")
    scene = scene[0]
    final = data["candidates"][0]
    require(
        final["candidate_index"] == data["final_selected_candidate_index"] == 0
        and final["kind"] == "predict"
        and final["final_83000_backtrace_selected"],
        "saved final candidate0 differs",
    )
    labels = ("Candidate19 PRE-FIT", "Candidate19 POST-FIT", "Candidate0 FINAL predict")
    poses = np.asarray(
        [scene["pre_fit_pose_xyz"], scene["post_fit_pose_xyz"], final["pose_xyz"]], np.float64
    )
    require(poses.shape == (3, 3, 3, 3) and np.isfinite(poses).all(), "three finite body poses")
    tracking_path = next(
        Path(p) for p in data["inputs"] if Path(p).name == "tracking.provenance.json"
    )
    evidence.add(tracking_path, data["inputs"][str(tracking_path)]["sha256"])
    tracking = proof.read_json(tracking_path)
    require(tracking["status"] == "ok", "completed tracking source")
    plan = proof.read_json(
        common.bound_member(tracking_path.parent, tracking, "plan.json", evidence)
    )
    shared = plan["original_binding"]["inputs"]["seed"]["SL_and_calibration_sha256"]
    primary = drawing.PRIMARY_ROOT
    evidence.add(primary / "render.provenance.json", drawing.PRIMARY_SEAL_SHA)
    primary_seal = proof.read_json(primary / "render.provenance.json")
    require(primary_seal["status"] == "ok", "registered native photographs")
    for path, digest in shared.items():
        require(
            tracking["inputs"].get(path) == primary_seal["inputs"].get(path) == digest,
            "saved photograph/tracker SL-calibration lineage differs",
        )
        if Path(path).suffix == ".json":
            evidence.add(path, digest)
    folders = [Path(p).parent for p in shared if Path(p).name == "calibration_manifest.json"]
    require(len(folders) == 1, "one registered rig")
    rig = common.metrics.old.prior.load_rig(folders[0])
    require(
        plan["tracker_context"]["cameras"]
        == common.support.COMMON.normalize([asdict(c) for c in rig.cameras]),
        "rig differs",
    )
    document = common.bound_member(
        primary, primary_seal, "source_and_frame_evidence.json", evidence
    )
    require(proof.sha(document) == drawing.PRIMARY_EVIDENCE_SHA, "saved SL points differ")
    context = dict(
        primary_seal=primary_seal,
        records={r["frame"]: r for r in proof.read_json(document)["records"]},
    )
    projections = {view: rig.camera(view).project(poses) for view in drawing.VIEWS}
    require(
        all(p.shape == (3, 3, 3, 2) and np.isfinite(p).all() for p in projections.values()),
        "finite three-camera projections required",
    )
    photos, crops = [], {}
    fig, axes = plt.subplots(3, 3, figsize=(18, 15), squeeze=False)
    for vi, view in enumerate(drawing.VIEWS):
        image, points, reference = drawing.photograph_and_points(context, 82720, view, evidence)
        photos.append(reference)
        lo, hi = drawing.visual.display_crop(image, None, points, projections[view])
        crops[view] = dict(lo=lo.tolist(), hi=hi.tolist())
        for row, label in enumerate(labels):
            ax = axes[row, vi]
            drawing.draw_panel(ax, image, points, projections[view][row])
            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(hi[1], lo[1])
            detail = (
                "S1 tail is template-filled; direct support not recomputed"
                if row == 0
                else "Direct support not recomputed for candidate19"
                if row == 1
                else "Stored direct M/C/T: S0=111, S1=000, S2=000"
            )
            ax.set_title(f"{label} | {view}\n{detail}", fontsize=10)
    require(
        data["final_selected_observed"] == [[True] * 3, [False] * 3, [False] * 3],
        "final support caption differs",
    )
    fig.suptitle(
        "Native82720 | Candidate19 before/after fit and final candidate0 at horizon83000\n"
        "Open markers: all estimated M circle / C square / T triangle; orange +: saved SLEAP\n"
        "S0 cyan / S1 yellow / S2 magenta: raw slots, NOT matched physical IDs. "
        "Same crop per camera.",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    target = output / "frame_82720_pre_post_final.png"
    fig.savefig(target, dpi=160)
    plt.close(fig)
    out.records[target.name] = dict(
        path=str(target), sha256=proof.sha(target), size_bytes=target.stat().st_size
    )
    out.write(
        "summary.json",
        dict(
            schema=STAGE,
            status="ok",
            frame_idx=82720,
            labels=list(labels),
            diagnosis_sha256=DIAG_SHA,
            pose_xyz=poses.tolist(),
            pose_signature=proof.signature(poses),
            projections={v: p.tolist() for v, p in projections.items()},
            crops=crops,
            photographs=photos,
            template_filled_nodes=scene["template_filled_nodes"],
            final_stored_observed=data["final_selected_observed"],
            support_recomputed=False,
            physical_ID_truth=False,
            original_video_decoded=False,
            original_SL_cache_or_NPZ_read=False,
            GPU_or_fitting_used=False,
        ),
    )
    evidence.final_guard()
    out.final_guard()
    (output / "render.provenance.json").write_bytes(
        proof.json_bytes(
            dict(
                stage=STAGE,
                status="ok",
                inputs=evidence.paths,
                outputs=out.records,
            )
        )
    )
    print(str(target), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=WORK / STAGE)
    run(parser.parse_args().output)
