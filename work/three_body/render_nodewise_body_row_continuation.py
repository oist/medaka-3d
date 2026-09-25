"""Three completed same-horizon trajectories on four registered native photographs."""

from __future__ import annotations

import argparse
import io
from dataclasses import asdict
from pathlib import Path

import compare_nodewise_body_row_continuation as comparison
import numpy as np
import render_body_row_continuation as frozen

common, proof, require, WORK = (
    comparison.common,
    comparison.proof,
    comparison.require,
    comparison.WORK,
)
drawing = frozen.drawing
STAGE = "nodewise_body_row_ten_second_visual_comparison_v1"
FROZEN_SHA = "ff09bff96392c3445c61dbb24324897a4b37d6c91ecd1fa81e23a056a9d05af2"


def project_samples(samples, index, rig):
    comparison.validate_samples(samples)
    result = {}
    for arm in comparison.ARMS:
        pose = samples[arm + "_pose_xyz"][index]
        result[arm] = {v: rig.camera(v).project(pose) for v in drawing.VIEWS}
        require(
            all(p.shape == (3, 3, 2) and np.isfinite(p).all() for p in result[arm].values()),
            "finite three-body projections required",
        )
    return result


def draw_frame(samples, index, rig, context, evidence):
    frame = comparison.FRAMES[index]
    projections = project_samples(samples, index, rig)
    photos = []
    fig, axes = frozen.plt.subplots(3, 3, figsize=(18, 15), squeeze=False)
    try:
        for vi, view in enumerate(drawing.VIEWS):
            picture, points, reference = drawing.photograph_and_points(
                context, frame, view, evidence
            )
            photos.append(reference)
            lo, hi = drawing.visual.display_crop(
                picture, None, points, np.stack([projections[a][view] for a in comparison.ARMS])
            )
            for ai, arm in enumerate(comparison.ARMS):
                ax = axes[ai, vi]
                drawing.draw_panel(ax, picture, points, projections[arm][view])
                ax.set_xlim(lo[0], hi[0])
                ax.set_ylim(hi[1], lo[1])
                kind = str(samples[arm + "_proposal_kind"][index])
                flags = samples[arm + "_observed"][index].astype(int).tolist()
                ax.set_title(
                    f"{arm} | {view} | final kind={kind}\nstored direct M/C/T={flags}", fontsize=9
                )
        fig.suptitle(
            f"Native {frame} | All three paths end at83000 | Shared state at81340\n"
            "Open M circle / C square / T triangle: fitted pose. Orange +: saved SLEAP.\n"
            "S0/S1/S2: raw slots, NOT matched physical IDs. "
            "Lossless native PNG; same crop per camera. Stored support is not rescored.",
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        stream = io.BytesIO()
        fig.savefig(stream, format="png", dpi=160)
        return stream.getvalue(), photos, projections
    finally:
        frozen.plt.close(fig)


def run(result_root, result_sha, output):
    result_root, output = Path(result_root).resolve(), Path(output).resolve()
    require(
        result_root.is_relative_to(WORK) and output.is_relative_to(WORK) and not output.exists(),
        "workspace comparison and fresh output",
    )
    out, evidence, started = proof.Output(output), proof.Evidence(), proof.utc()
    try:
        evidence.add(Path(frozen.__file__), FROZEN_SHA)
        evidence.add(Path(comparison.__file__))
        evidence.add(Path(__file__))
        seal = drawing.read(result_root / "comparison.provenance.json", result_sha, evidence)
        require(
            seal["status"] == "ok" and seal["stage"] == comparison.STAGE, "successful comparison"
        )
        summary, plan, signs = [
            drawing.document(result_root, seal, n + ".json", evidence)
            for n in ("summary", "plan", "review_samples_signatures")
        ]
        require(
            summary["arms"] == list(comparison.ARMS)
            and summary["terminal_horizon"] == 83000
            and plan["all_three_paths_terminal_horizon"] == 83000
            and plan["same_original39_signatures_and_metadata"] is True,
            "same state/horizon",
        )
        require(
            plan["candidate_references"] == summary["candidate_references"], "method references"
        )
        own_comparer = str(Path(comparison.__file__).resolve())
        require(
            plan["source_sha256"].get(own_comparer)
            == evidence.paths[own_comparer]
            == seal["inputs"].get(own_comparer),
            "comparison implementation differs",
        )
        path = common.bound_member(result_root, seal, "review_samples.npz", evidence)
        with np.load(path, allow_pickle=False) as archive:
            samples = {k: archive[k] for k in archive.files}
        require(common.signature(samples) == signs, "15 sample signatures differ")
        comparison.validate_samples(samples)
        new_reference = summary["candidate_references"]["nodewise_body_row"]
        variant = Path(new_reference["root"]).resolve()
        require(variant.is_relative_to(WORK), "registered new producer root")
        tracking = drawing.read(
            variant / new_reference["seal_name"], new_reference["seal_sha256"], evidence
        )
        require(
            tracking["stage"] == comparison.NEW_STAGE and tracking["status"] == "ok",
            "successful new path",
        )
        shared = plan["original_binding"]["inputs"]["seed"]["SL_and_calibration_sha256"]
        primary = drawing.PRIMARY_ROOT
        primary_seal = drawing.read(
            primary / "render.provenance.json", drawing.PRIMARY_SEAL_SHA, evidence
        )
        require(primary_seal["status"] == "ok", "registered native pictures")
        for p, digest in shared.items():
            require(
                tracking["inputs"].get(p) == primary_seal["inputs"].get(p) == digest,
                "image/SL/calibration lineage differs",
            )
            if Path(p).suffix == ".json":
                evidence.add(p, digest)
        folders = [Path(p).parent for p in shared if Path(p).name == "calibration_manifest.json"]
        require(len(folders) == 1, "one registered rig")
        rig = common.metrics.old.prior.load_rig(folders[0])
        require(
            plan["tracker_context"]["cameras"]
            == common.support.COMMON.normalize([asdict(c) for c in rig.cameras]),
            "camera context",
        )
        p = common.bound_member(primary, primary_seal, "source_and_frame_evidence.json", evidence)
        require(proof.sha(p) == drawing.PRIMARY_EVIDENCE_SHA, "saved pixel observation evidence")
        context = dict(
            primary_seal=primary_seal,
            records={r["frame"]: r for r in proof.read_json(p)["records"]},
        )
        cases = []
        for index, frame in enumerate(comparison.FRAMES):
            picture, photos, projections = draw_frame(samples, index, rig, context, evidence)
            target = output / f"frame_{frame}.png"
            target.write_bytes(picture)
            out.records[target.name] = dict(
                path=str(target), sha256=proof.sha(target), size_bytes=target.stat().st_size
            )
            cases.append(
                dict(
                    frame_idx=frame,
                    photographs=photos,
                    projections={
                        a: {v: p.tolist() for v, p in views.items()}
                        for a, views in projections.items()
                    },
                )
            )
        out.write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                arms=list(comparison.ARMS),
                frames=list(comparison.FRAMES),
                cases=cases,
                comparison_sha256=result_sha,
                physical_ID_truth=False,
                original_video_decoded=False,
                masks_read=False,
                support_recalculated=False,
            ),
        )
        evidence.final_guard()
        out.final_guard()
        receipt = dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=out.records,
        )
    except BaseException as exc:
        receipt = dict(
            stage=STAGE,
            status="failed",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=out.records,
            error=repr(exc),
        )
        (output / "render.provenance.json").write_bytes(proof.json_bytes(receipt))
        raise
    (output / "render.provenance.json").write_bytes(proof.json_bytes(receipt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--comparison-sha", required=True)
    parser.add_argument("--output", type=Path, default=WORK / STAGE)
    args = parser.parse_args()
    run(args.comparison, args.comparison_sha, args.output)
