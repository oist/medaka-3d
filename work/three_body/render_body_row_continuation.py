"""Show the two completed same-horizon paths on four registered native images."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import compare_body_row_continuation as comparison
import matplotlib.pyplot as plt
import numpy as np
import render_partial_body_bank_comparison as drawing

common = comparison.common
proof, require, WORK = comparison.proof, comparison.require, comparison.WORK
STAGE = "body_row_ten_second_visual_comparison_v1"


def run(result_root, result_sha, variant, output):
    result_root, variant, output = map(lambda p: Path(p).resolve(), (result_root, variant, output))
    require(all(p.is_relative_to(WORK) for p in (result_root, variant, output)), "workspace paths")
    require(not output.exists(), "fresh visual output required")
    out, evidence, started = proof.Output(output), proof.Evidence(), proof.utc()
    try:
        evidence.add(result_root / "comparison.provenance.json", result_sha)
        seal = proof.read_json(result_root / "comparison.provenance.json")
        require(
            seal["status"] == "ok" and seal["stage"] == comparison.STAGE, "successful comparison"
        )
        summary = proof.read_json(common.bound_member(result_root, seal, "summary.json", evidence))
        signs = proof.read_json(
            common.bound_member(result_root, seal, "review_samples_signatures.json", evidence)
        )
        path = common.bound_member(result_root, seal, "review_samples.npz", evidence)
        with np.load(path, allow_pickle=False) as archive:
            samples = {k: archive[k] for k in archive.files}
        require(common.signature(samples) == signs, "review sample signatures differ")
        tracking_path = variant / "tracking.provenance.json"
        evidence.add(tracking_path, summary["variant_tracking_sha256"])
        tracking = proof.read_json(tracking_path)
        require(tracking["status"] == "ok", "completed tracking required")
        plan = proof.read_json(common.bound_member(variant, tracking, "plan.json", evidence))
        shared = plan["original_binding"]["inputs"]["seed"]["SL_and_calibration_sha256"]
        primary_root = drawing.PRIMARY_ROOT
        evidence.add(primary_root / "render.provenance.json", drawing.PRIMARY_SEAL_SHA)
        primary_seal = proof.read_json(primary_root / "render.provenance.json")
        require(primary_seal["status"] == "ok", "successful native picture preparation")
        for p, digest in shared.items():
            require(
                tracking["inputs"].get(p) == primary_seal["inputs"].get(p) == digest,
                "SL/calibration lineage differs between pictures and tracking",
            )
            if Path(p).suffix == ".json":
                evidence.add(p, digest)
        folders = [Path(p).parent for p in shared if Path(p).name == "calibration_manifest.json"]
        require(len(folders) == 1, "one registered refractive rig")
        rig = common.metrics.old.prior.load_rig(folders[0])
        require(
            plan["tracker_context"]["cameras"]
            == common.support.COMMON.normalize([asdict(c) for c in rig.cameras]),
            "tracker calibration differs",
        )
        p = common.bound_member(
            primary_root, primary_seal, "source_and_frame_evidence.json", evidence
        )
        require(proof.sha(p) == drawing.PRIMARY_EVIDENCE_SHA, "registered picture observations")
        context = dict(
            primary_seal=primary_seal,
            records={r["frame"]: r for r in proof.read_json(p)["records"]},
        )
        rows, photos = [], []
        for index, frame in enumerate(comparison.FRAMES):
            projections, kinds = {}, {}
            for arm in ("control", "body_row"):
                require(samples[arm + "_frame_idx"][index] == frame, "native sample order differs")
                projections[arm] = {
                    v: rig.camera(v).project(samples[arm + "_pose_xyz"][index])
                    for v in drawing.VIEWS
                }
                kinds[arm] = str(samples[arm + "_proposal_kind"][index])
            fig, axes = plt.subplots(2, 3, figsize=(18, 10), squeeze=False)
            for vi, view in enumerate(drawing.VIEWS):
                picture, points, reference = drawing.photograph_and_points(
                    context, frame, view, evidence
                )
                photos.append(reference)
                lo, hi = drawing.visual.display_crop(
                    picture,
                    None,
                    points,
                    np.stack([projections[a][view] for a in ("control", "body_row")]),
                )
                for ai, arm in enumerate(("control", "body_row")):
                    ax = axes[ai, vi]
                    drawing.draw_panel(ax, picture, points, projections[arm][view])
                    ax.set_xlim(lo[0], hi[0])
                    ax.set_ylim(hi[1], lo[1])
                    flags = samples[arm + "_observed"][index].astype(int).tolist()
                    ax.set_title(
                        f"{arm} | {view} | final kind={kinds[arm]}\nstored direct M/C/T={flags}",
                        fontsize=9,
                    )
            fig.suptitle(
                f"Native {frame} | Both paths end at 83000 | Shared inherited state at 81340\n"
                "Open M circle / C square / T triangle: fitted pose. Orange +: SLEAP.\n"
                "S0/S1/S2 are raw slots, NOT independently matched physical IDs. "
                "Lossless native PNG.",
                fontsize=12,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.89))
            file = output / f"frame_{frame}.png"
            fig.savefig(file, dpi=160)
            plt.close(fig)
            out.records[file.name] = dict(
                path=str(file), sha256=proof.sha(file), size_bytes=file.stat().st_size
            )
            rows.append(
                dict(
                    frame_idx=frame,
                    image=out.records[file.name],
                    kinds=kinds,
                    projections={
                        a: {v: p.tolist() for v, p in d.items()} for a, d in projections.items()
                    },
                )
            )
        out.write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                frames=list(comparison.FRAMES),
                cases=rows,
                photographs=photos,
                physical_ID_truth=False,
                original_video_decoded=False,
                masks_read=False,
                support_recalculated=False,
                comparison_sha256=result_sha,
            ),
        )
        out.write(
            "plan.json",
            dict(
                schema=STAGE,
                source_sha256={
                    str(Path(__file__).resolve()): evidence.add(Path(__file__)),
                    str(Path(drawing.__file__).resolve()): evidence.add(Path(drawing.__file__)),
                },
                same_terminal_horizon=83000,
                actual_continuous_paths=True,
                fixed_bank_reweighting=False,
                physical_IDs_aligned=False,
            ),
        )
        evidence.final_guard()
        out.final_guard()
        result = dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=out.records,
        )
    except BaseException as exc:
        result = dict(
            stage=STAGE,
            status="failed",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=out.records,
            error=repr(exc),
        )
        (output / "render.provenance.json").write_bytes(proof.json_bytes(result))
        raise
    (output / "render.provenance.json").write_bytes(proof.json_bytes(result))
    print("Completed four-frame whole-row path rendering", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--comparison-sha", required=True)
    parser.add_argument("--variant", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=WORK / STAGE)
    args = parser.parse_args()
    run(args.comparison, args.comparison_sha, args.variant, args.output)
