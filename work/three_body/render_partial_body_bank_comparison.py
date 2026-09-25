"""Display three saved within-bank path rankings on registered photographs.

Read only successful reranking JSON, saved visual evidence and PNG/JPEG files.
All body markers are open proposals: no strict support, tracking or ID inference.
Old negative-control JPEGs are lossy context, not exact source BGR pixels.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import render_fast_tail_five_minute_id_transition as frozen

proof, require, prior = frozen.proof, frozen.require, frozen.prior
previous, visual = frozen.previous, frozen.visual
WORK, REPO, VIEWS = frozen.WORK, frozen.REPO, frozen.VIEWS
STAGE = "partial_body_saved_bank_visual_comparison_v1"
COMPARISON_STAGE = "partial_body_saved_bank_comparison_v1"
COMPARISON_SOURCE_SHA = "7e069ea6b727c05c951f1e40d70ea8ad54276db2a0a344fa8b4dca534f7d2dd8"
MODES = ("baseline", "row_only", "row_plus_mask")
PRIMARY = (82720, 82727, 82728, 82750)
NEGATIVE = (90141, 90258, 90259)
ARMS = ("sparse_instance_rescore", "expanded_depth_instance_rescore")
PRIMARY_ROOT = WORK / "depth_identity_transition_raw_review_v1"
PRIMARY_SEAL_SHA = "9f3721806523e5b656f317684ffffa19cc5c08b94341a5e353d1bd5af383deea"
PRIMARY_EVIDENCE_SHA = "b0176b780b77c7205911fae1839aa8771bece0bb5fa368ce4afac902f4435f74"
OLD_ROOT = WORK / "fast_tail_five_minute_id_transition_raw_review_v2"
OLD_SEAL_SHA = "0063b4be2dd9b37200d279c9042dff4862e38d84f9c01446f6559beb146a1924"
OLD_EVIDENCE_SHA = "5405ce02b34cd4c1673487f7427d101ccf333cfef869dc740a9acb426ea70d41"
JPEG_ROOT = WORK / "fast_tail_transition_raw_review_pack_v1"
JPEG_SEAL_SHA = "8622bd5655ecc34058c5e3a7a2fe64e7ff357a3d9eb508cfca4e258405fb0c7f"
PINS = {
    "render_fast_tail_five_minute_id_transition.py": (
        "169c8a7dcd7e3ab932412a968ddae4e04574d99c4e517ee986cdbe4aa544236e"
    ),
    "render_expanded_depth_10s_review.py": (
        "36eea92fa9126a65495b055a0f9046d88de31382d36535ba45478651b5933e25"
    ),
    "render_instance_rescore_10s_review.py": (
        "0d2e01e4b6bcc99bbeef3e85d7189daa0fe2b6d9623e1efd85b2984e83d13a7d"
    ),
    "compare_temporal_point64_v2.py": (
        "2461f383d7c69223c98cb9371058373361a7fb30a409a37d617b60fc2db53790"
    ),
}
COLORS = ("#00e5ff", "#ffe600", "#ff40e5")
LABELS, MARKERS = ("M", "C", "T"), ("o", "s", "^")


def edge(seal, path, digest):
    require(seal["inputs"].get(str(Path(path).resolve())) == digest, "consumed input differs")


def read(path, digest, evidence):
    evidence.add(path, digest)
    return proof.read_json(path)


def document(root, seal, name, evidence):
    return proof.read_json(previous.member(root, seal, name, evidence))


def source_code(plan, seal, evidence):
    require(proof.valid_sha(COMPARISON_SOURCE_SHA), "reranking source must be frozen")
    code = dict(plan["source_code"])
    caller = str(WORK / "compare_partial_body_saved_banks.py")
    require(code.get(caller) == COMPARISON_SOURCE_SHA, "reranking source differs")
    for path, digest in code.items():
        edge(seal, path, digest)
        evidence.add(Path(path), digest)
    for name, digest in PINS.items():
        path = str(WORK / name)
        require(code.get(path) == digest, "drawing helper differs from scored sources")
    own = str(Path(__file__).resolve())
    code[own] = evidence.add(Path(own))
    return code


def observation_document(root, seal_sha, doc_sha, comparison, evidence):
    path = root / "render.provenance.json"
    edge(comparison, path, seal_sha)
    seal = read(path, seal_sha, evidence)
    require(seal["status"] == "ok", "successful image preparation required")
    name = "source_and_frame_evidence.json"
    edge(comparison, root / name, doc_sha)
    require(seal["outputs"][name]["sha256"] == doc_sha, "saved observation output differs")
    doc = document(root, seal, name, evidence)
    require(
        doc["status"] == "ok" and doc["physical_ID_truth"] is False, "observation scope differs"
    )
    return seal, doc


def case_key(case):
    return case["arm"], case["frame_idx"]


def selected_case(case, summary):
    require(case_key(case) == case_key(summary), "selected case differs")
    require(set(case["selections"]) == set(MODES), "three ranking selections required")
    count = summary["candidates"]
    require(type(count) is int and count > 0, "candidate count differs")
    selections = {}
    for mode in MODES:
        row, rank = case["selections"][mode], summary["rankings"][mode]
        require(
            row["support_not_recomputed"] is True
            and row["plan_is_assigned_views_not_direct_support"] is True,
            "proposed plan/support scope differs",
        )
        index = row["candidate_index"]
        require(type(index) is int and 0 <= index < count, "selected index invalid")
        for field in ("path_minimum_indices", "local_minimum_indices"):
            indices = row[field]
            require(indices == rank[field] and len(indices) > 0, "saved minimum set differs")
            require(
                all(type(i) is int and 0 <= i < count for i in indices)
                and indices == sorted(set(indices)),
                "minimum set invalid",
            )
        require(
            index == rank["path_order"][0] and index in row["path_minimum_indices"],
            "wrong path winner",
        )
        costs = np.asarray(
            [row[k] for k in ("path_cost", "local_cost", "parent_path_cost")], np.float64
        )
        require(
            np.isfinite(costs).all() and costs[0] == costs[2] + costs[1],
            "selected cost ledger differs",
        )
        require(
            row["best_path_cost"] == rank["best_path_cost"] == row["path_cost"]
            and row["best_local_cost"] == rank["best_local_cost"] <= row["local_cost"],
            "selected cost minima differ",
        )
        pose = np.asarray(row["pose_xyz"], np.float64)
        plan = np.asarray(row["candidate_plan"])
        require(pose.shape == (3, 3, 3) and np.isfinite(pose).all(), "selected pose invalid")
        require(
            plan.shape == (3, 3, 3) and plan.dtype.kind in "iu" and np.all(plan >= -1),
            "proposed plan invalid",
        )
        require(set(row["projections"]) == set(VIEWS), "three saved projections required")
        projections = {v: np.asarray(row["projections"][v], np.float64) for v in VIEWS}
        require(all(p.shape == (3, 3, 2) for p in projections.values()), "projection shape differs")
        for a in (pose, plan, *projections.values()):
            a.flags.writeable = False
        selections[mode] = dict(row=row, pose=pose, plan=plan, projections=projections)
    return dict(arm=case["arm"], frame=case["frame_idx"], selections=selections)


def load_context(root, digest, evidence):
    root = Path(root).resolve()
    require(root.is_relative_to(WORK) and proof.valid_sha(digest), "registered comparison required")
    seal = read(root / "comparison.provenance.json", digest, evidence)
    require(
        seal["stage"] == COMPARISON_STAGE and seal["status"] == "ok",
        "successful reranking required",
    )
    plan, summary, payload = [
        document(root, seal, n + ".json", evidence) for n in ("plan", "summary", "render_payload")
    ]
    require(
        plan["schema"] == summary["schema"] == COMPARISON_STAGE and summary["status"] == "ok",
        "reranking schema differs",
    )
    require(
        payload["schema"] == COMPARISON_STAGE + "_render_payload_v1"
        and payload["modes"] == list(MODES),
        "render payload differs",
    )
    expected_sources = [
        dict(root=str(root), seal_sha256=seal_sha, evidence_sha256=doc_sha, frames=list(frames))
        for root, seal_sha, doc_sha, frames in (
            (PRIMARY_ROOT, PRIMARY_SEAL_SHA, PRIMARY_EVIDENCE_SHA, PRIMARY),
            (OLD_ROOT, OLD_SEAL_SHA, OLD_EVIDENCE_SHA, NEGATIVE),
        )
    ]
    require(
        payload["original_observation_evidence"] == expected_sources
        and payload["physical_ID_truth"] is False
        and payload["raw_slot_labels_only"] is True
        and payload["new_trajectory_created"] is False,
        "render observation references or scope differ",
    )
    require(
        plan["fixed_parent_history"] is True
        and plan["physical_ID_truth"] is False
        and summary["new_trajectory_created"] is False
        and summary["baseline_bank_unchanged"] is True,
        "fixed-bank scope differs",
    )
    expected = {(a, f) for a in ARMS for f in PRIMARY} | {(ARMS[0], f) for f in NEGATIVE}
    by_key = {case_key(r): r for r in summary["cases"]}
    require(
        len(by_key) == len(summary["cases"]) == len(expected) and set(by_key) == expected,
        "all eleven cases required",
    )
    require(
        len(payload["cases"]) == len(expected)
        and {case_key(r) for r in payload["cases"]} == expected,
        "render cases differ",
    )
    cases = [selected_case(c, by_key[case_key(c)]) for c in payload["cases"]]
    primary_seal, primary_doc = observation_document(
        PRIMARY_ROOT, PRIMARY_SEAL_SHA, PRIMARY_EVIDENCE_SHA, seal, evidence
    )
    _old_seal, old_doc = observation_document(
        OLD_ROOT, OLD_SEAL_SHA, OLD_EVIDENCE_SHA, seal, evidence
    )
    jpeg_seal = read(JPEG_ROOT / "review.provenance.json", JPEG_SEAL_SHA, evidence)
    require(
        jpeg_seal["status"] == "ok"
        and jpeg_seal["stage"] == "fast_tail_transition_raw_review_pack_v1",
        "registered JPEG pack failed",
    )
    edge(jpeg_seal, OLD_ROOT / "render.provenance.json", OLD_SEAL_SHA)
    jpeg_summary = document(JPEG_ROOT, jpeg_seal, "summary.json", evidence)
    require(
        jpeg_summary["status"] == "ok"
        and jpeg_summary["review_clip_range"] == [90040, 90400]
        and jpeg_summary["consumed_record"] == old_doc["source_chunk"]
        and jpeg_summary["selection_render_sha256"] == OLD_SEAL_SHA,
        "JPEG context native source differs",
    )
    records = {}
    for doc, frames in ((primary_doc, PRIMARY), (old_doc, NEGATIVE)):
        selected = [r for r in doc["records"] if r["frame"] in frames]
        require(
            len(selected) == len(frames) and {r["frame"] for r in selected} == set(frames),
            "saved frame evidence differs",
        )
        records.update({r["frame"]: r for r in selected})
    return dict(
        cases=cases,
        records=records,
        primary_seal=primary_seal,
        jpeg_seal=jpeg_seal,
        jpeg_summary=jpeg_summary,
        source_code=source_code(plan, seal, evidence),
        comparison_sha256=digest,
        comparison_root=str(root),
        payload_reference=seal["outputs"]["render_payload.json"],
    )


def photograph_and_points(context, frame, view, evidence):
    observation = context["records"][frame]["observations"][view]
    points = np.asarray(observation["SLEAP_points"], np.float32)
    require(
        proof.signature(points) == observation["SL_points_signature"],
        "saved SL reconstruction differs",
    )
    require(points.ndim == 3 and points.shape[1:] == (3, 2), "SL node shape differs")
    lossless = frame in PRIMARY
    if lossless:
        name = f"native/frame_{frame}_{view}.png"
        require(
            observation["native_png"] == name and observation["canonical_video_frame"] == frame,
            "native PNG frame differs",
        )
        root, seal = PRIMARY_ROOT, context["primary_seal"]
    else:
        require(frame in NEGATIVE, "unregistered photograph frame")
        name, root, seal = f"frames/f{frame}_{view}.jpg", JPEG_ROOT, context["jpeg_seal"]
        require(
            context["jpeg_summary"]["native_bgr_sha256"][view][frame - 90040]
            == observation["bgr_sha256"],
            "JPEG source BGR lineage differs",
        )
    path = previous.member(root, seal, name, evidence)
    image = prior.cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), prior.cv2.IMREAD_COLOR)
    require(
        image is not None and image.dtype == np.uint8 and image.shape == (1200, 1200, 3),
        "photograph pixels invalid",
    )
    if lossless:
        require(
            prior.prepare_sequence.array_sha256(image) == observation["bgr_sha256"],
            "native PNG pixel SHA differs",
        )
    image.flags.writeable = points.flags.writeable = False
    return (
        image,
        points,
        dict(
            **seal["outputs"][name],
            view=view,
            frame=frame,
            source_bgr_sha256=observation["bgr_sha256"],
            exact_source_pixels=lossless,
            lossy_JPEG_context=not lossless,
            SL_points_signature=observation["SL_points_signature"],
        ),
    )


def draw_panel(ax, image, points, projected):
    ax.imshow(prior.cv2.cvtColor(image, prior.cv2.COLOR_BGR2RGB))
    for row, nodes in enumerate(points):
        finite = nodes[np.isfinite(nodes).all(1)]
        if len(finite):
            ax.plot(
                finite[:, 0], finite[:, 1], ls="none", marker="+", color="#ff9632", ms=5, alpha=0.8
            )
        if np.isfinite(nodes[0]).all():
            ax.annotate(
                f"SL{row}M",
                nodes[0],
                xytext=(3, -8),
                textcoords="offset points",
                color="#ff9632",
                fontsize=7,
                clip_on=True,
            )
    for slot, nodes in enumerate(projected):
        for left, right in ((0, 1), (1, 2)):
            if np.isfinite(nodes[[left, right]]).all():
                ax.plot(
                    nodes[[left, right], 0],
                    nodes[[left, right], 1],
                    color=COLORS[slot],
                    lw=1.5,
                    ls="--",
                )
        for xy, label, marker in zip(nodes, LABELS, MARKERS, strict=True):
            if np.isfinite(xy).all():
                ax.plot(
                    *xy,
                    marker=marker,
                    markerfacecolor="none",
                    markeredgecolor=COLORS[slot],
                    color=COLORS[slot],
                    ms=6,
                    mew=1.4,
                )
                ax.annotate(
                    f"S{slot}{label}",
                    xy,
                    xytext=(4, 4),
                    textcoords="offset points",
                    color=COLORS[slot],
                    fontsize=8,
                    clip_on=True,
                )


def render(case, images, points):
    frame = case["frame"]
    photo_kind = "lossless native PNG" if frame in PRIMARY else "LOSSY JPEG CONTEXT; not exact BGR"
    fig, axes = prior.plt.subplots(3, 3, figsize=(18, 17), dpi=160)
    try:
        fig.suptitle(
            f"Native {frame} ({frame / 166:.6f} source seconds) | {case['arm']} | {photo_kind}\n"
            "Rows: original baseline / row only / row + mask; "
            "each row selects best saved fixed-parent PATH cost.\n"
            "S0 cyan / S1 yellow / S2 magenta are raw body slots. "
            "M circle / C square / T triangle.\n"
            "ALL open markers and dashed lines are PROPOSALS; strict support was not recomputed. "
            "SL is detector output.\n"
            "Fixed fitted pool and original parent histories; "
            "no new continuous trajectory, sex or physical-ID truth.",
            fontsize=10,
        )
        crops = {}
        for vi, view in enumerate(VIEWS):
            proposals = [case["selections"][mode]["projections"][view] for mode in MODES]
            lo, hi = visual.display_crop(images[view], None, points[view], proposals)
            crops[view] = dict(lower_xy=lo.tolist(), upper_xy=hi.tolist())
            for ri, mode in enumerate(MODES):
                selected, ax = case["selections"][mode], axes[ri, vi]
                draw_panel(ax, images[view], points[view], selected["projections"][view])
                row = selected["row"]
                ax.set_title(
                    f"{view} | {mode} candidate {row['candidate_index']}\n"
                    f"path={row['path_cost']:.6g}; exact minima={row['path_minimum_indices']}",
                    fontsize=9,
                )
                ax.set_xlim(float(lo[0]), float(hi[0]))
                ax.set_ylim(float(hi[1]), float(lo[1]))
                ax.tick_params(labelsize=7)
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        stream = io.BytesIO()
        fig.savefig(stream, format="png")
        return stream.getvalue(), crops
    finally:
        prior.plt.close(fig)


def run(root, digest, output):
    output = Path(output).resolve()
    require(output.is_relative_to(WORK) and not output.exists(), "fresh workspace output required")
    require(
        "torch" not in sys.modules or not sys.modules["torch"].cuda.is_initialized(), "CPU only"
    )
    out, evidence, started = proof.Output(output), proof.Evidence(), proof.utc()
    try:
        context = load_context(root, digest, evidence)
        photos, points, photo_records = {}, {}, []
        for frame in (*PRIMARY, *NEGATIVE):
            photos[frame], points[frame] = {}, {}
            for view in VIEWS:
                image, sl, record = photograph_and_points(context, frame, view, evidence)
                photos[frame][view], points[frame][view] = image, sl
                photo_records.append(record)
        out.write(
            "plan.json",
            dict(
                schema=STAGE,
                comparison_root=context["comparison_root"],
                comparison_sha256=digest,
                render_payload=context["payload_reference"],
                source_code=context["source_code"],
                cases=[dict(arm=c["arm"], frame=c["frame"]) for c in context["cases"]],
                modes=list(MODES),
                input_photographs=photo_records,
                photographs_decoded=21,
                original_video_reads=0,
                original_SLEAP_reads=0,
                mask_archive_reads=0,
                candidate_archive_reads=0,
                support_recomputed=False,
                physical_ID_truth=False,
                proposed_plans_are_observed=False,
                D2_contours_displayed=False,
            ),
        )
        records = []
        for case in context["cases"]:
            data, crops = render(case, photos[case["frame"]], points[case["frame"]])
            name = f"{case['arm']}_f{case['frame']}.png"
            record = out.write(name, data)
            records.append(
                dict(
                    arm=case["arm"],
                    frame=case["frame"],
                    image=record,
                    crops=crops,
                    displayed_selections={m: c["row"] for m, c in case["selections"].items()},
                )
            )
            print(f"Rendered {case['arm']} native {case['frame']}", flush=True)
        out.write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                cases=records,
                contact_sheets=len(records),
                physical_ID_truth=False,
                new_trajectory_created=False,
                support_recomputed=False,
                negative_photographs_are_lossy=True,
                fixed_parent_candidate_ranking_only=True,
            ),
        )
        evidence.final_guard()
        out.final_guard()
        require(
            "torch" not in sys.modules or not sys.modules["torch"].cuda.is_initialized(),
            "CUDA initialized",
        )
        seal = dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=out.records,
        )
    except BaseException as error:
        seal = dict(
            stage=STAGE,
            status="failed",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=out.records,
            error=repr(error),
        )
        (output / "render.provenance.json").write_bytes(proof.json_bytes(seal))
        raise
    (output / "render.provenance.json").write_bytes(proof.json_bytes(seal))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--comparison-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.comparison_root, args.comparison_sha, args.output)


if __name__ == "__main__":
    main()
