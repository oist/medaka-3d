"""Fixed six-frame raw/D2/SLEAP/projection contact sheets for shape diagnosis.

Visual evidence only. Native decoded BGR/masks must match the consumed chunk's
hashes. No fit, ID label, confidence clearing, new segmentation or GPU is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import audit_bounded_first_chunk as interim
import cv2
import matplotlib
import numpy as np
import prepare_bounded_mouth_fit_packets as packets

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from medaka3d.provenance import provenance
from medaka3d.types import VIEWS

checks, REPO = packets.checks, packets.REPO
STAGE = "bounded_shape_raw_review_v1"
FRAMES = (66524, 66600, 66628, 66703, 66800, 66908)
AUDIT = REPO / "work/three_body/bounded_mouth_first_chunk_interim_audit_v1"
AUDIT_SHA = "08a7440cb5c851c4a97d98d1d67f665f6762a89c524c2ba87bfa7c0245656f04"
PACKET = REPO / "work/three_body/bounded_mouth_local_fit_packets_33_v1"
PACKET_SHA = "c1b4fc4b0f369f7f4a376f8bcd4a5b1736ad42efdcc0101118f9a407243da080"


def run(output):
    output = Path(output).resolve()
    checks.require(output.is_relative_to(REPO / "work") and not output.exists(), "new work output")
    code = packets.comparison.source_registry.local_code()
    hashes = dict(code)
    output.mkdir(parents=True)
    with provenance(STAGE, output, name="review") as record:

        def emit(name, value=None):
            path = output / name
            if value is not None:
                path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
            record.output(name, path, sha256=checks.sha(path))

        emit(
            "plan.json",
            dict(
                frames=FRAMES,
                views=VIEWS,
                crop_rule="union of S0/S2 old/new projected nodes +40px",
                raw_native_size=[1200, 1200],
                fit=False,
                new_ID_truth=False,
                frame_selection=(
                    "diagnosed divergence, mouth change, gap onset, old/middle/new recovery"
                ),
            ),
        )
        interim.bind(AUDIT / "audit.provenance.json", AUDIT_SHA, hashes)
        audit = checks.verify(AUDIT, "audit.provenance.json", interim.STAGE, hashes)
        summary = checks.read(AUDIT / "summary.json")
        packets.prior.bind(PACKET / "preparation.provenance.json", PACKET_SHA, hashes)
        prepared = checks.verify(PACKET, "preparation.provenance.json", packets.STAGE, hashes)
        meta = checks.read(PACKET / "metadata.json")
        checks.require(
            meta["checkpoint_binding"]["original_binding"] == summary["shared_input_binding"],
            "raw preparation and same68060 candidate input differ",
        )
        old_audit_path = packets.prior.AUDIT / "audit.provenance.json"
        packets.prior.audit_input(prepared, old_audit_path, hashes)
        checks.require(hashes[str(old_audit_path)] == packets.prior.AUDIT_SHA, "raw audit changed")
        old_audit = checks.read(old_audit_path)
        traces = {}
        for name in ("directional", "bounded"):
            path = AUDIT / f"{name}_same68060_candidate.npz"
            checks.require(str(path) in hashes, "unsealed candidate")
            with np.load(path, allow_pickle=False) as data:
                checks.require(
                    np.array_equal(data["frame_idx"], np.arange(66400, 68060)), "horizon"
                )
                traces[name] = {k: data[k].copy() for k in ("pose_xyz", "observed", "sources")}
        for path, digest in meta["checkpoint_binding"]["original_binding"]["inputs"]["seed"][
            "SL_and_calibration_sha256"
        ].items():
            packets.prior.audit_input(prepared, path, hashes)
            interim.input_member(audit, path, hashes)
            checks.require(hashes[str(Path(path).resolve())] == digest, "SL/calibration differs")
        rig = packets.load_rig(meta["calibration_directory"])
        reader, chunk = packets.get_reader(meta["checkpoint_binding"], old_audit, hashes)
        checks.require(chunk == meta["source_chunk"], "consumed chunk differs")
        evidence = []
        for native in FRAMES:
            iterator = packets.iter_from(reader, native)
            try:
                frame = next(iterator)
                checks.require(frame.frame_idx == native, "raw native mismatch")
                evidence.extend(packets.observation_evidence(frame, reader))
                fig, axes = plt.subplots(3, 4, figsize=(18, 13), constrained_layout=True)
                projected = {
                    name: {
                        v: rig.camera(v).project(values["pose_xyz"][native - 66400]) for v in VIEWS
                    }
                    for name, values in traces.items()
                }
                for vi, view in enumerate(VIEWS):
                    image = frame.images_bgr[view]
                    checks.require(image.shape == (1200, 1200, 3), "native image size differs")
                    raw_name = f"raw_{native}_{view}.png"
                    checks.require(
                        cv2.imwrite(str(output / raw_name), image), "raw PNG write failed"
                    )
                    emit(raw_name)
                    points = np.concatenate(
                        [projected[n][view][[0, 2]].reshape(-1, 2) for n in traces]
                    )
                    points = points[np.isfinite(points).all(-1)]
                    checks.require(len(points) > 0, "no finite crop projection")
                    lo = np.maximum(np.floor(points.min(0) - 40), 0).astype(int)
                    hi = np.minimum(np.ceil(points.max(0) + 40), 1200).astype(int)
                    checks.require(np.all(hi > lo), "empty crop")
                    for col, title in enumerate(
                        ("Raw", "D2 instances + SLEAP rows", "Directional", "Bounded mouth")
                    ):
                        ax = axes[vi, col]
                        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                        ax.set_xlim(lo[0], hi[0])
                        ax.set_ylim(hi[1], lo[1])
                        ax.set_title(view + " | " + title)
                        ax.set_aspect("equal")
                        if col == 1:
                            masks = frame.masks[view]
                            for mask in [] if masks is None else masks:
                                contours, _ = cv2.findContours(
                                    mask.astype(np.uint8),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE,
                                )
                                for contour in contours:
                                    xy = contour[:, 0]
                                    ax.plot(
                                        xy[:, 0],
                                        xy[:, 1],
                                        color="#eeeeee",
                                        lw=0.8,
                                    )
                            for row, node in enumerate(frame.detections[view]):
                                color = plt.get_cmap("tab10")(row % 10)
                                ax.plot(node[:, 0], node[:, 1], ".:", color=color, lw=0.8, ms=3)
                                if np.isfinite(node[1]).all():
                                    ax.text(
                                        *node[1],
                                        "R" + str(row),
                                        color=color,
                                        fontsize=7,
                                        clip_on=True,
                                    )
                        elif col >= 2:
                            name = ("directional", "bounded")[col - 2]
                            for slot, nodes in enumerate(projected[name][view]):
                                color = ("#00e5ff", "#ffdf00", "#ff50ed")[slot]
                                ax.plot(nodes[:, 0], nodes[:, 1], "o-", color=color, ms=4, lw=1.2)
                                for ni, label in enumerate(("M", "C", "T")):
                                    if np.isfinite(nodes[ni]).all():
                                        ax.text(
                                            *nodes[ni],
                                            f"S{slot}{label}",
                                            color=color,
                                            fontsize=7,
                                            clip_on=True,
                                        )
                    np.savez_compressed(
                        output / f"observation_{native}_{view}.npz",
                        detections=frame.detections[view],
                        masks=(
                            frame.masks[view]
                            if frame.masks[view] is not None
                            else np.empty((0, 1200, 1200), bool)
                        ),
                        mask_available=np.asarray(frame.masks[view] is not None),
                        directional_projection=projected["directional"][view],
                        bounded_projection=projected["bounded"][view],
                        crop_xyxy=np.r_[lo, hi],
                    )
                    emit(f"observation_{native}_{view}.npz")
                fig.suptitle(
                    f"Native {native} | Same 68060 horizon | "
                    "S0 cyan / S1 yellow / S2 magenta | No ID truth"
                )
                name = f"comparison_{native}.png"
                fig.savefig(output / name, dpi=140)
                plt.close(fig)
                emit(name)
            finally:
                iterator.close()
        emit("native_observation_evidence.json", evidence)
        emit(
            "summary.json",
            dict(
                status="ok",
                frames=FRAMES,
                native_images=18,
                figures=6,
                native_BGR_and_mask_SHA_checks=18,
                horizon_stop=68060,
                biological_ID_truth=False,
                fits=0,
                GPU=False,
                masks_are_untracked_instances=True,
                crops_do_not_change_saved_native_images=True,
            ),
        )
        (output / "RESULT.md").write_text(
            "# 形状変化の原画照合\n\n"
            "同68060終端の新旧3D候補を、屈折校正で3視点へ投影した。各行は同じ原画・同じ切り出し。"
            "左から原画、D2個別輪郭とSLEAP検出行、胸部改良版、口候補追加版。"
            "白線はD2輪郭、R番号はSLEAP検出行で、両者の個体対応は未指定。S0/S1/S2は匿名slot。\n\n"
            + "\n\n".join(f"![native {n}](comparison_{n}.png)" for n in FRAMES)
            + "\n\n原画18枚のBGRとD2 maskを消費済みchunkのSHAに照合した。"
            "図は問題のある時刻から選んだ診断で、独立ID正解や1分全体の精度ではない。\n",
            encoding="utf-8",
        )
        emit("RESULT.md")
        checks.require(code == packets.comparison.source_registry.local_code(), "source changed")
        for path, digest in hashes.items():
            checks.require(checks.sha(path) == digest, "bound input changed")
            record.input(path, path, sha256=digest)
    print(json.dumps(dict(status="ok", native_images=18, figures=6)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
