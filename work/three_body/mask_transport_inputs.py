"""Sealed native200 observations and pose-independent fit/evaluation pixels.

The constructor seed is the original registration only. No saved reference pose
or future winner enters this reader. The original offline background is retained.
This CPU preparation performs no tracker/model/GPU execution.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import track_semantic_node200 as prepared

WORK, REPO = prepared.WORK, prepared.REPO
sys.path[:0] = [str(REPO / "src"), str(REPO / "labs/20260909_three_body")]
import torch  # noqa: E402
from continuous_tracker import _canonical  # noqa: E402
from observations import FrameObservation, sample_pixels  # noqa: E402
from renderer import RefractiveFishRenderer  # noqa: E402
from score_temporal_disjoint_pixels import unused_samples  # noqa: E402

from medaka3d.calib import load_rig  # noqa: E402

base = prepared.base
require = prepared.require
FIRST, STOP, COUNT = 333519, 333719, 200
VIEWS = prepared.VIEWS
PREPARATION = prepared.PREPARATION
PREPARATION_SHA = "153c9f5f4a316f193b609afd2113b875cf1a62380dca4b80a6b27834b53afc15"
PLAN = WORK / "MASK_TRANSPORT_200_IMPLEMENTATION_PLAN.md"
PLAN_SHA = "2fe9c7681cbefd6b1f79c0d49f39e07ff7bc3867c2b5559029b513ea5475b130"
STAGE = "mask_transport200_fixed_samples_v1"
SOURCE_PINS = {
    "work/three_body/track_semantic_node200.py": (
        "219703a8101aac1bc69e60ee061ffbb5875c41027df6ceb337e4a1f8f029bcca"
    ),
    "labs/20260909_three_body/observations.py": (
        "f1efcd27b31e63ce693229c9e765152acfc7984b3b8839f1792bcc2efd9cb5df"
    ),
    "labs/20260909_three_body/continuous_tracker.py": (
        "91a811cab27a8a26e605aa00159fdb2a2b450bbcca1a2505e49543f58ef7c7e2"
    ),
    "labs/20260909_three_body/renderer.py": (
        "dfb585e9d51214a2a3c9447d4bc81da8af80d3451e8fc3ef7a44b1a841732e1c"
    ),
    "work/three_body/score_temporal_disjoint_pixels.py": (
        "8a86b3942d93b5ed506ac85cb3520fa9ccd51a2a11c55201fd1884e4e5c31d47"
    ),
    "src/medaka3d/calib.py": "60849004cb53febe69f437e7afa8b18cb2b65213ba76ce7c921f7022d2be4274",
}


def load_common(evidence):
    for name, digest in SOURCE_PINS.items():
        evidence.add(REPO / name, digest)
    evidence.add(Path(__file__).resolve())
    evidence.add(PLAN, PLAN_SHA)
    context = prepared.load_context(evidence, PREPARATION_SHA)
    members = context["members"]
    seed, original, backgrounds, contract = (
        base.bound_json(PREPARATION, members, name)
        for name in (
            "original_seed.json",
            "original_context.json",
            "backgrounds.json",
            "seed_contract.json",
        )
    )
    initial = np.asarray(seed["pose_xyz"], np.float64)
    require(
        initial.shape == (3, 3, 3)
        and np.isfinite(initial).all()
        and base.signature(initial)
        == contract["initial_pose_signature"]
        == original["initial_pose"]
        and seed["frame"] == FIRST
        and seed["initialization_count"] == 1
        and seed["later_baseline_input"] is False,
        "original registered constructor seed differs",
    )
    shared = context["plan"]["shared_SL_calibration_sha256"]
    calibration = {Path(p): digest for p, digest in shared.items() if Path(p).suffix == ".json"}
    require(
        len(calibration) == 4 and len({p.parent for p in calibration}) == 1,
        "exact four calibration inputs required",
    )
    for path, digest in calibration.items():
        evidence.add(path, digest)
    rig = load_rig(next(iter(calibration)).parent)
    require(
        original["cameras"] == json.loads(_canonical([dataclasses.asdict(c) for c in rig.cameras]))
        and original["views"] == list(rig.views) == list(VIEWS)
        and original["world_units"] == rig.world_units == "mm",
        "refractive rig differs",
    )
    require([b["view"] for b in backgrounds] == list(VIEWS), "three ordered backgrounds required")
    reader = Prepared200(context, backgrounds)
    return dict(
        reader=reader,
        rig=rig,
        initial=initial,
        original_context=original,
        seed=seed,
        context=context,
        background_rows=backgrounds,
    )


class Prepared200:
    """Reusable readonly input index; each traversal owns newly decoded frames."""

    def __init__(self, context, background_rows, *, root=PREPARATION):
        self.root, self.members = Path(root), context["members"]
        prepared.validate_native_rows(context["natives"])
        self.rows = {(r["frame_idx"], r["view"]): r for r in context["natives"]}
        self.backgrounds = {}
        for row in background_rows:
            raw = self.read(row["path"], row["encoded_sha256"])
            image = decode_bgr(raw, row["decoded_bgr_signature"]["sha256"])
            image.setflags(write=False)
            self.backgrounds[row["view"]] = image
        require(set(self.backgrounds) == set(VIEWS), "all three backgrounds required")

    def read(self, relative, expected=None):
        path = prepared.bound_path(self.root, self.members, relative, expected)
        return prepared.checked_bytes(path, self.members)

    def at(self, native):
        require(type(native) is int and FIRST <= native < STOP, "native outside fixed200")
        images, masks, detections = {}, {}, {}
        for view in VIEWS:
            row = self.rows[native, view]
            images[view] = decode_bgr(
                self.read(row["png_path"], row["png_sha256"]), row["bgr_sha256"]
            )
            arrays = prepared.arrays_from_bytes(
                self.read(row["observation_path"], row["observation_sha256"])
            )
            validate_observation(arrays, row)
            masks[view] = arrays["masks"] if bool(arrays["mask_available"]) else None
            detections[view] = arrays["detections"]
        return FrameObservation(native, detections, images, masks, self.backgrounds)

    def __iter__(self):
        for native in range(FIRST, STOP):
            yield self.at(native)


def decode_bgr(raw, digest):
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    require(
        image is not None and image.dtype == np.uint8 and image.shape == (1200, 1200, 3),
        "native1200 BGR required",
    )
    require(hashlib.sha256(image.tobytes()).hexdigest() == digest, "native BGR bytes differ")
    return image


def validate_observation(arrays, row):
    require(
        set(arrays) == set(row["observation_array_signatures"]),
        "all prepared observation fields required",
    )
    require(
        prepared.delta.signatures(arrays) == row["observation_array_signatures"],
        "prepared observation bytes differ",
    )
    native, available = arrays["frame_idx"], arrays["mask_available"]
    require(
        native.dtype == np.int64 and native.shape == () and int(native) == row["frame_idx"],
        "observation native mismatch",
    )
    require(
        available.dtype == bool
        and available.shape == ()
        and bool(available) == row["mask_available"],
        "mask availability differs",
    )
    masks, points = arrays["masks"], arrays["detections"]
    require(
        masks.dtype == bool
        and masks.shape == (row["mask_count"], 1200, 1200)
        and base.signature(masks)["sha256"] == row["mask_sha256"],
        "all-N masks differ",
    )
    require(
        points.dtype == np.float32
        and points.ndim == 3
        and points.shape[1:] == (3, 2)
        and base.signature(points)["sha256"] == row["detections_sha256"],
        "untracked SLEAP node bytes differ",
    )


def fixed_samples(frame, rig):
    fit = sample_pixels(frame, "detectron2", 384, 384, 256)
    fitted = np.stack([fit.pixels[v] for v in VIEWS])
    pixels, target, weights, overlap, audit = unused_samples(frame, fitted)
    require(not overlap.any(), "unused samples overlap fit coordinates")
    renderer = RefractiveFishRenderer(
        rig, dict(zip(VIEWS, pixels, strict=True)), device="cpu", dtype=torch.float64
    )
    valid = renderer.ray_valid.cpu().numpy().copy()
    arrays = dict(
        frame_idx=np.asarray(frame.frame_idx, np.int64),
        fit_pixels=fitted,
        fit_target=fit.target,
        fit_weights=fit.weights,
        fit_foreground=fit.foreground,
        unused_pixels=pixels,
        unused_target=target,
        unused_weights=weights,
        unused_fit_overlap=overlap,
        unused_ray_valid=valid,
    )
    for vi, row in enumerate(audit):
        row["invalid_ray_draws"] = int((~valid[vi]).sum())
        row["unused_draw_count"] = len(pixels[vi])
        row["unused_unique_total"] = len(np.unique(pixels[vi], axis=0))
    return arrays, audit


def prepare(destination):
    destination = Path(destination).resolve()
    require(destination.is_relative_to(WORK) and not destination.exists(), "new workspace output")
    output, evidence, captured = base.Output(destination), base.Evidence(), {}
    start, started, counts = time.perf_counter(), base.utc(), []
    old_cv = cv2.getNumThreads()
    try:
        cv2.setNumThreads(32)
        torch.set_num_threads(2)
        require(not torch.cuda.is_initialized(), "preparation must not initialize CUDA")
        common = load_common(evidence)
        prepared.snapshot_sources(evidence, output, captured)
        output.write(
            "plan.json",
            dict(
                schema=STAGE,
                native_interval=[FIRST, STOP],
                preparation_sha256=PREPARATION_SHA,
                implementation_plan_sha256=PLAN_SHA,
                fit_counts=[384, 384, 256],
                unused_counts=[96, 96, 64],
                opencv_threads=32,
                torch_threads=2,
                before_new_fit=True,
                new_model=False,
                new_fit=False,
                new_GPU=False,
                spatial_resampling_same_D2=True,
                independent_ID_truth=False,
            ),
        )
        output.write("original_seed.json", common["seed"])
        output.write("original_context.json", common["original_context"])
        for frame in common["reader"]:
            arrays, audit = fixed_samples(frame, common["rig"])
            output.write(f"samples/native_{frame.frame_idx}.npz", arrays)
            counts.extend(audit)
        require(len(counts) == 600, "all native200 by3 samples required")
        output.write("sampling_audit.json", counts)
        summary = dict(
            schema=STAGE,
            status="ok",
            frames=200,
            views=3,
            native_png_decodes=600,
            fit_pixel_draws=614400,
            unused_pixel_draws=153600,
            overlap_draws=sum(r["overlap_count"] for r in counts),
            invalid_ray_draws=sum(r["invalid_ray_draws"] for r in counts),
            unused_unique_per_frame_view_total=sum(r["unused_unique_total"] for r in counts),
            new_model=False,
            new_fit=False,
            CUDA_initialized=torch.cuda.is_initialized(),
            elapsed_before_guards_seconds=time.perf_counter() - start,
        )
        require(
            not summary["CUDA_initialized"] and not summary["overlap_draws"], "preparation scope"
        )
        output.write("summary.json", summary)
        prepared.snapshot_sources(evidence, output, captured)
        evidence.final_guard()
        output.final_guard()
        seal = dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=base.utc(),
            inputs=evidence.paths,
            outputs=output.records,
            params=dict(all_guards_passed=True),
        )
    except BaseException as error:
        seal = dict(
            stage=STAGE,
            status="failed",
            started_at=started,
            finished_at=base.utc(),
            inputs=evidence.paths,
            outputs=output.records,
            error_type=type(error).__name__,
            error=str(error),
        )
        with (destination / "samples.provenance.json").open("xb") as f:
            f.write(base.json_bytes(seal))
        raise
    finally:
        cv2.setNumThreads(old_cv)
    with (destination / "samples.provenance.json").open("xb") as f:
        f.write(base.json_bytes(seal))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output)), flush=True)


if __name__ == "__main__":
    main()
