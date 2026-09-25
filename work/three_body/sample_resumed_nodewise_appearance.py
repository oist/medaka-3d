"""Frozen, source-pixel-verified appearance proxy; no training or identity inference."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from medaka3d.calib import load_rig  # noqa: E402
from medaka3d.provenance import provenance  # noqa: E402

WORK = REPO / "work/three_body"
OUT = WORK / "resumed_nodewise_risk_review_v1"
ROOT = WORK / "nodewise_body_row_five_minute_resume_099600_v1"
PLAN_SHA = "78f77901a056f65e911e52b193a30784316730210ebeff3ccd679d3a3e0ed411"
VIEWS = ("top_cam", "side1_cam", "side2_cam")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run():
    assert not (OUT / "appearance_samples.npz").exists(), "fresh appearance output required"
    assert sha(OUT / "selection_plan.json") == PLAN_SHA
    plan = read(OUT / "selection_plan.json")
    config = plan["appearance"]
    with provenance("resumed_nodewise_appearance_samples_v1", OUT, name="appearance") as rec:
        bindings = {}

        def bind(name, path, expected=None):
            path = Path(path)
            digest = sha(path)
            if expected is not None and expected != digest:
                raise ValueError(f"SHA mismatch: {path}")
            bindings[str(path)] = digest
            rec.input(name, path, sha256=digest)
            return path

        bind("source", Path(__file__))
        bind("calib_source", REPO / "src/medaka3d/calib.py")
        bind("plan", OUT / "selection_plan.json", PLAN_SHA)
        seal = read(bind("tracking_seal", ROOT / "tracking.provenance.json"))
        assert seal["status"] == "ok" and seal["stage"] == ROOT.name
        for key in ("input_index.json", "candidate.npz"):
            bind(key, ROOT / key, seal["outputs"][key]["sha256"])
        index = read(ROOT / "input_index.json")
        registered = index["original_binding"]["inputs"]["seed"]["SL_and_calibration_sha256"]
        for path, digest in registered.items():
            if path.endswith(".json"):
                bind(Path(path).name, path, digest)
        calibration = next(Path(p).parent for p in registered if p.endswith("calibration_manifest.json"))
        rig = load_rig(calibration)
        frames = np.arange(*plan["interval"], config["sample_stride_frames"], dtype=np.int64)
        with np.load(ROOT / "candidate.npz", allow_pickle=False) as z:
            rows = np.searchsorted(z["frame_idx"], frames)
            assert np.array_equal(z["frame_idx"][rows], frames)
            poses, observed = z["pose_xyz"][rows], z["observed"][rows]
        manifests = {}
        for chunk in index["prefix_chunks"]:
            if chunk["stop"] <= frames[0]:
                continue
            path = chunk["output_hashes"]["manifest"]["path"]
            manifests[chunk["index"]] = read(bind(f"manifest_{chunk['index']}", path, chunk["manifest_sha256"]))
        samples = np.full((len(frames), 3, 3, 3), np.nan)
        pixel_hashes = []
        for vi, view in enumerate(VIEWS):
            source = index["prefix_chunks"][10]["raw_hardlinks"][view]
            video = Path(source["source_path"])
            # Consumed native BGR hashes verify actual content; source container hash is inherited.
            assert video.stat().st_size == source["bytes"]
            rec.input(f"video_{view}", video)
            rec.param(f"inherited_container_sha256_{view}", source["sha256"])
            pixels = rig.camera(view).project(poses)
            chest = pixels[:, :, 1]
            separation = np.linalg.norm(chest[:, :, None] - chest[:, None, :], axis=-1)
            separation[:, np.arange(3), np.arange(3)] = np.inf
            isolated = separation.min(axis=2) >= config["projected_isolation_px"]
            cap = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG)
            if not cap.isOpened():
                raise RuntimeError(f"cannot open source: {video}")
            try:
                for ti, frame in enumerate(frames):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
                    ok, bgr = cap.read()
                    if not ok or bgr.shape != (1200, 1200, 3):
                        raise RuntimeError(f"source decode failed {view}/{frame}")
                    digest = hashlib.sha256(bgr.tobytes()).hexdigest()
                    chunk_index = (int(frame) - 49800) // 1660
                    manifest = manifests[chunk_index]
                    expected = manifest["raw"][view]["frame_bgr_sha256"][int(frame) - manifest["frames"][0]]
                    if digest != expected:
                        raise RuntimeError(f"native BGR mismatch {view}/{frame}")
                    pixel_hashes.append(dict(view=view, frame=int(frame), sha256=digest))
                    for fish in range(3):
                        if not isolated[ti, fish] or not observed[ti, fish].all():
                            continue
                        mouth, chest_point, tail = pixels[ti, fish]
                        points = np.array([
                            mouth * .25 + chest_point * .75,
                            mouth * .1 + chest_point * .9,
                            chest_point,
                            chest_point * .8 + tail * .2,
                            chest_point * .6 + tail * .4,
                        ])
                        if not np.isfinite(points).all():
                            continue
                        points = np.rint(points).astype(int)
                        radius = config["patch_radius_px"]
                        if (points < radius).any() or (points >= 1200 - radius).any():
                            continue
                        patches = [bgr[y-radius:y+radius+1, x-radius:x+radius+1].reshape(-1, 3) for x, y in points]
                        samples[ti, fish, vi] = np.median(np.concatenate(patches), axis=0)
                    if ti % 100 == 0:
                        print(f"appearance {view}: {ti + 1}/{len(frames)}", flush=True)
            finally:
                cap.release()
        delta = np.sqrt(np.mean(np.diff(samples, axis=0) ** 2, axis=-1))
        flagged = (delta >= config["per_channel_rms_change_threshold"]).sum(axis=-1) >= config["minimum_agreeing_views"]
        path = OUT / "appearance_samples.npz"
        np.savez_compressed(path, frame_idx=frames, median_bgr=samples, delta_bgr_rms=delta, discontinuity=flagged)
        write(OUT / "appearance_source_frame_hashes.json", pixel_hashes)
        write(OUT / "appearance_summary.json", {
            "schema": "resumed_nodewise_appearance_samples_v1", "status": "ok",
            "sample_frames": len(frames), "decoded_view_frames": len(pixel_hashes),
            "valid_samples_by_slot_view": np.isfinite(samples).all(axis=-1).sum(axis=0).tolist(),
            "two_view_discontinuities": int(flagged.sum()),
            "source_container_fully_rehashed": False, "every_consumed_BGR_hash_verified": True,
            "limitations": ["Pixel color proxy depends on pose, lighting and projection; no learned appearance classifier.", "Sampling at 2 Hz cannot exclude between-sample changes."]})
        for path, digest in bindings.items():
            assert sha(path) == digest, f"input changed: {path}"
        for name in ("appearance_samples.npz", "appearance_source_frame_hashes.json", "appearance_summary.json"):
            rec.output(name, OUT / name, sha256=sha(OUT / name))
        rec.param("configuration", config)


if __name__ == "__main__":
    cv2.setNumThreads(2)
    run()
