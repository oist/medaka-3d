"""Prepare only the previously fixed 66524/top_cam/mask0 CPU smoke input."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

import dino_mask_partial_correspondence as cropper


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    work = Path(__file__).resolve().parent
    raw = work / "bounded_shape_raw_review_v1"
    out = work / "dino_first_crop_smoke_input_v1"
    assert not out.exists()
    seals = list(raw.glob("*.provenance.json"))
    assert len(seals) == 1
    seal_path = seals[0]
    assert sha(seal_path) == "d5fa256848a0298fc0d5f5cbca2e7b7a0d55867b5874f9159d5859c56ddcbd70"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    assert seal["status"] == "ok"
    inputs = {str(seal_path): sha(seal_path), str(Path(__file__).resolve()): sha(__file__)}
    for row in seal["outputs"].values():
        p = Path(row["path"]).resolve(strict=True)
        assert p.is_relative_to(raw) and p.stat().st_size == row["size"]
        assert sha(p) == row["sha256"]
        inputs[str(p)] = row["sha256"]
    for name, digest in {
        "dino_mask_partial_correspondence.py": "de487b2095c09c27be14e7fec92980d5a84f8c0ec6bf08017d7342aab145e400",
        "DINO_PARTIAL_PROTOCOL.md": "2fd1870cf23c2c8395d45622a5c198dd8060c46b7a68c71d4fe07417edd87f40",
    }.items():
        assert sha(work / name) == digest
        inputs[str(work / name)] = digest
    names = ["raw_66524_top_cam.png", "observation_66524_top_cam.npz", "native_observation_evidence.json"]
    assert all(str(raw / name) in inputs for name in names)
    evidence = json.loads((raw / names[2]).read_text(encoding="utf-8"))
    rows = [r for r in evidence if r["frame_idx"] == 66524 and r["view"] == "top_cam"]
    assert len(rows) == 1
    observation = rows[0]
    bgr = cv2.imread(str(raw / names[0]), cv2.IMREAD_COLOR)
    assert bgr.dtype == np.uint8 and bgr.shape == (1200, 1200, 3)
    assert hashlib.sha256(bgr.tobytes()).hexdigest() == observation["bgr_sha256"]
    with np.load(raw / names[1], allow_pickle=False) as z:
        masks = z["masks"]
        assert bool(z["mask_available"]) and masks.dtype == bool
    assert masks.shape == (observation["mask_count"], 1200, 1200)
    assert hashlib.sha256(masks.tobytes()).hexdigest() == observation["mask_sha256"]
    prepared = cropper.prepare_mask_crop(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), masks[0])
    assert prepared["crop_status"] == "ready", "fixed first mask unavailable; do not substitute"
    out.mkdir()
    arrays = {k: v for k, v in prepared.items() if isinstance(v, np.ndarray)}
    metadata = {k: v for k, v in prepared.items() if not isinstance(v, np.ndarray)}
    np.savez_compressed(out / "crop.npz", **arrays)
    for condition in ("raw", "masked"):
        np.save(out / (condition + "_normalized_chw.npy"), arrays[condition + "_normalized_chw"], allow_pickle=False)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    assert all(sha(path) == digest for path, digest in inputs.items())
    outputs = {p.name: dict(path=str(p), sha256=sha(p), size=p.stat().st_size) for p in out.iterdir()}
    (out / "preparation.provenance.json").write_text(json.dumps(dict(
        stage="dino_first_crop_smoke_input_v1", status="ok", inputs=inputs, outputs=outputs,
        fixed_frame=66524, fixed_view="top_cam", fixed_mask_row=0,
        source_mask_count=4, cropped_mask_count=1, model_load=False, GPU_used=False,
        purpose="CPU runtime smoke only; not the fixed71 appearance experiment",
    ), indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(metadata=metadata, outputs=outputs, seal_sha256=sha(out / "preparation.provenance.json"))))


if __name__ == "__main__":
    main()
