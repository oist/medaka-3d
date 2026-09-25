"""CPU-only cache loader and native-pixel checks against existing staging."""

import hashlib
import json
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))
from prepare_sequence import VIDEO_NAMES, validate_capture, verify_assets  # noqa: E402


def run():
    cache = REPO / "work/three_body/source_cache"
    result_path = cache / "decode_audit.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    manifest_path = cache / "source_cache_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    provenance = json.loads((cache / "source_cache.provenance.json").read_text(encoding="utf-8"))
    if manifest["status"] != "verified" or provenance["status"] != "ok":
        raise ValueError("copy did not finish verification")
    assets, asset_manifest = verify_assets(cache)
    stage_root = REPO / "work/three_body/inputs"
    sequences = [
        ("interval_76620-77286", (76620, 76928, 77285)),
        ("transient_333519-333719", (333519, 333522, 333718)),
    ]
    cases = {}
    for sequence, frames in sequences:
        path = stage_root / sequence / "manifest.json"
        info = json.loads(path.read_text(encoding="utf-8"))
        for view in VIDEO_NAMES:
            for frame in frames:
                index = info["frames"].index(frame)
                cases[(view, frame)] = {
                    "expected_bgr_sha256": info["raw"][view]["frame_bgr_sha256"][index],
                    "staging_manifest": str(path.relative_to(REPO)),
                    "staging_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
    results, details = [], {}
    for view, filename in VIDEO_NAMES.items():
        path = cache / "medaka_202512/20251225_K_1_2_FCMC" / view / filename
        cap = cv2.VideoCapture(str(path))
        try:
            details[view] = validate_capture(cap, path)
            for (case_view, frame), expected in cases.items():
                if case_view != view:
                    continue
                if not cap.set(cv2.CAP_PROP_POS_FRAMES, frame):
                    raise RuntimeError("local original video seek failed")
                ok, image = cap.read()
                if not ok or round(cap.get(cv2.CAP_PROP_POS_FRAMES)) != frame + 1:
                    raise RuntimeError("local original source frame position differs")
                digest = hashlib.sha256(image.tobytes()).hexdigest()
                if digest != expected["expected_bgr_sha256"]:
                    raise ValueError(f"native pixels differ from previous staging: {view}/{frame}")
                results.append({"view": view, "frame": frame, "bgr_sha256": digest, **expected})
        finally:
            cap.release()
    result = {
        "status": "ok",
        "gpu_used": False,
        "models_reverified_via_frozen_prepare_sequence": assets,
        "local_asset_manifest": str(asset_manifest),
        "videos": details,
        "native_pixel_cases": results,
        "cache_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python_executable": sys.executable,
        "opencv": cv2.__version__,
        "scope": "18 exact decoded native source frames, metadata of all videos, all model hashes",
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "native_pixel_cases": len(results)}))


if __name__ == "__main__":
    run()
