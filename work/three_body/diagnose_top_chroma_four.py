"""Fixed four-frame, full-image color diagnostic; no masks, fit, model or thresholds.

Raw uint8 BGR and each period's already registered background are used unchanged.
C(R,G,B)=(R-G,(R+G)/2-B). D=norm(C(raw)-C(background)), maximum 255*sqrt(5).
Also retain D after the SAME original per-BGR-channel median subtraction, to
separate the change of color feature from global-illumination correction.
The old continuous score is max(abs(raw-background-channel_median)), maximum510.
No connected components, threshold selection, region selection or ID labels.
"""

from __future__ import annotations

import ast
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

WORK = Path(__file__).resolve().parent
OUT = WORK / "top_chroma_four_frame_diagnostic_v1"
OLD = WORK / "top_contrast_13frames_v1"
PREP = WORK / "semantic_node200_preparation_v2"
OLD_SHA = "2dc437ef827420e1b036fb9603e8f8b2681b9585fb2113489bdbfe9ca4737b4d"
PREP_SHA = "153c9f5f4a316f193b609afd2113b875cf1a62380dca4b80a6b27834b53afc15"
READER_SHA = "321c21e75c3818e38c2c140a2065d136dfa13f383d15fb61103edcbf650254fc"
inputs, outputs = {}, {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def bind(path, expected=None):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    digest = sha(path)
    after = path.stat()
    require(
        (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "input changed"
    )
    require(expected is None or digest == expected, "input SHA differs: " + str(path))
    row = dict(sha256=digest, size_bytes=after.st_size, mtime_ns=after.st_mtime_ns)
    require(str(path) not in inputs or inputs[str(path)] == row, "input changed")
    inputs[str(path)] = row
    return path


def sealed(root, seal, name):
    path = (root / name).resolve()
    rows = [r for r in seal["outputs"].values() if Path(r["path"]).resolve() == path]
    require(len(rows) == 1, "single declared output required: " + name)
    row = rows[0]
    bind(path, row["sha256"])
    require(path.stat().st_size == row.get("size_bytes", row.get("size")), "sealed size differs")
    return path


def read_json(path):
    return json.loads(Path(path).read_bytes())


def decode(path, expected=None):
    image = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    require(image is not None and image.shape == (1200, 1200, 3), "native1200 BGR required")
    require(expected is None or array_sha(image) == expected, "decoded BGR differs")
    return image


def save(name, value):
    path = OUT / name
    with path.open("xb") as stream:
        if name.endswith(".npz"):
            np.savez_compressed(stream, **value)
        elif name.endswith(".png"):
            ok, data = cv2.imencode(".png", value)
            require(
                ok and np.array_equal(cv2.imdecode(data, cv2.IMREAD_UNCHANGED), value),
                "PNG roundtrip",
            )
            stream.write(data.tobytes())
        elif isinstance(value, bytes):
            stream.write(value)
        else:
            stream.write((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())
    outputs[name] = dict(sha256=sha(path), size_bytes=path.stat().st_size)


def color(bgr):
    b, g, r = np.moveaxis(np.asarray(bgr, np.float64), -1, 0)
    return np.stack((r - g, (r + g) / 2 - b), axis=-1)


def maps(raw, background):
    delta = raw.astype(np.float32) - background.astype(np.float32)
    median = np.median(delta, axis=(0, 1))
    corrected = delta - median
    return dict(
        raw_bgr=raw,
        background_bgr=background,
        chroma_delta=color(delta),
        chroma_distance=np.linalg.norm(color(delta), axis=-1),
        corrected_chroma_distance=np.linalg.norm(color(corrected), axis=-1),
        old_channel_absolute=np.abs(corrected),
        old_channel_max=np.max(np.abs(corrected), axis=-1),
        old_channel_median_bgr=median,
    )


def panel(image, label):
    canvas = np.zeros((650, 600, 3), np.uint8)
    canvas[50:] = cv2.resize(image, (600, 600), interpolation=cv2.INTER_AREA)
    cv2.putText(canvas, label, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return canvas


def heat(value, maximum):
    require(np.isfinite(value).all() and value.min() >= 0 and value.max() <= maximum, "map range")
    return cv2.applyColorMap(np.rint(value / maximum * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def run():
    OUT.mkdir(exist_ok=False)
    started = datetime.now(UTC).isoformat()
    try:
        bind(__file__)
        save("source.py", Path(__file__).read_bytes())
        cv2.setNumThreads(2)
        old = read_json(bind(OLD / "contrast.provenance.json", OLD_SHA))
        prep = read_json(bind(PREP / "preparation.provenance.json", PREP_SHA))
        require(old["status"] == prep["status"] == "ok", "completed source seals required")
        summary = read_json(sealed(OLD, old, "summary.json"))
        sequence = Path(summary["source_sequence"])
        manifest = read_json(bind(sequence / "manifest.json", summary["source_manifest_sha256"]))
        old_background = decode(
            sealed(OLD, old, "registered_background_top.png"), summary["background_bgr_sha256"]
        )
        background_rows = read_json(sealed(PREP, prep, "backgrounds.json"))
        bg = next(r for r in background_rows if r["view"] == "top_cam")
        current_background = decode(
            sealed(PREP, prep, bg["path"]), bg["decoded_bgr_signature"]["sha256"]
        )
        # Reuse only the frozen native_read function, without importing its
        # unrelated tracking/model dependencies or executing any module code.
        source = bind(WORK / "measure_persistent_silhouette.py", READER_SHA)
        parsed = ast.parse(source.read_text(encoding="utf-8"))
        node = next(
            n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == "native_read"
        )
        namespace = dict(cv2=cv2, np=np, require=require, array_sha256=array_sha)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        video = manifest["raw"]["top_cam"]
        video_path = (sequence / video["path"]).resolve()
        require(old["inputs"][str(video_path)]["sha256"] == video["sha256"], "old raw lineage")
        print("Hashing one sealed old top AVI; only one native frame will be decoded", flush=True)
        bind(video_path, video["sha256"])
        capture = cv2.VideoCapture(str(video_path))
        try:
            raw = namespace["native_read"](
                capture, 79381 - 76620, video["frame_bgr_sha256"][79381 - 76620]
            )
        finally:
            capture.release()
        frames = [(79381, raw, old_background)]
        natives = read_json(sealed(PREP, prep, "native_evidence.json"))
        for native in (333519, 333559, 333718):
            row = next(r for r in natives if r["frame_idx"] == native and r["view"] == "top_cam")
            frames.append(
                (
                    native,
                    decode(sealed(PREP, prep, row["png_path"]), row["bgr_sha256"]),
                    current_background,
                )
            )
        # Small exact arithmetic check, including the cube-vertex maximum.
        sample = np.array([[[20, 40, 60], [90, 60, 20]]], np.uint8)
        require(np.array_equal(color(sample), color(sample + 30)), "neutral additive cancellation")
        vertices = np.array(
            [[b, g, r] for b in (-255, 255) for g in (-255, 255) for r in (-255, 255)]
        )
        maximum = float(255 * np.sqrt(5))
        require(np.linalg.norm(color(vertices), axis=-1).max() == maximum, "theoretical maximum")
        rows = []
        for native, raw, background in frames:
            arrays = maps(raw, background)
            # Median subtraction can double a channel bound; same factor for C.
            save(f"native_{native}.npz", arrays)
            save(f"raw_{native}.png", raw)
            colors = [
                heat(arrays[k], limit)
                for k, limit in (
                    ("chroma_distance", maximum),
                    ("corrected_chroma_distance", 2 * maximum),
                    ("old_channel_max", 510),
                )
            ]
            for name, image in zip(("chroma", "corrected_chroma", "old_abs"), colors, strict=True):
                save(f"{name}_{native}.png", image)
            save(
                f"compare_{native}.png",
                np.hstack(
                    [
                        panel(raw, f"Raw top {native}"),
                        panel(background, "Registered background"),
                        panel(colors[0], f"D: 0..{maximum:.3f}"),
                        panel(colors[1], f"D median-corrected: 0..{2 * maximum:.3f}"),
                        panel(colors[2], "Old channel abs max: 0..510"),
                    ]
                ),
            )
            rows.append(
                dict(
                    native=native,
                    raw_bgr_sha256=array_sha(raw),
                    background_bgr_sha256=array_sha(background),
                    ranges={
                        k: dict(
                            min=float(arrays[k].min()),
                            max=float(arrays[k].max()),
                            mean=float(arrays[k].mean()),
                        )
                        for k in ("chroma_distance", "corrected_chroma_distance", "old_channel_max")
                    },
                    old_channel_median_bgr=arrays["old_channel_median_bgr"].tolist(),
                )
            )
            print(json.dumps(rows[-1]), flush=True)
        save(
            "summary.json",
            dict(
                schema="top_chroma_four_frame_diagnostic_v1",
                rows=rows,
                theoretical_chroma_max=maximum,
                theoretical_corrected_chroma_bound=2 * maximum,
                theoretical_old_abs_bound=510,
                neutral_addition_check_passed=True,
                native_frame_decodes=4,
                background_decodes=2,
                masks_read=0,
                thresholds_selected=0,
                model_or_GPU=False,
                original_background_frames=summary["background_frames"],
                current_background_metadata=bg,
                current_background_contains_future_information=True,
                display="full field, linear 0..theoretical bound, TURBO; no per-image rescaling",
                quantitative_maps_native_shape=[1200, 1200],
            ),
        )
        for path, row in inputs.items():
            info = Path(path).stat()
            require(
                (info.st_size, info.st_mtime_ns) == (row["size_bytes"], row["mtime_ns"]),
                "final input stat drift",
            )
            if Path(path) != video_path:
                require(sha(path) == row["sha256"], "final small input SHA drift")
        for name, row in outputs.items():
            require(sha(OUT / name) == row["sha256"], "final output SHA drift")
        seal = dict(
            status="ok",
            inputs=inputs,
            outputs=outputs,
            started_at=started,
            old_large_AVI_SHA_verifications=1,
            large_AVI_final_guard="size+mtime, decoded selected BGR exact; no second7GB scan",
        )
    except BaseException as error:
        seal = dict(
            status="failed", inputs=inputs, outputs=outputs, started_at=started, error=repr(error)
        )
        raise
    finally:
        (OUT / "manifest.json").write_text(json.dumps(seal, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()
