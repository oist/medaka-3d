"""Registrations 4--5: clean crops, one side model, silhouette stopping; P-1-1 and K_1_2.

Subcommands, in order, each with ``--recording p11|k12`` and an optional
``--config`` (default: Registration 4's): ``bundle`` (fixed inputs; for a
recording whose crops come from video also the label-free fragments),
``extract`` (crops and pose masks from video, when the configuration says
so), ``prepare`` (dorsal-up flips and the clean flag, or reuse of another
run's prepared crops), ``train`` (top and pooled side encoders with
silhouette stopping, then embeddings) and ``evaluate``. The method is fixed
in ``labs/20261106_contrastive_appearance_link/README.md``, Registrations 4
and 5. Earlier runs and the production chain are read, never written.
Reference labels are read only by ``evaluate``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import tomllib
from concurrent.futures import ProcessPoolExecutor
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_appearance_gap as rg  # noqa: E402
import run_appearance_link as r1  # noqa: E402
import run_appearance_naming as rn  # noqa: E402

from medaka3d import load_paths  # noqa: E402
from medaka3d.eval.review_identity import contiguous_runs, interval_weights  # noqa: E402
from medaka3d.provenance import provenance  # noqa: E402

REPO = r1.REPO
CONFIG = HERE / "config_appearance_clean_v1.toml"
VIEWS = r1.VIEWS
RISK_LAB = REPO / "labs/20261104_geometry_flip_risk/linker.py"
LOWER_COLUMNS = [
    "segment_id",
    "start_frame",
    "stop_frame_exclusive",
    "start_s",
    "end_s",
    "duration_s",
]
WARNING_COLUMNS = [
    "interval_id",
    "start_frame",
    "stop_frame_exclusive",
    "start_s",
    "end_s",
    "duration_s",
    "missing_geometry_s",
    "reasons",
]
CROP_KEYS = (
    "crops",
    "frame",
    "slot",
    "interval",
    "overlap",
    "centre",
    "heading",
    "length",
    "color",
    "row",
)
CROP_SOURCES = ("registrations_1_3", "video", "prepared_run")
BUNDLE_FILES = (
    "timing.npz",
    "lower_risk_intervals.csv",
    "warning_intervals_identity_core.csv",
    "warning_mask.npz",
)
# Validated categorical palette (dataviz skill reference instance), fixed order.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
INK = "#52514e"


def settings(recording, config=None):
    path = Path(config).resolve() if config else CONFIG
    cfg = tomllib.loads(path.read_text(encoding="utf-8"))
    cfg["_path"] = str(path)
    if recording not in cfg["run_ids"]:
        raise ValueError(f"Unknown recording {recording!r}")
    paths = load_paths()
    root = paths.raw_data_dir.parent
    rc = dict(cfg[recording])
    if "output_root" in rc:
        if rc["output_root"] != "appearance_staging_dir":
            raise ValueError(f"Unknown output_root: {rc['output_root']!r}")
        if paths.appearance_staging_dir is None:
            raise ValueError("Set appearance_staging_dir in configs/paths.local.toml")
        relative = Path(rc["output"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Staged output must be relative to appearance_staging_dir")
        rc["output"] = str(paths.appearance_staging_dir / relative)
    if rc["crop_source"] not in CROP_SOURCES:
        raise ValueError(f"{recording}: crop_source must be one of {CROP_SOURCES}")
    return cfg, rc, paths, root, root / rc["output"] / cfg["run_ids"][recording]


def load_risk_lab():
    spec = importlib.util.spec_from_file_location("p11_motion_risk", RISK_LAB)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def snapshot(rec, out, cfg, folder):
    lab = REPO / cfg["lab"]
    for path in (
        Path(__file__),
        Path(cfg["_path"]),
        Path(r1.__file__),
        r1.CONFIG,
        Path(rn.__file__),
        rn.CONFIG,
        Path(rg.__file__),
        rg.CONFIG,
        lab / "README.md",
        lab / "appearance.py",
        RISK_LAB,
        REPO / "pyproject.toml",
        REPO / "uv.lock",
    ):
        path = Path(path).resolve()
        dest = out / folder / path.relative_to(REPO)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        rec.input(path.relative_to(REPO).as_posix(), path, sha256=r1.sha256(path))


def status(out, value):
    r1.write_json(out / "status.json", {"status": value})


def require_status(out, value):
    if json.loads((out / "status.json").read_text())["status"] != value:
        raise ValueError(f"{out.name}: expected status {value!r}")


def bundle_inputs(out):
    return json.loads((out / "bundle" / "inputs.json").read_text())


def timing(out):
    with np.load(out / "bundle" / "timing.npz") as z:
        return z["times"], z["weights"]


def clean_file(rec, out, inputs, sub, view):
    """Verified path of a prepared crop file: this run's, or a reused run's."""
    key = f"{sub}_{view}"
    if key in inputs:
        return r1.verified(rec, key, inputs[key]["path"], inputs[key]["sha256"])
    return r1.verified(
        rec,
        key,
        out / sub / f"{view}.npz",
        rn.recorded_output(out, "appearance_clean_prepare", key),
    )


def run_clean_path(run, sub, view):
    """Unverified path of a run's prepared crop file (for a baseline read by evaluate)."""
    inputs = bundle_inputs(run)
    key = f"{sub}_{view}"
    return Path(inputs[key]["path"]) if key in inputs else run / sub / f"{view}.npz"


# --------------------------------------------------------------------------- bundle


class Inputs:
    """Hash-recorded fixed inputs of one run."""

    def __init__(self, rec):
        self.rec = rec
        self.table = {}

    def put(self, name, path, expected=None):
        path = Path(path)
        digest = r1.sha256(path)
        if expected is not None and digest != expected:
            raise ValueError(f"Frozen input changed: {path}")
        self.rec.input(name, path, sha256=digest)
        self.table[name] = {"path": str(path), "sha256": digest}
        return path


def bundle_p11(cfg, rc, root, out, rec):
    """P-1-1 fixed inputs; crops from Registrations 1--3 or, for video mode, the videos."""
    configs = {
        k: tomllib.loads((HERE / rc[k]).read_text(encoding="utf-8"))
        for k in ("link_config", "naming_config", "gap_config")
    }
    r1cfg = configs["link_config"]
    runs = {k: root / "jingyi/P-1-1" / c["run_id"] for k, c in configs.items()}
    track, warn = root / r1cfg["input"]["tracking_run"], root / r1cfg["input"]["warning_run"]
    manifest = json.loads((track / "artifact_manifest.json").read_text())
    warn_manifest = json.loads((warn / "artifact_manifest.json").read_text())
    tinputs = json.loads((track / "inputs.json").read_text())
    fixed = Inputs(rec)
    fixed.put("chain", track / r1cfg["input"]["chain"], manifest[r1cfg["input"]["chain"]]["sha256"])
    fixed.put(
        "scoring", track / r1cfg["input"]["scoring"], manifest[r1cfg["input"]["scoring"]]["sha256"]
    )
    mask = fixed.put(
        "warning_mask",
        warn / r1cfg["input"]["warning_mask"],
        warn_manifest[r1cfg["input"]["warning_mask"]]["sha256"],
    )
    lower = fixed.put(
        "lower_risk",
        warn / r1cfg["input"]["lower_risk"],
        warn_manifest[r1cfg["input"]["lower_risk"]]["sha256"],
    )
    table = fixed.put(
        "warning_intervals",
        warn / configs["gap_config"]["input"]["warning_intervals"],
        warn_manifest[configs["gap_config"]["input"]["warning_intervals"]]["sha256"],
    )
    for view in VIEWS:
        fixed.put(f"cache_{view}", tinputs[view]["cache"], tinputs[view]["cache_sha256"])
    if rc["crop_source"] == "registrations_1_3":
        for run in runs.values():
            require_status(run, "evaluated")
        train_record = json.loads(
            (runs["link_config"] / "p11_appearance_train.provenance.json").read_text()
        )
        for view in VIEWS:
            fixed.put(
                f"crops_{view}",
                runs["link_config"] / "crops" / f"{view}.npz",
                train_record["inputs"][f"crops_{view}"]["sha256"],
            )
            fixed.put(
                f"masked_{view}",
                runs["naming_config"] / "masked" / f"{view}.npz",
                rn.recorded_output(
                    runs["naming_config"], "p11_appearance_naming_mask", f"masked_{view}"
                ),
            )
            fixed.put(
                f"warning_masked_{view}",
                runs["gap_config"] / "masked" / f"{view}.npz",
                rn.recorded_output(
                    runs["gap_config"], "p11_appearance_gap_extract", f"masked_{view}"
                ),
            )
            # Registration 3 hashed its masked crops, not the raw crops that hold the
            # heading; the raw file is fixed here and aligned row by row in prepare.
            fixed.put(f"warning_crops_{view}", runs["gap_config"] / "crops" / f"{view}.npz")
    else:
        videos = r1.video_hashes(track / r1cfg["input"]["video_provenance"])
        for view in VIEWS:
            video = Path(videos[view]["path"])
            if video.stat().st_size != videos[view]["size"]:
                raise ValueError(f"{view}: source video size changed")
            # Verified against the review pack's hash during extraction.
            fixed.table[f"video_{view}"] = {
                "path": str(video),
                "sha256": videos[view]["sha256"],
                "size": videos[view]["size"],
            }
    with np.load(fixed.table["scoring"]["path"]) as z:
        times, weights = z["recording_time_s"], z["duration_s"]
    np.savez(out / "bundle" / "timing.npz", times=times, weights=weights)
    # Label-free columns only; the reference columns are read by evaluate alone.
    pd.read_csv(lower)[LOWER_COLUMNS].to_csv(
        out / "bundle" / "lower_risk_intervals.csv", index=False
    )
    pd.read_csv(table)[WARNING_COLUMNS].to_csv(
        out / "bundle" / "warning_intervals_identity_core.csv", index=False
    )
    shutil.copy2(mask, out / "bundle" / "warning_mask.npz")
    return fixed.table, {
        "source_runs": {k: str(v) for k, v in runs.items()},
        "crop_source": rc["crop_source"],
    }


def bundle_k12_video(cfg, rc, paths, root, out, rec):
    """K_1_2 fixed inputs and label-free fragments from the production chain output."""
    exp, lam = rc["experiment"], cfg["lambda_mm"]
    chain_dir, cache_dir = root / rc["chain_dir"], root / rc["cache_dir"]
    raw = paths.raw_data_dir / exp
    meta = json.loads((raw / "experiment_metadata.json").read_text())
    fixed = Inputs(rec)
    label = fixed.put("chain", chain_dir / f"C34-label__{exp}__lam{lam:g}.npz")
    gated = fixed.put("chain_gated", chain_dir / f"C34-gated__{exp}__lam{lam:g}.npz")
    fixed.put("chain_provenance", chain_dir / "c34.provenance.json")
    for view in VIEWS:
        fixed.put(f"cache_{view}", cache_dir / f"{exp}__{view}.npz")
        video = raw / view / Path(meta["views"][view]["video_path"]).name
        if not video.exists():
            raise FileNotFoundError(video)
        # Hashed during extraction, which reads the whole file anyway.
        fixed.table[f"video_{view}"] = {
            "path": str(video),
            "sha256": None,
            "size": video.stat().st_size,
        }
    frametimes = fixed.put("frametimes", raw / "top_cam" / "frametimes.npy")
    times = np.load(frametimes)[1].astype(np.float64)
    wc = rc["warning"]
    weights, duration = interval_weights(times, wc["preparation_seconds"])
    with np.load(label) as z, np.load(gated) as g:
        xyz, source = z["xyz"], z["source"]
        residual, margin, local = z["residual_mm"], z["margin_mm"], g["slot_track"]
        if not np.array_equal(xyz, g["xyz"], equal_nan=True):
            raise ValueError("Label/gated geometry mismatch")
        for view in VIEWS:
            with np.load(fixed.table[f"cache_{view}"]["path"]) as c:
                if c["frame_idx"].size != z[f"track__{view}"].size:
                    raise ValueError(f"{view}: cache rows differ from the chain's track rows")
                if str(c["experiment"]) != exp or str(c["view"]) != view:
                    raise ValueError(f"{view}: cache names another experiment or view")
    n = xyz.shape[0]
    if n != times.size or source.shape != (n, 3, 3):
        raise ValueError("Chain frames do not match the acquisition clock")
    risk = load_risk_lab()
    flags, nearest, valid = risk.signals(xyz, source, residual, local, margin, wc)
    table, mask = risk.warning_intervals(
        {k: flags[k] for k in wc["rules"]},
        nearest,
        times,
        weights,
        wc["proximity_mm"],
        wc["padding_s"],
    )
    selected = ~mask & valid.all(axis=1) & (weights > 0)
    rows = []
    for i, (lo, hi) in enumerate(contiguous_runs(selected)):
        rows.append(
            {
                "segment_id": i,
                "start_frame": int(lo),
                "stop_frame_exclusive": int(hi),
                "start_s": float(times[lo]),
                "end_s": float(times[min(hi, n - 1)]),
                "duration_s": float(weights[lo:hi].sum()),
            }
        )
    lower = pd.DataFrame(rows, columns=LOWER_COLUMNS)
    lower.to_csv(out / "bundle" / "lower_risk_intervals.csv", index=False)
    table[WARNING_COLUMNS].to_csv(
        out / "bundle" / "warning_intervals_identity_core.csv", index=False
    )
    np.savez_compressed(out / "bundle" / "warning_mask.npz", warning=mask)
    np.savez(out / "bundle" / "timing.npz", times=times, weights=weights)
    summary = {
        "frames": int(n),
        "expected_seconds": duration,
        "warning_seconds": float(weights[mask].sum()),
        "warning_intervals": len(table),
        "lower_risk_seconds": float(lower.duration_s.sum()),
        "lower_risk_intervals": len(lower),
        "invalid_position_seconds": float(weights[~valid.all(axis=1)].sum()),
        "signal_seconds": {k: float(weights[v].sum()) for k, v in flags.items()},
        "chain_written": json.loads((chain_dir / "c34.provenance.json").read_text()).get(
            "finished_at"
        ),
        "note": "Chain predates the 2026-11-02 adapter repair; see Registration 4.",
        "crop_source": "video",
    }
    return fixed.table, summary


def bundle_prepared(cfg, rc, root, out, rec):
    """Reuse another run's bundle and prepared crops (Registration 5, variant A)."""
    src = root / rc["output"] / rc["prepared_run"]
    if json.loads((src / "status.json").read_text())["status"] not in (
        "prepared",
        "trained",
        "evaluated",
    ):
        raise ValueError(f"{src.name}: crops are not prepared")
    record = json.loads((src / "appearance_clean_bundle.provenance.json").read_text())
    for name in (*BUNDLE_FILES, "inputs.json"):
        path = r1.verified(
            rec, f"prepared_bundle_{name}", src / "bundle" / name, record["outputs"][name]["sha256"]
        )
        if name != "inputs.json":
            shutil.copy2(path, out / "bundle" / name)
    src_inputs = bundle_inputs(src)
    fixed = Inputs(rec)
    for key, value in src_inputs.items():
        if key.startswith(("chain", "scoring", "lower_risk", "warning_", "cache_", "frametimes")):
            fixed.put(key, value["path"], value["sha256"])
    for view in VIEWS:
        for sub in ("clean", "clean_warning"):
            fixed.put(
                f"{sub}_{view}",
                run_clean_path(src, sub, view),
                rn.recorded_output(src, "appearance_clean_prepare", f"{sub}_{view}"),
            )
    return fixed.table, {"prepared_run": str(src), "crop_source": "prepared_run"}


def cmd_bundle(cfg, rc, paths, root, out, recording):
    out.mkdir(exist_ok=False, parents=True)
    (out / "bundle").mkdir()
    status(out, "bundle running")
    with provenance("appearance_clean_bundle", out) as rec:
        rec.param("configuration", cfg)
        rec.param("recording", recording)
        snapshot(rec, out, cfg, "source")
        if rc["crop_source"] == "prepared_run":
            inputs, summary = bundle_prepared(cfg, rc, root, out, rec)
        elif recording == "p11":
            inputs, summary = bundle_p11(cfg, rc, root, out, rec)
        else:
            inputs, summary = bundle_k12_video(cfg, rc, paths, root, out, rec)
        r1.write_json(out / "bundle" / "inputs.json", inputs)
        for name in (*BUNDLE_FILES, "inputs.json"):
            rec.output(name, out / "bundle" / name, sha256=r1.sha256(out / "bundle" / name))
        rec.param("summary", summary)
        r1.write_json(out / "bundle" / "summary.json", summary)
    status(out, "bundled")


# --------------------------------------------------------------------------- extract


def split_crops(path_all, path_lower, path_warning):
    """Separate one decode pass into lower-risk and warning crop files."""
    with np.load(path_all) as z:
        arrays = {k: z[k] for k in CROP_KEYS}
    lower = arrays["interval"] >= 0
    np.savez(path_lower, **{k: v[lower] for k, v in arrays.items()})
    warning = {k: v[~lower] for k, v in arrays.items()}
    warning["interval"] = (-2 - warning["interval"]).astype(arrays["interval"].dtype)
    np.savez(path_warning, **warning)
    return int(lower.sum()), int((~lower).sum())


def cmd_extract(cfg, rc, paths, root, out, recording):
    if rc["crop_source"] != "video":
        raise SystemExit(f"{recording}: crop_source is {rc['crop_source']!r}; nothing to decode")
    require_status(out, "bundled")
    inputs = bundle_inputs(out)
    lower = pd.read_csv(out / "bundle" / "lower_risk_intervals.csv")
    table = pd.read_csv(out / "bundle" / "warning_intervals_identity_core.csv")
    with np.load(inputs["chain"]["path"]) as z:
        source = z["source"]
    n, step = source.shape[0], cfg["crops"]["frame_step"]
    out_px = cfg["crops"]["out_px"]
    interval, _ = r1.lower_risk_frames(lower, n, step)
    warn_id = np.full(n, -1, np.int32)
    for row in table.itertuples():
        warn_id[row.start_frame : row.stop_frame_exclusive] = row.interval_id
    with np.load(out / "bundle" / "warning_mask.npz") as z:
        if not np.array_equal(warn_id >= 0, z["warning"]):
            raise ValueError("Warning intervals do not reproduce the warning mask")
    if np.any((interval >= 0) & (warn_id >= 0)):
        raise ValueError("Lower-risk intervals overlap warnings")
    ids = np.where(interval >= 0, interval, np.where(warn_id >= 0, -2 - warn_id, -1))
    frames = np.flatnonzero((ids != -1) & (np.arange(n) % step == 0))
    for sub in ("crops_all", "crops", "masked", "warning_crops", "warning_masked"):
        (out / sub).mkdir(exist_ok=False)
    status(out, "extract running")
    with provenance("appearance_clean_extract", out) as rec:
        rec.param("frame_step", step)
        rec.param("out_px", out_px)
        caches, jobs = {}, {}
        with ProcessPoolExecutor(max_workers=3) as pool:
            for v, view in enumerate(VIEWS):
                caches[view] = r1.verified(
                    rec,
                    f"cache_{view}",
                    inputs[f"cache_{view}"]["path"],
                    inputs[f"cache_{view}"]["sha256"],
                )
                video = Path(inputs[f"video_{view}"]["path"])
                if video.stat().st_size != inputs[f"video_{view}"]["size"]:
                    raise ValueError(f"{view}: source video size changed since bundle")
                jobs[view] = pool.submit(
                    r1.extract_view,
                    view,
                    video,
                    caches[view],
                    source[:, :, v],
                    frames,
                    ids,
                    out / "crops_all" / f"{view}.npz",
                    out_px,
                )
            results = [jobs[view].result() for view in VIEWS]
        videos = {}
        for result in results:
            view = result["view"]
            expected = inputs[f"video_{view}"]["sha256"]
            if expected is not None and result["video_sha256"] != expected:
                raise ValueError(f"{view}: source video checksum changed")
            videos[view] = result["video_sha256"]
            rec.input(
                f"video_{view}", inputs[f"video_{view}"]["path"], sha256=result["video_sha256"]
            )
        counts = {}
        for view in VIEWS:
            counts[view] = split_crops(
                out / "crops_all" / f"{view}.npz",
                out / "crops" / f"{view}.npz",
                out / "warning_crops" / f"{view}.npz",
            )
        shutil.rmtree(out / "crops_all")
        masks = {}
        with ProcessPoolExecutor(max_workers=3) as pool:
            lower_masks = {
                view: pool.submit(
                    rn.mask_view,
                    view,
                    out / "crops" / f"{view}.npz",
                    caches[view],
                    out / "masked" / f"{view}.npz",
                )
                for view in VIEWS
            }
            masks["lower_risk"] = {view: lower_masks[view].result() for view in VIEWS}
            # The warning crops take the lower-risk fill, as Registration 3 intended.
            warning_masks = {
                view: pool.submit(
                    rn.mask_view,
                    view,
                    out / "warning_crops" / f"{view}.npz",
                    caches[view],
                    out / "warning_masked" / f"{view}.npz",
                    masks["lower_risk"][view]["fill_bgr"],
                )
                for view in VIEWS
            }
            masks["warning"] = {view: warning_masks[view].result() for view in VIEWS}
        for view in VIEWS:
            for sub in ("crops", "masked", "warning_crops", "warning_masked"):
                path = out / sub / f"{view}.npz"
                rec.output(f"{sub}_{view}", path, sha256=r1.sha256(path))
            r1.contact_sheet(out / "crops" / f"{view}.npz", out / "crops" / f"{view}_sample.png")
        summary = {
            "sampled_frames": int(frames.size),
            "out_px": out_px,
            "lower_risk_frames": int(((ids >= 0) & (np.arange(n) % step == 0)).sum()),
            "warning_frames": int(((ids <= -2) & (np.arange(n) % step == 0)).sum()),
            "crops": {v: {"lower_risk": c[0], "warning": c[1]} for v, c in counts.items()},
            "extraction": results,
            "masking": masks,
            "video_sha256": videos,
        }
        rec.param("results", summary)
        r1.write_json(out / "extract_summary.json", summary)
    status(out, "extracted")


# --------------------------------------------------------------------------- prepare


def crop_sources(out, recording, inputs):
    """(path, sha256) of the crops and masks feeding prepare, per view and set."""
    sources = {}
    for view in VIEWS:
        if f"crops_{view}" in inputs:
            sources[view] = {
                kind: (Path(inputs[f"{kind}_{view}"]["path"]), inputs[f"{kind}_{view}"]["sha256"])
                for kind in ("crops", "masked", "warning_crops", "warning_masked")
            }
        else:
            sources[view] = {
                kind: (
                    out / kind / f"{view}.npz",
                    rn.recorded_output(out, "appearance_clean_extract", f"{kind}_{view}"),
                )
                for kind in ("crops", "masked", "warning_crops", "warning_masked")
            }
    return sources


def prepare_view(view, sources, cache, dest_lower, dest_warning):
    lab = r1.load_lab()
    with np.load(cache) as c:
        rows_per_frame = np.bincount(c["frame_idx"].astype(np.int64))
    summary = {"view": view}
    for kind, crops_key, masked_key, dest in (
        ("lower_risk", "crops", "masked", dest_lower),
        ("warning", "warning_crops", "warning_masked", dest_warning),
    ):
        with np.load(sources[crops_key][0]) as z:
            raw = {k: z[k] for k in ("frame", "slot", "interval", "overlap", "row", "heading")}
        with np.load(sources[masked_key][0]) as z:
            masked, occluded, usable = z["crops"], z["occluded"], z["usable"]
            for key in ("frame", "slot", "row"):
                if not np.array_equal(z[key], raw[key]):
                    raise ValueError(f"{view} {kind}: masked crops are not aligned with crops")
        frame = raw["frame"].astype(np.int64)
        rows_in_frame = np.where(
            frame < rows_per_frame.size,
            rows_per_frame[np.minimum(frame, rows_per_frame.size - 1)],
            0,
        )
        clean = lab.clean_mask(occluded, raw["overlap"], rows_in_frame)
        flipped = lab.leftward(raw["heading"]) if view != "top_cam" else np.zeros(frame.size, bool)
        crops = masked if view == "top_cam" else lab.dorsal_up(masked, raw["heading"])
        np.savez(
            dest,
            crops=crops,
            frame=raw["frame"],
            slot=raw["slot"],
            interval=raw["interval"],
            heading=raw["heading"],
            occluded=occluded,
            overlap=raw["overlap"],
            rows_in_frame=rows_in_frame.astype(np.int16),
            clean=clean,
            usable=usable,
            row=raw["row"],
        )
        summary[kind] = {
            "crops": int(frame.size),
            "clean": int(clean.sum()),
            "clean_fraction": float(clean.mean()) if frame.size else np.nan,
            "usable_b": int(usable.sum()),
            "flipped": int(flipped.sum()),
            "unoccluded": int((occluded == 0).sum()),
            "no_overlap": int((~raw["overlap"]).sum()),
            "rows_at_least_3": int((rows_in_frame >= lab.CLEAN_MIN_ROWS).sum()),
            "crop_px": int(masked.shape[1]),
        }
    return summary


def clean_sheet(path_in: Path, path_out: Path, n=60, seed=0):
    import cv2

    with np.load(path_in) as z:
        clean = np.flatnonzero(z["clean"])
        pick = np.sort(
            np.random.default_rng(seed).choice(clean, size=min(n, clean.size), replace=False)
        )
        tiles = [cv2.resize(z["crops"][i], (96, 96), interpolation=cv2.INTER_NEAREST) for i in pick]
    rows = [np.hstack(tiles[i : i + 10]) for i in range(0, len(tiles) - len(tiles) % 10, 10)]
    if rows:
        cv2.imwrite(str(path_out), np.vstack(rows))


def cmd_prepare(cfg, rc, paths, root, out, recording):
    source = rc["crop_source"]
    if source == "prepared_run":
        require_status(out, "bundled")
        r1.write_json(out / "prepare_summary.json", {"reused": rc["prepared_run"]})
        status(out, "prepared")
        return
    require_status(out, "bundled" if source == "registrations_1_3" else "extracted")
    inputs = bundle_inputs(out)
    sources = crop_sources(out, recording, inputs)
    (out / "clean").mkdir(exist_ok=False)
    (out / "clean_warning").mkdir(exist_ok=False)
    status(out, "prepare running")
    with provenance("appearance_clean_prepare", out) as rec:
        caches = {}
        for view in VIEWS:
            caches[view] = r1.verified(
                rec,
                f"cache_{view}",
                inputs[f"cache_{view}"]["path"],
                inputs[f"cache_{view}"]["sha256"],
            )
            for kind, (path, digest) in sources[view].items():
                r1.verified(rec, f"{kind}_{view}", path, digest)
        with ProcessPoolExecutor(max_workers=3) as pool:
            jobs = [
                pool.submit(
                    prepare_view,
                    view,
                    sources[view],
                    caches[view],
                    out / "clean" / f"{view}.npz",
                    out / "clean_warning" / f"{view}.npz",
                )
                for view in VIEWS
            ]
            results = [j.result() for j in jobs]
        for view in VIEWS:
            for sub in ("clean", "clean_warning"):
                path = out / sub / f"{view}.npz"
                rec.output(f"{sub}_{view}", path, sha256=r1.sha256(path))
            clean_sheet(out / "clean" / f"{view}.npz", out / "clean" / f"{view}_sample.png")
        rec.param("results", results)
        r1.write_json(out / "prepare_summary.json", results)
    status(out, "prepared")


# --------------------------------------------------------------------------- train


def train_model(name, views, out, cfg, clean_paths):
    import torch

    tc = cfg["train"]
    torch.set_num_threads(tc["threads_per_model"])
    lab = r1.load_lab()
    parts = []
    for v, view in enumerate(views):
        with np.load(clean_paths["clean"][view]) as z:
            idx = np.flatnonzero(z["clean"])
            parts.append(
                {
                    "crops": z["crops"][idx],
                    "frame": z["frame"][idx],
                    "slot": z["slot"][idx],
                    "interval": z["interval"][idx],
                    "camera": np.full(idx.size, v, np.int8),
                    "index": idx,
                }
            )
    crops = np.concatenate([p.pop("crops") for p in parts])
    meta = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    fragment = meta["interval"].astype(np.int64) * 3 + meta["slot"]
    rng = np.random.default_rng(tc["seed"])
    validation = np.sort(
        rng.choice(len(crops), min(tc["validation_crops"], len(crops)), replace=False)
    )
    cross = None
    if len(views) > 1:
        df = pd.DataFrame({k: meta[k] for k in ("frame", "slot", "interval", "camera")})
        df["i"] = np.arange(len(crops))
        pairs = (
            df[df.camera == 0]
            .merge(df[df.camera == 1], on=["frame", "slot", "interval"], suffixes=("_a", "_b"))[
                ["i_a", "i_b"]
            ]
            .to_numpy()
        )
        if len(pairs) > tc["cross_pairs"]:
            pairs = pairs[np.sort(rng.choice(len(pairs), tc["cross_pairs"], replace=False))]
        cross = pairs
    monitor = lab.SilhouetteMonitor(
        validation,
        cross,
        patience=tc["patience"],
        target=tc["target"],
        target_patience=tc["target_patience"],
        seed=tc["seed"],
    )
    model, stats, log = lab.train_contrastive(
        crops,
        fragment,
        meta["interval"],
        np.ones(len(crops), bool),
        steps=tc["max_steps"],
        pairs=tc["pairs"],
        seed=tc["seed"],
        log_every=tc["eval_every"],
        monitor=monitor,
        encoder=tc["encoder"],
    )
    device = str(lab.model_device(model))
    del crops
    best_score, best_step, _ = monitor.best
    torch.save(
        {
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "mean": stats[0],
            "std": stats[1],
            "best_step": best_step,
            "best_silhouette": best_score,
            "stopped": monitor.stopped,
            "views": list(views),
            "encoder": tc["encoder"],
            "out_dim": model.out_dim,
            "device": device,
        },
        out / "models" / f"{name}.pt",
    )
    r1.write_json(out / "models" / f"{name}_log.json", log)
    ev = monitor.evaluations
    np.savez_compressed(
        out / "models" / f"{name}_history.npz",
        snapshots=np.stack(monitor.snapshots),
        clusters=np.stack(monitor.clusters),
        steps=np.array([e["step"] for e in ev]),
        silhouette=np.array([e["silhouette"] for e in ev]),
        inertia=np.array([e["inertia"] for e in ev]),
        cross_camera_median=np.array([e["cross_camera_median"] for e in ev]),
        validation=validation,
        validation_camera=meta["camera"][validation],
        validation_frame=meta["frame"][validation],
        validation_slot=meta["slot"][validation],
        validation_interval=meta["interval"][validation],
        validation_index=meta["index"][validation],
        views=np.array(views),
        cross_pairs=np.zeros((0, 2), np.int64) if cross is None else cross,
    )
    for view in views:
        for sub, dest in (("clean", "embeddings"), ("clean_warning", "embeddings_warning")):
            with np.load(clean_paths[sub][view]) as z:
                emb = lab.embed(model, z["crops"], stats)
            np.save(out / dest / f"{view}.npy", emb)
    return {
        "model": name,
        "views": list(views),
        "encoder": tc["encoder"],
        "device": device,
        "training_crops": len(fragment),
        "fragments": int(np.unique(fragment).size),
        "validation_crops": int(validation.size),
        "cross_pairs": 0 if cross is None else len(cross),
        "evaluations": len(ev),
        "best_step": best_step,
        "best_silhouette": best_score,
        "stopped": monitor.stopped,
        "last_step": ev[-1]["step"],
        "final_log": log[-1] if log else None,
    }


def cmd_train(cfg, rc, paths, root, out, recording):
    require_status(out, "prepared")
    for sub in ("models", "embeddings", "embeddings_warning"):
        (out / sub).mkdir(exist_ok=False)
    status(out, "train running")
    inputs = bundle_inputs(out)
    with provenance("appearance_clean_train", out) as rec:
        rec.param("train", cfg["train"])
        clean_paths = {
            sub: {view: clean_file(rec, out, inputs, sub, view) for view in VIEWS}
            for sub in ("clean", "clean_warning")
        }
        models = cfg["train"]["models"]
        with ProcessPoolExecutor(max_workers=len(models)) as pool:
            jobs = {
                name: pool.submit(train_model, name, tuple(views), out, cfg, clean_paths)
                for name, views in models.items()
            }
            results = [jobs[name].result() for name in models]
        for name in models:
            for suffix in (".pt", "_log.json", "_history.npz"):
                path = out / "models" / f"{name}{suffix}"
                rec.output(f"model_{name}{suffix}", path, sha256=r1.sha256(path))
        for view in VIEWS:
            for sub in ("embeddings", "embeddings_warning"):
                path = out / sub / f"{view}.npy"
                rec.output(f"{sub}_{view}", path, sha256=r1.sha256(path))
        rec.param("results", results)
        rec.param("device", sorted({r["device"] for r in results}))
        r1.write_json(out / "train_summary.json", results)
    status(out, "trained")


# --------------------------------------------------------------------------- evaluate


def female_anchors(gt, caches, source, match_px):
    """K_1_2 female anchors: (frame, slot) where a confirmed identification maps to one slot.

    The annotated mouth in a side view is matched to the nearest detection row
    of that camera-frame within ``match_px``; the slot whose source row that is
    must be unique; views that are both linked must agree.
    """
    lab = r1.load_lab()
    rows = gt[(gt.gt_status == "confirmed") & (gt.sex_label == "F")]
    n = source.shape[0]
    per_view, counts = {}, {"confirmed": len(rows)}
    for v, view in enumerate(VIEWS):
        if view == "top_cam":
            continue
        prefix = view.split("_")[0]
        with np.load(caches[view]) as c:
            frame_idx, points = c["frame_idx"].astype(np.int64), c["points"]
        start = np.searchsorted(frame_idx, np.arange(n + 1))
        found, dropped = {}, {"no_row": 0, "far": 0, "slot_not_unique": 0}
        linked = rows[rows[f"{prefix}_selection_status"] == "sleap_linked"]
        for row in linked.itertuples():
            f = int(row.frame)
            target = np.array(
                [getattr(row, f"{prefix}_mouth_x"), getattr(row, f"{prefix}_mouth_y")], float
            )
            lo, hi = int(start[f]), int(start[f + 1])
            if lo == hi or not np.isfinite(target).all():
                dropped["no_row"] += 1
                continue
            d = np.linalg.norm(points[lo:hi, lab.MOUTH] - target, axis=1)
            d = np.where(np.isfinite(d), d, np.inf)
            k = int(np.argmin(d))
            if d[k] > match_px:
                dropped["far"] += 1
                continue
            slots = np.flatnonzero(source[f, :, v] == k)
            if slots.size != 1:
                dropped["slot_not_unique"] += 1
                continue
            found[f] = (int(slots[0]), float(d[k]))
        per_view[view] = found
        counts[view] = {"linked": len(linked), "matched": len(found), **dropped}
    anchors, disagreed = [], 0
    for f in sorted(set().union(*(set(d) for d in per_view.values()))):
        seen = {view: per_view[view][f] for view in per_view if f in per_view[view]}
        slots = {s for s, _ in seen.values()}
        if len(slots) != 1:
            disagreed += 1
            continue
        anchors.append(
            {
                "frame": f,
                "slot": slots.pop(),
                "views": "+".join(sorted(seen)),
                "match_px": max(d for _, d in seen.values()),
            }
        )
    counts["views_disagree"] = disagreed
    counts["anchors"] = len(anchors)
    return pd.DataFrame(anchors, columns=["frame", "slot", "views", "match_px"]), counts


def score_anchors(name, anchors, weights, female=None, region=None):
    """Correct / wrong / unknown female anchors after the preparation period.

    ``female`` is the identity index the female carries; when None it is the
    name at the first named scored anchor (the chain's slot number, for A0).
    """
    keep = anchors[weights[anchors.frame.to_numpy()] > 0]
    if region is not None:
        keep = keep[region[keep.frame.to_numpy()]]
    held = name[keep.frame.to_numpy(), keep.slot.to_numpy()].astype(int)
    named = held[held >= 0]
    if female is None:
        female = int(named[0]) if named.size else -1
    correct = int((held == female).sum()) if female >= 0 else 0
    wrong = int(((held >= 0) & (held != female)).sum())
    unknown = int((held < 0).sum())
    total = int(held.size)
    return {
        "anchors": total,
        "correct": correct,
        "wrong": wrong,
        "unknown": unknown,
        "named": correct + wrong,
        "wrong_fraction_of_named": wrong / (correct + wrong) if correct + wrong else np.nan,
        "named_fraction": (correct + wrong) / total if total else np.nan,
        "female_identity": female,
    }


def male_diagnostics(names, lower, female):
    """Label-free: consecutive named intervals whose names change, by whether the female moves.

    Slot continuation is defined across unnamed intervals too, so the
    comparison runs between consecutive fully named intervals (the anchors).
    """
    named = [int(i) for i in lower.segment_id if (names[int(i)] >= 0).all()]
    out = {
        "consecutive_named_pairs": max(len(named) - 1, 0),
        "changed": 0,
        "female_involved": 0,
        "male_only": 0,
    }
    for a, b in pairwise(named):
        na, nb = names[a], names[b]
        diff = na != nb
        if diff.any():
            out["changed"] += 1
            if female in set(na[diff].tolist()):
                out["female_involved"] += 1
            else:
                out["male_only"] += 1
    return out


def identity_tracks(path, frame, xyz, times, labels):
    n = frame.shape[0]
    by_identity = np.full((n, 3, xyz.shape[2]), np.nan, np.float32)
    for k in range(3):
        for s in range(3):
            sel = frame[:, s] == k
            by_identity[sel, k] = xyz[sel, s]
    np.savez_compressed(
        path,
        name=frame,
        xyz_by_identity=by_identity,
        identity_labels=np.array(labels),
        recording_time_s=times,
    )


def reference_projection(rec, track, names, n):
    """P-1-1 review positions per frame and the calibrated rig (evaluation only)."""
    from medaka3d.calib import load_rig

    tinputs = json.loads((track / "inputs.json").read_text())
    manifest = tinputs["calibration_calibration_manifest.json"]
    man = r1.verified(rec, "calibration_manifest", Path(manifest["path"]), manifest["sha256"])
    rig = load_rig(man.parent)
    review = r1.verified(
        rec, "review", Path(tinputs["review"]["path"]), tinputs["review"]["sha256"]
    )
    ref = pd.read_csv(
        review,
        usecols=[
            "frame",
            "label",
            "pectoral_X",
            "pectoral_Y",
            "pectoral_Z",
            "is_interpolated",
            "bad3d",
            "uncertain",
        ],
    )
    ref_xyz = np.full((n, 3, 3), np.nan)
    for r, label in enumerate(names):
        d = ref[(ref.label == label) & ~ref.is_interpolated & ~ref.bad3d & ~ref.uncertain]
        ref_xyz[d.frame.to_numpy(), r] = d[["pectoral_X", "pectoral_Y", "pectoral_Z"]].to_numpy()
    return rig, ref_xyz


def run_centroids(lab, run, names, times, lower, window_s):
    """Identity centroids of a run from its lower-risk clean crops and interval names."""
    meta, feats = {}, {}
    for view in VIEWS:
        with np.load(run_clean_path(run, "clean", view)) as z:
            meta[view] = {k: z[k] for k in ("frame", "slot", "interval")}
            meta[view]["overlap"] = ~z["clean"]
        feats[view] = np.load(run / "embeddings" / f"{view}.npy")
    whole = r1.fragment_rows(meta, times, lower, window_s)[2]
    descs = [[r1.describe(lab, feats, whole, i, s) for s in range(3)] for i in lower.segment_id]
    labels = np.stack([names[int(i)] for i in lower.segment_id])
    return lab.label_centroids(descs, labels)


def warning_crop_table(
    lab, rig, ref_xyz, slot_ref, caches, run, centroids, near_px=25.0, sep_px=25.0
):
    """Per clean, reference-clear warning crop of a run: nearest-centroid correctness."""
    rows = []
    for view in VIEWS:
        cam = rig.camera(view)
        with np.load(run_clean_path(run, "clean_warning", view)) as z:
            frame, slot, row, clean = (z[k] for k in ("frame", "slot", "row", "clean"))
        emb = np.load(run / "embeddings_warning" / f"{view}.npy")
        with np.load(caches[view]) as c:
            pect = c["points"][row, lab.PECTORAL]
        d = np.hypot(*(cam.project(ref_xyz[frame]) - pect[:, None, :]).transpose(2, 0, 1))
        dd = np.where(np.isfinite(d), d, np.inf)
        o = np.sort(dd, axis=1)
        clear = np.isfinite(pect).all(axis=1) & (o[:, 0] < near_px) & (o[:, 1] - o[:, 0] >= sep_px)
        sr = slot_ref[frame, slot]
        c = np.stack([centroids[view][k] for k in range(3)])
        pred = ((emb[:, None, :] - c[None]) ** 2).sum(axis=2).argmin(axis=1)
        for i in np.flatnonzero(clean & clear & (sr >= 0)):
            rows.append(
                {
                    "view": view,
                    "frame": int(frame[i]),
                    "slot": int(slot[i]),
                    "correct": bool(pred[i] == sr[i]),
                }
            )
    return pd.DataFrame(rows, columns=["view", "frame", "slot", "correct"])


def accuracy_by_view(table):
    out = {}
    for view in VIEWS:
        t = table[table.view == view]
        out[view] = {"crops": len(t), "accuracy": float(t.correct.mean()) if len(t) else np.nan}
    out["pooled"] = {
        "crops": len(table),
        "accuracy": float(table.correct.mean()) if len(table) else np.nan,
    }
    return out


def accuracy_rule(variant, baseline, ev):
    """Registration 5: pooled gain >= 2 points, no camera more than 1 point below."""
    diffs = {view: 100 * (variant[view]["accuracy"] - baseline[view]["accuracy"]) for view in VIEWS}
    pooled = 100 * (variant["pooled"]["accuracy"] - baseline["pooled"]["accuracy"])
    if any(d < -ev["max_camera_loss_points_reject"] for d in diffs.values()):
        verdict = "reject"
    elif pooled >= ev["min_pooled_gain_points"] and all(
        d >= -ev["max_camera_loss_points"] for d in diffs.values()
    ):
        verdict = "accuracy condition met"
    else:
        verdict = "inconclusive"
    return {"pooled_gain_points": pooled, "camera_gain_points": diffs, "verdict": verdict}


def separation_figure(lab, history, log, name, picks, dest, reference=None, ref_names=None):
    """V: PCA panels of the validation embeddings over training, plus the curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps, sil = history["steps"], history["silhouette"]
    snaps, clusters = history["snapshots"], history["clusters"]
    camera = history["validation_camera"]
    best, last = int(np.argmax(sil)), len(steps) - 1
    idx = sorted({min(int(p), last) for p in picks} | {best, last})
    by_camera = camera.max() > 0
    rows = 1 + int(reference is not None) + int(by_camera)
    fig, axes = plt.subplots(rows, len(idx), figsize=(2.9 * len(idx), 2.9 * rows), squeeze=False)
    for j, e in enumerate(idx):
        xy, explained = lab.pca2(snaps[e])
        tag = f"step {steps[e]}  s = {sil[e]:.2f}" + ("  (best)" if e == best else "")
        layers = [("k-means cluster", clusters[e], ["cluster 0", "cluster 1", "cluster 2"])]
        if reference is not None:
            layers.append(("reference identity", reference, [*ref_names, "unassessed"]))
        if by_camera:
            layers.append(("camera", camera, [str(v) for v in history["views"]]))
        for r, (what, labels, legend) in enumerate(layers):
            ax = axes[r, j]
            labels = np.asarray(labels)
            order = np.argsort(np.random.default_rng(0).random(len(labels)))
            colours = [SERIES[int(v)] if 0 <= int(v) < 3 else "#b9b8b2" for v in labels[order]]
            ax.scatter(xy[order, 0], xy[order, 1], s=4, c=colours, linewidths=0, alpha=0.7)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#d8d7d2")
            if r == 0:
                ax.set_title(tag, fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(what, fontsize=9, color=INK)
                handles = [
                    plt.Line2D(
                        [],
                        [],
                        marker="o",
                        linestyle="",
                        markersize=5,
                        color=SERIES[k] if k < 3 else "#b9b8b2",
                        label=text,
                    )
                    for k, text in enumerate(legend)
                ]
                ax.legend(handles=handles, fontsize=7, frameon=False, loc="upper left")
            ax.set_xlabel(f"PC1+PC2 explain {100 * explained.sum():.0f}%", fontsize=7, color=INK)
    fig.suptitle(f"{name} encoder: validation embeddings during training", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(dest / f"{name}_separation.png", dpi=150)
    plt.close(fig)

    panels = 3 if by_camera else 2
    fig, axes = plt.subplots(1, panels, figsize=(3.6 * panels, 3.0))
    ax = axes[0]
    ax.plot(steps, sil, color=SERIES[0], linewidth=2)
    ax.scatter([steps[best]], [sil[best]], s=36, color=SERIES[1], zorder=3)
    ax.annotate(
        f"best {sil[best]:.3f} at {steps[best]}",
        (steps[best], sil[best]),
        textcoords="offset points",
        xytext=(6, -12),
        fontsize=7,
        color=INK,
    )
    ax.axhline(lab.STOP_TARGET, color="#b9b8b2", linewidth=1, linestyle=":")
    ax.set_title("silhouette (k-means, 3 clusters)", fontsize=9, color=INK)
    ax.set_xlabel("step", fontsize=8, color=INK)
    ax = axes[1]
    log_steps = [e["step"] for e in log]
    ax.plot(
        log_steps,
        [e["d_pos_median"] for e in log],
        color=SERIES[0],
        linewidth=2,
        label="same fragment",
    )
    ax.plot(
        log_steps,
        [e["d_neg_median"] for e in log],
        color=SERIES[1],
        linewidth=2,
        label="coexisting fragment",
    )
    ax.set_title("median pair distance in the batch", fontsize=9, color=INK)
    ax.set_xlabel("step", fontsize=8, color=INK)
    ax.legend(fontsize=7, frameon=False)
    if by_camera:
        ax = axes[2]
        ax.plot(steps, history["cross_camera_median"], color=SERIES[2], linewidth=2)
        ax.set_title("same fish, same frame, two cameras", fontsize=9, color=INK)
        ax.set_xlabel("step", fontsize=8, color=INK)
    for ax in axes:
        ax.tick_params(labelsize=7, colors=INK)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(dest / f"{name}_curves.png", dpi=150)
    plt.close(fig)


def rule_named(res, ev, wrong_key="wrong_fraction_of_named", named_key="named_fraction"):
    if res[wrong_key] > ev["max_wrong_fraction_of_named"]:
        return "reject"
    if res[named_key] >= ev["min_named_fraction"]:
        return "met"
    return "inconclusive"


def cmd_evaluate(cfg, rc, paths, root, out, recording):
    require_status(out, "trained")
    lab = r1.load_lab()
    ev, gp = cfg["evaluate"], cfg["gap"]
    inputs = bundle_inputs(out)
    dest = out / "evaluation"
    # A network share may keep an emptied folder undeletable for a while, and a
    # blocked delete under MSYS leaves a hidden ``.msys*`` stub behind.
    if dest.exists() and any(not p.name.startswith(".msys") for p in dest.iterdir()):
        raise FileExistsError(f"{dest} already holds files")
    dest.mkdir(exist_ok=True)
    (dest / "figures").mkdir()
    status(out, "evaluate running")
    with provenance("appearance_clean_evaluate", out) as rec:
        snapshot(rec, out, cfg, "source_evaluate")
        times, weights = timing(out)
        n = times.size
        post = weights > 0
        lower = pd.read_csv(out / "bundle" / "lower_risk_intervals.csv")
        table = pd.read_csv(out / "bundle" / "warning_intervals_identity_core.csv")
        with np.load(out / "bundle" / "warning_mask.npz") as z:
            warning = z["warning"]
        chain = r1.verified(rec, "chain", inputs["chain"]["path"], inputs["chain"]["sha256"])
        with np.load(chain) as z:
            xyz, source = z["xyz"], z["source"]
        meta, meta_w, feats, feats_w, counts = {}, {}, {}, {}, {}
        for view in VIEWS:
            with np.load(clean_file(rec, out, inputs, "clean", view)) as z:
                meta[view] = {k: z[k] for k in ("frame", "slot", "interval", "clean")}
                meta[view]["overlap"] = ~meta[view]["clean"]  # fragment_rows drops these
            with np.load(clean_file(rec, out, inputs, "clean_warning", view)) as z:
                meta_w[view] = {k: z[k] for k in ("frame", "slot", "interval")}
                meta_w[view]["usable"] = z["clean"]
            feats[view] = np.load(
                r1.verified(
                    rec,
                    f"embeddings_{view}",
                    out / "embeddings" / f"{view}.npy",
                    rn.recorded_output(out, "appearance_clean_train", f"embeddings_{view}"),
                )
            )
            feats_w[view] = np.load(
                r1.verified(
                    rec,
                    f"embeddings_warning_{view}",
                    out / "embeddings_warning" / f"{view}.npy",
                    rn.recorded_output(out, "appearance_clean_train", f"embeddings_warning_{view}"),
                )
            )
            counts[view] = {
                "lower_risk_clean": int(meta[view]["clean"].sum()),
                "lower_risk_crops": int(meta[view]["clean"].size),
                "warning_clean": int(meta_w[view]["usable"].sum()),
                "warning_crops": int(meta_w[view]["usable"].size),
            }

        # Reference, read here only.
        if recording == "p11":
            scoring = r1.verified(
                rec, "scoring", inputs["scoring"]["path"], inputs["scoring"]["sha256"]
            )
            with np.load(scoring) as z:
                matched, anchor = z["matched_slot"], z["anchor_reference_to_slot"]
                labels = tuple(str(x) for x in z["reference_labels"])
                if not np.array_equal(z["duration_s"], weights):
                    raise ValueError("Bundle timing differs from the scoring arrays")
            if labels != rn.NAMES:
                raise ValueError(f"Unexpected reference labels {labels}")
            initial = [int(np.flatnonzero(anchor == s)[0]) for s in range(3)]
            lower_full = pd.read_csv(
                r1.verified(
                    rec,
                    "lower_risk_reference",
                    inputs["lower_risk"]["path"],
                    inputs["lower_risk"]["sha256"],
                )
            )
            ref = r1.reference_slots(lower_full, list(labels), ev["min_reference_coverage"])
            identity_labels = rn.NAMES
        else:
            initial = [0, 1, 2]
            ref = {int(i): None for i in lower.segment_id}
            identity_labels = ("id0", "id1", "id2")
        rec.param("initial_names", initial)

        # Naming on clean descriptors; S; boundaries.
        rows = r1.fragment_rows(meta, times, lower, ev["end_window_s"])
        names, how, rounds = rn.naming(lab, feats, rows[2], lower, initial)
        bounds = [(r.segment_id, r.start_frame, r.stop_frame_exclusive) for r in lower.itertuples()]
        chain_names = {int(i): np.array(initial, np.int8) for i in lower.segment_id}
        frames = {
            "A0_chain": lab.frame_names(n, bounds, chain_names),
            "N_carry": lab.frame_names(n, bounds, names, carry_warnings=True),
            "N_strict": lab.frame_names(n, bounds, names, carry_warnings=False),
            "S": lab.frame_names_bridged(n, bounds, names),
        }
        pd.DataFrame(
            [
                {"segment_id": i, "how": how[i], **{f"slot{s}": int(names[i][s]) for s in range(3)}}
                for i in lower.segment_id
            ]
        ).to_csv(dest / "interval_names.csv", index=False)
        tables = [
            r1.boundary_table(lab, feats, rows[0], rows[1], lower, ref, "A0_chain", "end"),
            r1.boundary_table(lab, feats, rows[0], rows[1], lower, ref, "clean", "end"),
            r1.boundary_table(lab, feats, rows[2], rows[2], lower, ref, "clean", "whole"),
        ]
        boundaries = pd.concat(tables, ignore_index=True)
        boundaries.to_csv(dest / "boundaries.csv", index=False)
        outcomes = {
            f"{m}|{w}": r1.outcome_counts(t)
            for (m, w), t in boundaries.groupby(["method", "window"])
        }
        decided = {
            f"{m}|{w}": {
                "decided": int(t.permutation.notna().sum()),
                "undecidable": int(t.permutation.isna().sum()),
                "margin_median": float(t.margin.median()),
            }
            for (m, w), t in boundaries.groupby(["method", "window"])
            if m != "A0_chain"
        }

        # G on clean evidence.
        anchors, gaps = lab.anchors_and_gaps(bounds, names)
        in_gap = np.zeros(n, bool)
        for _, _, first, stop in gaps:
            in_gap[first:stop] = True
        label_rows = np.stack([names[i] for i in lower.segment_id])
        descs = [
            [r1.describe(lab, feats, rows[2], i, s) for s in range(3)] for i in lower.segment_id
        ]
        centroids = lab.label_centroids(descs, label_rows)
        tau, calibration = {}, {}
        for view in VIEWS:
            m = meta[view]
            ident = np.full(m["frame"].size, -1)
            for seg in lower.segment_id:
                sel = m["interval"] == seg
                ident[sel] = names[seg][m["slot"][sel]]
            pick = m["clean"] & (ident >= 0)
            c = np.stack([centroids[view][k] for k in range(3)])
            dist2 = ((feats[view][pick][:, None, :] - c[None]) ** 2).sum(axis=2)
            tau[view] = lab.fit_temperature(dist2, ident[pick])
            calibration[view] = {
                "crops": int(pick.sum()),
                "tau": tau[view],
                "nearest_centroid_agreement": float((dist2.argmin(axis=1) == ident[pick]).mean()),
            }
        disagree = sum(bool((names[a] != names[b]).any()) for a, b, _, _ in gaps)
        gap_seconds = float(weights[in_gap].sum())
        rate = disagree / gap_seconds if gap_seconds else 0.0
        switch = {"disagreeing_gaps": disagree, "gap_seconds": gap_seconds, "rate_per_s": rate}
        if disagree > 0:
            warn_evidence = rg.gap_evidence(lab, feats_w, meta_w, centroids, tau, set())
            unnamed = {i for i in lower.segment_id if not (names[i] >= 0).all()}
            lr_meta = {
                v: {
                    "frame": meta[v]["frame"],
                    "slot": meta[v]["slot"],
                    "usable": meta[v]["clean"],
                    "interval": meta[v]["interval"],
                }
                for v in VIEWS
            }
            lr_evidence = rg.gap_evidence(
                lab, feats, lr_meta, centroids, tau, set(lower.segment_id) - unnamed
            )
            if set(warn_evidence) & set(lr_evidence):
                raise AssertionError("Warning and lower-risk evidence overlap")
            frames["G"], _ = rg.run_g(
                lab,
                n,
                names,
                anchors,
                gaps,
                {**warn_evidence, **lr_evidence},
                times,
                rate,
                gp["alpha"],
                gp["posterior"],
            )
            evidence_nodes = {
                "warning": len({f for f, _ in warn_evidence}),
                "unnamed_lower_risk": len({f for f, _ in lr_evidence}),
            }
        else:
            evidence_nodes = {
                "note": "no gap joins two anchors with different names; "
                "the switch rate is zero and G is undefined"
            }

        summary = {
            "recording": recording,
            "configuration": Path(cfg["_path"]).name,
            "counts": counts,
            "intervals": {
                "rounds": rounds,
                **{
                    k: sum(h == k for h in how.values())
                    for k in ("global", "partial_two", "partial_one", "unknown")
                },
            },
            "boundary_outcomes": outcomes,
            "boundaries_decided": decided,
            "anchors": len(anchors),
            "gaps": len(gaps),
            "calibration": calibration,
            "switch_rate": switch,
            "evidence_nodes": evidence_nodes,
        }

        if recording == "p11":
            regions = {
                "all": post,
                "lower_risk": post & ~warning,
                "warning": post & warning,
                "gap": post & in_gap,
            }
            for wid in rc["encounters"]:
                row = table[table.interval_id == wid].iloc[0]
                keep = np.zeros(n, bool)
                keep[row.start_frame : row.stop_frame_exclusive] = True
                regions[f"encounter_{wid}_{row.start_s:.3f}s"] = post & keep
            results = {
                label: rn.score(frame, matched, weights, regions) for label, frame in frames.items()
            }
            if not np.isclose(
                results["A0_chain"]["lower_risk"]["wrong"],
                lower_full.wrong_seconds.sum(),
                atol=1e-6,
            ):
                raise AssertionError("Chain naming does not reproduce the frozen score")
            earlier = {}
            for key, stage in (("naming_config", "N_B|carry"), ("gap_config", None)):
                run = (
                    root
                    / "jingyi/P-1-1"
                    / tomllib.loads((HERE / rc[key]).read_text(encoding="utf-8"))["run_id"]
                )
                prior = json.loads((run / "evaluation" / "summary.json").read_text())
                if stage:
                    earlier["registration2_N_B_carry"] = prior["naming"][stage]
                else:
                    earlier["registration3"] = {k: prior["results"][k] for k in ("S", "G")}
            stops = lower.set_index("segment_id").stop_frame_exclusive
            chain_wrong = boundaries[
                (boundaries.method == "A0_chain") & (boundaries.outcome == "wrong")
            ]
            clean_end = boundaries[(boundaries.method == "clean") & (boundaries.window == "end")]
            chain_rows = []
            for row in chain_wrong.itertuples():
                here = clean_end[clean_end.pre_interval == row.pre_interval].iloc[0]
                chain_rows.append(
                    {
                        "boundary_s": float(row.boundary_s),
                        "stop_frame": int(stops[row.pre_interval]),
                        "clean_outcome": here.outcome,
                        "margin": here.margin,
                    }
                )
            summary.update(
                {
                    "results": results,
                    "earlier_registrations": earlier,
                    "chain_wrong_boundaries": chain_rows,
                    "decision_rules": {
                        "N_carry_named_rule": rule_named(
                            results["N_carry"]["all"], ev, named_key="named_fraction_of_assessed"
                        ),
                        "G_vs_S_gap": (
                            "not run"
                            if "G" not in frames
                            else "reject"
                            if results["G"]["gap"]["wrong_fraction_of_named"]
                            > ev["max_wrong_fraction_of_named"]
                            or results["G"]["gap"]["named"] < results["S"]["gap"]["named"]
                            else "adopt"
                            if results["G"]["gap"]["named"] > results["S"]["gap"]["named"]
                            else "inconclusive"
                        ),
                    },
                }
            )
            slot_ref = np.full((n, 3), -1, np.int8)
            for r_ in range(3):
                ok = matched[:, r_] >= 0
                slot_ref[np.flatnonzero(ok), matched[ok, r_]] = r_
            reference_of = lambda frame, slot: slot_ref[frame, slot]  # noqa: E731
            ref_names = rn.NAMES

            # Registration 5: single-crop accuracy on reference-clear warning crops,
            # for this run and the baseline run, on their common crop set.
            if ev.get("baseline_run"):
                r1cfg = tomllib.loads((HERE / rc["link_config"]).read_text(encoding="utf-8"))
                track = root / r1cfg["input"]["tracking_run"]
                rig, ref_xyz = reference_projection(rec, track, rn.NAMES, n)
                caches = {view: inputs[f"cache_{view}"]["path"] for view in VIEWS}
                baseline = root / rc["output"] / ev["baseline_run"]
                require_status(baseline, "evaluated")
                base_names_df = pd.read_csv(baseline / "evaluation" / "interval_names.csv")
                base_names = {
                    int(r.segment_id): np.array([r.slot0, r.slot1, r.slot2], np.int8)
                    for r in base_names_df.itertuples()
                }
                base_lower = pd.read_csv(baseline / "bundle" / "lower_risk_intervals.csv")
                if not base_lower.equals(lower):
                    raise ValueError("Baseline run uses different lower-risk intervals")
                base_centroids = run_centroids(
                    lab, baseline, base_names, times, lower, ev["end_window_s"]
                )
                own = warning_crop_table(lab, rig, ref_xyz, slot_ref, caches, out, centroids)
                base = warning_crop_table(
                    lab, rig, ref_xyz, slot_ref, caches, baseline, base_centroids
                )
                own.to_csv(dest / "warning_crop_accuracy.csv", index=False)
                base.to_csv(dest / "warning_crop_accuracy_baseline.csv", index=False)
                common = own.merge(base, on=["view", "frame", "slot"], suffixes=("", "_base"))
                summary["warning_crop_accuracy"] = {
                    "baseline_run": ev["baseline_run"],
                    "this_run_full_set": accuracy_by_view(own),
                    "baseline_full_set": accuracy_by_view(base),
                    "this_run_common_set": accuracy_by_view(common),
                    "baseline_common_set": accuracy_by_view(
                        common[["view", "frame", "slot"]].assign(correct=common.correct_base)
                    ),
                }
                summary["decision_rules"]["variant_accuracy_vs_baseline"] = accuracy_rule(
                    summary["warning_crop_accuracy"]["this_run_common_set"],
                    summary["warning_crop_accuracy"]["baseline_common_set"],
                    ev,
                )
        elif "female_reference" in rc:
            gt_source = Path(paths.asset_root) / rc["female_reference"]
            gt_path = r1.verified(rec, "female_reference", gt_source, r1.sha256(gt_source))
            gt = pd.read_csv(gt_path, low_memory=False)
            caches = {view: inputs[f"cache_{view}"]["path"] for view in VIEWS}
            anchors_df, match_counts = female_anchors(gt, caches, source, rc["match_px"])
            scored = anchors_df[weights[anchors_df.frame.to_numpy()] > 0]
            base = score_anchors(frames["N_carry"], anchors_df, weights)
            female = base["female_identity"]
            results = {
                label: score_anchors(
                    frame, anchors_df, weights, None if label == "A0_chain" else female
                )
                for label, frame in frames.items()
            }
            for label in ("N_carry", "S", "G"):
                if label in frames:
                    results[f"{label}|gap_anchors"] = score_anchors(
                        frames[label], anchors_df, weights, female, region=in_gap
                    )
            for label, frame in frames.items():
                anchors_df[f"name_{label}"] = frame[
                    anchors_df.frame.to_numpy(), anchors_df.slot.to_numpy()
                ]
            anchors_df["scored"] = weights[anchors_df.frame.to_numpy()] > 0
            anchors_df["in_gap"] = in_gap[anchors_df.frame.to_numpy()]
            anchors_df["in_warning"] = warning[anchors_df.frame.to_numpy()]
            anchors_df.to_csv(dest / "female_anchors.csv", index=False)
            summary.update(
                {
                    "results": results,
                    "female_anchor_matching": match_counts,
                    "scored_anchors": len(scored),
                    "female_identity": female,
                    "male_diagnostics": male_diagnostics(names, lower, female),
                    "decision_rules": {"K_1_2_anchor_rule": rule_named(results["N_carry"], ev)},
                }
            )
            reference_of = None
            ref_names = None
        else:
            # No reference in this run: identity tracks and label-free
            # diagnostics are written for scoring elsewhere.
            summary["decision_rules"] = {}
            reference_of = None
            ref_names = None

        # V: separation during training.
        figures = {}
        for name in cfg["train"]["models"]:
            with np.load(out / "models" / f"{name}_history.npz") as h:
                history = {k: h[k] for k in h.files}
            log = json.loads((out / "models" / f"{name}_log.json").read_text())
            reference = None
            if reference_of is not None:
                reference = reference_of(
                    history["validation_frame"].astype(np.int64),
                    history["validation_slot"].astype(np.int64),
                )
            separation_figure(
                lab,
                history,
                log,
                name,
                ev["figure_evaluations"],
                dest / "figures",
                reference,
                ref_names,
            )
            best = int(np.argmax(history["silhouette"]))
            figures[name] = {
                "evaluations": int(history["steps"].size),
                "best_step": int(history["steps"][best]),
                "best_silhouette": float(history["silhouette"][best]),
                "last_silhouette": float(history["silhouette"][-1]),
                "silhouette_at_step_0": float(history["silhouette"][0]),
            }
            if reference is not None:
                assessed = reference >= 0
                agree = []
                for e in (0, best):
                    cl = history["clusters"][e][assessed]
                    rf = reference[assessed]
                    purity = sum(
                        np.bincount(rf[cl == c], minlength=3).max()
                        for c in range(3)
                        if (cl == c).any()
                    ) / max(assessed.sum(), 1)
                    agree.append(
                        {
                            "step": int(history["steps"][e]),
                            "cluster_purity_vs_reference": float(purity),
                        }
                    )
                figures[name]["cluster_purity"] = agree
        summary["training"] = figures
        summary["train_summary"] = json.loads((out / "train_summary.json").read_text())

        for label in ("N_carry", "S", "G"):
            if label in frames:
                path = dest / f"identity_tracks_{label}.npz"
                identity_tracks(path, frames[label], xyz, times, identity_labels)
                rec.output(f"identity_tracks_{label}", path, sha256=r1.sha256(path))
        r1.write_json(dest / "summary.json", summary)
        rec.output("summary", dest / "summary.json")
        rec.param(
            "reference_use",
            "evaluation only"
            + (
                "; fixed 298-300 s anchor names interval 0"
                if recording == "p11"
                else "; female anchors score the female's slot only"
                if "female_reference" in rc
                else "; no reference read in this run"
            ),
        )
    status(out, "evaluated")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("bundle", "extract", "prepare", "train", "evaluate"))
    parser.add_argument("--recording", required=True)
    parser.add_argument("--config", help="configuration file; default: Registration 4's")
    args = parser.parse_args(argv)
    cfg, rc, paths, root, out = settings(args.recording, args.config)
    {
        "bundle": cmd_bundle,
        "extract": cmd_extract,
        "prepare": cmd_prepare,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
    }[args.command](cfg, rc, paths, root, out, args.recording)


if __name__ == "__main__":
    main()
