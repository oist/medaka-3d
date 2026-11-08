"""Freeze inputs and run the existing P-1-1 tracking chain in an isolated root."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from medaka3d import load_paths  # noqa: E402
from medaka3d.cohort import ExperimentId  # noqa: E402
from medaka3d.provenance import provenance  # noqa: E402
from medaka3d.types import VIEWS  # noqa: E402


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def configuration_path() -> Path:
    """Select an explicit run configuration, inherited by stage subprocesses."""
    configured = os.environ.get("MEDAKA3D_P11_CONFIG")
    return Path(configured).resolve(strict=True) if configured else HERE / "config.toml"


def settings():
    cfg = tomllib.loads(configuration_path().read_text(encoding="utf-8-sig"))
    paths = load_paths()
    root = paths.raw_data_dir.parent
    output = root / cfg["output"]["relative_to_data_parent"] / cfg["run_id"]
    return cfg, paths, root, output


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def prepare() -> None:
    cfg, paths, root, output = settings()
    output.mkdir(exist_ok=False)
    (output / "logs").mkdir()
    exp = cfg["experiment"]
    cohort = ExperimentId.parse(exp)
    frozen = {}
    with provenance("p11_input_freeze", output) as rec:
        rec.param("configuration", cfg)
        rec.param("cohort", {"group": cohort.group, "trial": cohort.trial, "day": cohort.day})
        source_dir = output / "source"
        for folder in ("src", "tools"):
            for source in (REPO / folder).rglob("*.py"):
                target = source_dir / source.relative_to(REPO)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        for source in (
            REPO / "uv.lock",
            REPO / "pyproject.toml",
            configuration_path(),
            Path(__file__),
        ):
            shutil.copy2(source, source_dir / source.name)
        write_json(
            output / "code_hashes.json",
            {
                str(p.relative_to(source_dir)): sha256(p)
                for p in sorted(source_dir.rglob("*"))
                if p.is_file()
            },
        )
        review = root / cfg["input"]["review_relative_to_data_parent"]
        review_hash = sha256(review)
        archive_dir = paths.asset_root / "groundtruth" / "identity_reviews" / exp / review_hash[:16]
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive = archive_dir / review.name
        if archive.exists():
            if sha256(archive) != review_hash:
                raise RuntimeError("Existing annotation archive has a different checksum")
        else:
            shutil.copy2(review, archive)
            if sha256(archive) != review_hash:
                raise RuntimeError("Annotation backup checksum mismatch")
        frozen["review"] = {"path": str(review), "sha256": review_hash, "backup": str(archive)}
        rec.input("review", review, sha256=review_hash)
        context = archive_dir / "annotation_context_20261031.json"
        if not context.exists():
            write_json(
                context,
                {
                    "source": str(review),
                    "sha256": review_hash,
                    "user_supplied_definitions": {
                        "watched": (
                            "Displayed in review GUI at any playback speed; "
                            "skipped frames are false."
                        ),
                        "analysed": "At or after the manually selected analysis-start frame.",
                        "uncertain": (
                            "Not used in P-1-1; false does not independently "
                            "establish decidability."
                        ),
                    },
                    "reviewer": "Not specified; annotation supplied by the user",
                },
            )
        timings = {}
        for view in VIEWS:
            source_cache = (
                paths.work_dir
                / "c01_s16_predictions"
                / cfg["input"]["cache_run"]
                / f"{exp}__{view}.npz"
            )
            slp = root / cfg["input"]["slp_relative_to_data_parent"] / f"{view}.slp"
            with np.load(source_cache, allow_pickle=False) as cached, h5py.File(slp, "r") as f:
                frames, instances, pp = f["frames"][:], f["instances"][:], f["pred_points"][:]
                nodes = [n["name"] for n in json.loads(f["metadata"].attrs["json"])["nodes"]]
                if nodes != [str(n) for n in cached["nodes"]]:
                    raise RuntimeError(f"Skeleton mismatch: {view}")
                if not np.array_equal(frames["frame_id"], np.arange(len(frames))):
                    raise RuntimeError(f"Non-contiguous SLP frame IDs: {view}")
                if not np.all(
                    instances["point_id_end"] - instances["point_id_start"] == len(nodes)
                ):
                    raise RuntimeError(f"Unexpected SLP point blocks: {view}")
                ix = instances["point_id_start"].astype(np.int64)[:, None] + np.arange(len(nodes))
                points = np.stack((pp["x"][ix], pp["y"][ix]), axis=-1)
                points[~pp["visible"][ix]] = np.nan
                native = frames["frame_idx"][instances["frame_id"]]
                order = np.argsort(native, kind="stable")
                checks = {
                    "frames": np.array_equal(native[order], cached["frame_idx"]),
                    "points": np.array_equal(
                        points[order].astype(np.float32), cached["points"], equal_nan=True
                    ),
                    "scores": np.array_equal(
                        instances["score"][order], cached["score"], equal_nan=True
                    ),
                    "experiment": str(cached["experiment"]) == exp,
                    "view": str(cached["view"]) == view,
                }
                if not all(checks.values()):
                    raise RuntimeError(f"Cache mismatch: {view}: {checks}")
                n_frames = int(cached["n_frames"])
                print(
                    f"Verified {view}: {len(instances):,} detections, {n_frames:,} frames",
                    flush=True,
                )
            target = (
                output
                / "pipeline"
                / "c01_s16_predictions"
                / cfg["input"]["cache_run"]
                / source_cache.name
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_cache, target)
            cache_hash = sha256(target)
            if cache_hash != sha256(source_cache):
                raise RuntimeError("Cache copy mismatch")
            slp_hash = sha256(slp)
            frozen[view] = {
                "cache": str(target),
                "cache_sha256": cache_hash,
                "slp": str(slp),
                "slp_sha256": slp_hash,
                "n_frames": n_frames,
                "verification": checks,
            }
            rec.input(f"slp_{view}", slp, sha256=slp_hash)
            rec.output(f"cache_{view}", target, sha256=cache_hash)
            timing_path = paths.raw_data_dir / exp / view / "frametimes.npy"
            t = np.load(timing_path, allow_pickle=False)
            if (
                t.shape != (2, n_frames)
                or not np.array_equal(t[0], np.arange(1, n_frames + 1))
                or not np.all(np.diff(t[1]) > 0)
            ):
                raise RuntimeError(f"Unexpected native timing: {view} {t.shape}")
            timings[view] = t[1]
            frozen[f"timing_{view}"] = {"path": str(timing_path), "sha256": sha256(timing_path)}
        timing_report = {}
        for view, times in timings.items():
            timing_report[view] = {
                "first_s": float(times[0]),
                "last_s": float(times[-1]),
                "delta_quantiles_s": np.quantile(np.diff(times), [0, 0.01, 0.5, 0.99, 1]).tolist(),
                "offset_to_top_quantiles_s": np.quantile(
                    times - timings["top_cam"], [0, 0.5, 1]
                ).tolist(),
            }
        write_json(output / "timing_inventory.json", timing_report)
        np.savez_compressed(output / "native_timing.npz", **timings)
        calibration = root / cfg["input"]["calibration_relative_to_data_parent"]
        for source in calibration.glob("*.json"):
            frozen[f"calibration_{source.name}"] = {"path": str(source), "sha256": sha256(source)}
        cache_provenance = (
            paths.work_dir
            / "c01_s16_predictions"
            / cfg["input"]["cache_run"]
            / f"c01_bottomup__{exp}.provenance.json"
        )
        shutil.copy2(cache_provenance, output / "input_cache.provenance.json")
        shutil.copy2(configuration_path(), output / "config.toml")
        write_json(output / "inputs.json", frozen)
    print(f"Prepared {output}", flush=True)


def stage(name: str) -> None:
    cfg, _, _, output = settings()
    if not (output / "inputs.json").is_file():
        raise RuntimeError("Prepare and verify inputs before execution")
    pipeline = output / "pipeline"
    wrapper_dir = output / "source" / "execution_wrappers"
    wrapper_dir.mkdir(parents=True, exist_ok=True)
    wrapper_copy = wrapper_dir / f"{name}_run_pipeline.py"
    shutil.copy2(Path(__file__), wrapper_copy)
    write_json(wrapper_dir / f"{name}.json", {"sha256": sha256(wrapper_copy), "configuration": cfg})
    if name != "c27":
        run_independent_arms(name)
        return
    invoke_stage(name, pipeline, cfg["pipeline"]["lambdas"])


def invoke_stage(name: str, destination: Path, lambdas: list[float]) -> None:
    cfg, paths, root, output = settings()
    pipeline = output / "pipeline"
    import medaka3d

    medaka3d.load_paths = lambda config=None: replace(paths, work_dir=destination)
    modules = {
        "c27": "c27_uncapped_tracklets_any",
        "c28": "c28_lambda_grid_transfer",
        "c33": "c33_ray_rejection",
        "c30": "c30_slot_stitch",
        "c34": "c34_microflip",
    }
    module = importlib.import_module(modules[name])
    original_load = module.c01.load
    module.c01.load = lambda work_dir, *a, **kw: original_load(pipeline, *a, **kw)
    if hasattr(module, "CALIBRATION"):
        module.CALIBRATION = root / cfg["input"]["calibration_relative_to_data_parent"]
    exp = cfg["experiment"]
    args = [modules[name], "--experiment", exp]
    if name != "c27":
        args += ["--lambdas", *map(str, lambdas)]
    if name == "c33":
        args += [
            "--arm-dir",
            str(pipeline / "c28_lambda_grid_transfer" / exp),
            "--arm-prefix",
            "C28-gated",
        ]
    if name == "c30":
        args += [
            "--arm-dir",
            str(pipeline / "c33_ray_rejection" / f"{exp}__C28-gated"),
            "--arm-prefix",
            "C33-gated",
            "--tag",
            "C33-on-C28-v4",
        ]
    if name == "c34":
        args += [
            "--arm-dir",
            str(pipeline / "c30_slot_stitch" / f"{exp}__C33-on-C28-v4"),
            "--tag",
            cfg["pipeline"]["chain_tag"],
        ]
    sys.argv = args
    module.main()


def stage_relative(name: str, cfg: dict) -> Path:
    exp = cfg["experiment"]
    return {
        "c28": Path("c28_lambda_grid_transfer") / exp,
        "c33": Path("c33_ray_rejection") / f"{exp}__C28-gated",
        "c30": Path("c30_slot_stitch") / f"{exp}__C33-on-C28-v4",
        "c34": Path("c34_microflip") / f"{exp}__{cfg['pipeline']['chain_tag']}",
    }[name]


def one_arm(name: str, lam: float) -> None:
    """Run one unchanged arm with private outputs and frozen shared inputs."""
    cfg, _, _, output = settings()
    exp = cfg["experiment"]
    pipeline = output / "pipeline"
    worker = pipeline / f"{name}_workers" / f"lam{lam:g}"
    worker.mkdir(parents=True, exist_ok=False)
    if name == "c28":
        label_relative = Path("c27_uncapped_tracklets_any") / exp / "mouth-pectoral_oks_serial"
        target_labels = worker / label_relative
        target_labels.mkdir(parents=True)
        for view in VIEWS:
            source = pipeline / label_relative / f"track2d__{view}.npy"
            target = target_labels / source.name
            shutil.copy2(source, target)
            if sha256(source) != sha256(target):
                raise RuntimeError("C27 label copy mismatch")
    invoke_stage(name, worker, [lam])


def run_independent_arms(name: str) -> None:
    """Parallelize independent lambda arms, never the temporal Viterbi sequence."""
    cfg, _, _, output = settings()
    exp = cfg["experiment"]
    workers = min(cfg["pipeline"]["lambda_workers"], max(1, (os.cpu_count() or 2) // 2))
    relative = stage_relative(name, cfg)
    destination = output / "pipeline" / relative
    destination.mkdir(parents=True, exist_ok=False)

    def execute(lam):
        log = output / "logs" / f"{name}_lam{lam:g}.log"
        started = time.time()
        with log.open("x", encoding="utf-8") as stream:
            result = subprocess.run(
                [
                    sys.executable,
                    "-u",
                    str(Path(__file__)),
                    "--stage",
                    f"{name}-one",
                    "--lambda",
                    str(lam),
                ],
                cwd=REPO,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode:
            raise RuntimeError(f"{name} lambda {lam:g} failed; inspect {log}")
        return {"lambda": lam, "seconds": time.time() - started, "log": str(log)}

    with provenance(f"p11_{name}_independent_arms", destination, name=name) as rec:
        rec.param("workers", workers)
        rec.param("temporal_chunking", False)
        rec.param("algorithm", f"Unchanged {name} main, one lambda per process")
        print(
            f"{name}: {workers} independent lambda processes; full temporal sequence in each",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(execute, cfg["pipeline"]["lambdas"]))
        merged = None
        for record in records:
            lam = record["lambda"]
            worker_result = output / "pipeline" / f"{name}_workers" / f"lam{lam:g}" / relative
            summary = json.loads((worker_result / f"{name}_summary.json").read_text())
            if merged is None:
                merged = {k: v for k, v in summary.items() if k != "arms"}
                merged["arms"] = {}
            merged["arms"].update(summary["arms"])
            for source in worker_result.glob("*.npz"):
                target = destination / source.name
                shutil.copy2(source, target)
                digest = sha256(target)
                if digest != sha256(source):
                    raise RuntimeError(f"{name} arm collection checksum mismatch")
                rec.output(source.stem, target, sha256=digest)
            rec.input(f"arm_{lam:g}_provenance", worker_result / f"{name}.provenance.json")
        merged["lambda_grid"] = cfg["pipeline"]["lambdas"]
        if name == "c28":
            merged["stage_a_labels"] = str(
                output
                / "pipeline"
                / "c27_uncapped_tracklets_any"
                / exp
                / "mouth-pectoral_oks_serial"
            )
        merged["execution"] = {
            "independent_lambda_workers": workers,
            "temporal_chunking": False,
            "arms": records,
        }
        write_json(destination / f"{name}_summary.json", merged)
        rec.param("arm_execution", records)


def run() -> None:
    cfg, _, _, output = settings()
    if not (output / "inputs.json").is_file():
        prepare()
    status = {"status": "running", "stages": []}
    status_path = output / "run_status.json"
    with provenance("p11_tracking_rerun", output) as rec:
        rec.input("frozen_inputs", output / "inputs.json", sha256=sha256(output / "inputs.json"))
        for name in cfg["pipeline"]["stages"]:
            started = time.time()
            status["active_stage"] = name
            write_json(status_path, status)
            print(f"Starting {name}", flush=True)
            log = output / "logs" / f"{name}.log"
            with log.open("x", encoding="utf-8") as stream:
                env = dict(
                    os.environ,
                    PYTHONIOENCODING="utf-8",
                    PYTHONUNBUFFERED="1",
                    OMP_NUM_THREADS="1",
                    MKL_NUM_THREADS="1",
                    OPENBLAS_NUM_THREADS="1",
                )
                result = subprocess.run(
                    [sys.executable, "-u", str(Path(__file__)), "--stage", name],
                    cwd=REPO,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    env=env,
                    check=False,
                )
            record = {
                "stage": name,
                "returncode": result.returncode,
                "seconds": round(time.time() - started, 2),
                "log": str(log),
            }
            status["stages"].append(record)
            write_json(status_path, status)
            rec.param(name, record)
            if result.returncode:
                status["status"] = "failed"
                write_json(status_path, status)
                raise RuntimeError(f"{name} failed; inspect {log}")
            print(f"Completed {name} in {record['seconds']:.0f} seconds", flush=True)
        status.update(status="complete", active_stage=None)
        write_json(status_path, status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=[
            "prepare",
            "c27",
            "c28",
            "c28-one",
            "c33",
            "c33-one",
            "c30",
            "c30-one",
            "c34",
            "c34-one",
            "run",
        ],
        default="run",
    )
    parser.add_argument("--lambda", dest="lambda_value", type=float)
    arg = parser.parse_args()
    if arg.stage == "prepare":
        prepare()
    elif arg.stage == "run":
        run()
    elif arg.stage.endswith("-one"):
        if arg.lambda_value is None:
            parser.error("--lambda is required for a single-arm run")
        one_arm(arg.stage.removesuffix("-one"), arg.lambda_value)
    else:
        stage(arg.stage)
