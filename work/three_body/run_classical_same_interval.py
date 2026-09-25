"""Replay the frozen classical functions on one bounded SL window.

Stage A sees native times once per camera. All geometric stages see the same
local 0..N-1 index. This is the existing unconstrained C25 initialization;
it does not share the new tracker seed and does not certify physical IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import sys
import time
import zipfile
from dataclasses import dataclass, replace
from itertools import permutations
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parent
REPO = WORK.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

import c01_s16_predictions as c01  # noqa: E402
import c24_uncapped_tracklets as c24  # noqa: E402
import c25_variable_cardinality as c25  # noqa: E402
import c30_slot_stitch as c30  # noqa: E402
import c33_ray_rejection as c33  # noqa: E402
import c34_microflip as c34  # noqa: E402
import classical_viterbi_checked_cache as checked_solver  # noqa: E402

from medaka3d.calib import load_rig  # noqa: E402
from medaka3d.reconstruct import Reconstruction  # noqa: E402
from medaka3d.track import UNTRACKED, propagate_to_views  # noqa: E402
from medaka3d.types import VIEWS  # noqa: E402

STAGE = "classical_same_interval_unseeded_v1"
EXECUTION_READY = True
EXPERIMENT = "20251225_K_1_2_FCMC"
START, STOP, SCIENCE_START = 49800, 116200, 66400
LAMBDAS = (0.0, 0.1, 0.25, 0.5, 0.767, 2.0, 8.0)
NODES = ("pectoral", "mouth")
INPUT_ROOT = REPO / "work/generative_tracking_inputs"
CATALOG_SHA = "7bcfcd453ad345d7a2ca6269554f0aae6f158c430f8ae8dd223382549325181f"
SEED_REFERENCE = WORK / "long_seed_registration_49800_v1/registration.json"
SEED_REFERENCE_SHA = "34da64c0c61a4c4e589b3e38a01627baff0e934b1d1a4325a280645e8a132b40"
CALIBRATION = (
    "calibration/fixed_refractive_192x192x186/"
    "phase1-calibration-preserve-20260802T180200JST-60e13a7746d1/calibration"
)
PRIMARY_SOURCE_SHA = {
    "work/three_body/classical_viterbi_checked_cache.py": (
        "fc1d3a2efecf5c133cc0209f869b4e93bcc7e329af0fb4da3c5a141d229a57e6"
    ),
    "tools/c24_uncapped_tracklets.py": (
        "9e744375909419ee922b3fbc155e5e8623fcde5941bdc4f80e1577750be615e4"
    ),
    "tools/c25_variable_cardinality.py": (
        "1b7662bd5c1235165f0aa9665ba899ee45c8a1b9acbfb27d629f5b63b5dfe7ad"
    ),
    "tools/c33_ray_rejection.py": (
        "77231493a749db6b55da55be75e71adda04ec17ede4adbedecc0f3ea0bef62f2"
    ),
    "tools/c30_slot_stitch.py": "823622b181859c2b56749f69480a3a64125ef41be8ba07183f0bad0ade234706",
    "tools/c34_microflip.py": "3e22c91899bede2b29c3710a3ce964c291b69a9aa3ce6e2bb014077b0ccbbb1a",
    "src/medaka3d/calib.py": "60849004cb53febe69f437e7afa8b18cb2b65213ba76ce7c921f7022d2be4274",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def signature(array):
    array = np.asarray(array)
    return dict(
        dtype=str(array.dtype),
        shape=list(array.shape),
        sha256=digest(np.ascontiguousarray(array).tobytes()),
    )


def frozen(array):
    array = np.array(array, copy=True)
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class Window:
    start: int
    stop: int

    def __post_init__(self):
        require(
            type(self.start) is int and type(self.stop) is int and 0 <= self.start < self.stop,
            "invalid window",
        )

    @property
    def size(self):
        return self.stop - self.start

    @property
    def native_frames(self):
        return np.arange(self.start, self.stop, dtype=np.int64)


def validate_detections(detections):
    frames = detections.frame_idx
    require(
        detections.experiment == EXPERIMENT
        and detections.view in VIEWS
        and tuple(detections.nodes) == c01.NODES,
        "foreign SL experiment/view/nodes",
    )
    require(
        frames.ndim == 1
        and frames.dtype == np.int32
        and np.all(np.diff(frames.astype(np.int64)) >= 0)
        and np.all((frames >= 0) & (frames < detections.n_frames)),
        "invalid sorted native SL index",
    )
    require(
        detections.points.shape == (len(frames), 3, 2)
        and detections.points.dtype == np.float32
        and detections.score.shape == (len(frames),)
        and detections.score.dtype == np.float32,
        "SL shape/type differs",
    )


def clip_native(detections, window):
    """Keep stable detector row order; outside observations never reach a stage."""
    validate_detections(detections)
    require(window.stop <= detections.n_frames, "window outside SL")
    lo, hi = np.searchsorted(detections.frame_idx, [window.start, window.stop])
    clipped = replace(
        detections,
        frame_idx=frozen(detections.frame_idx[lo:hi]),
        points=frozen(detections.points[lo:hi]),
        score=frozen(detections.score[lo:hi]),
        n_frames=window.stop,
    )
    return clipped, (int(lo), int(hi))


def local_detections(native, window):
    validate_detections(native)
    require(
        native.n_frames == window.stop
        and np.all((native.frame_idx >= window.start) & (native.frame_idx < window.stop)),
        "unclipped observations reached local conversion",
    )
    return replace(native, frame_idx=frozen(native.frame_idx - window.start), n_frames=window.size)


def detection_signatures(detections):
    return {key: signature(getattr(detections, key)) for key in ("frame_idx", "points", "score")}


def stage_a(native_by_view, window):
    require(tuple(native_by_view) == tuple(VIEWS), "camera order differs")
    local, labels, details = {}, {}, {}
    for view in VIEWS:
        native = native_by_view[view]
        local[view] = local_detections(native, window)  # guard before tracker call
        before = detection_signatures(native)
        assigned, effective = c24.run_tracker(
            native,
            ("mouth", "pectoral"),
            window_size=5,
            scoring="oks",
            min_match_points=0,
            frames=range(window.start, window.stop),
        )
        require(
            assigned.shape == native.frame_idx.shape
            and assigned.dtype == np.int64
            and np.all(assigned >= UNTRACKED),
            "stage A labels malformed",
        )
        require(before == detection_signatures(native), "stage A changed input SL")
        labels[view] = frozen(assigned)
        details[view] = dict(
            effective_config=effective,
            tracker_calls=1,
            input_signatures=before,
            label_signature=signature(assigned),
        )
    return local, labels, details


def degraded_window(local, size):
    """Exact c30 main arithmetic, evaluated only on the bounded local window."""
    degraded = np.zeros(size, bool)
    for view in VIEWS:
        require(local[view].n_frames == size, "degraded index scope differs")
        best = c30.best_scores(local[view], size)
        degraded |= (best >= 0) & (best < c30.SCORE_DEGRADED)
    # Keep sequential h=1 then h=2 updates, including their combined reach.
    for h in (1, 2):
        widened = degraded.copy()
        widened[h:] |= degraded[:-h]
        widened[:-h] |= degraded[h:]
        degraded = widened
    return degraded


def prepare_correspondence(local, labels, rig, window):
    require(tuple(rig.views) == tuple(VIEWS), "rig view order differs")
    uv, label_at, row_at, k = {}, {}, {}, {}
    count = 0
    for view in VIEWS:
        require(local[view].n_frames == window.size, "nonlocal correspondence input")
        uv[view], label_at[view], row_at[view], k[view] = c25.compact_layout(
            local[view], labels[view], NODES
        )
        count = max(count, int(labels[view].max()) + 1 if labels[view].size else 0)
    return dict(
        rays=c25.compact_rays(rig, uv, NODES),
        label_at=label_at,
        row_at=row_at,
        k=k,
        frames=np.arange(window.size, dtype=np.int64),
        base=count + 2,
        degraded=degraded_window(local, window.size),
    )


def storage_boundary(xyz, residual, source, slot_rays, margin):
    """Match the existing producers' NPZ float32 boundaries between stages."""
    return dict(
        xyz=np.asarray(xyz, np.float32),
        residual_mm=np.asarray(residual, np.float32),
        source=np.asarray(source).copy(),
        slot_rays=np.asarray(slot_rays).copy(),
        margin_mm=np.asarray(margin, np.float32),
        n_views_used=np.asarray(slot_rays).min(axis=1).astype(np.int8),
    )


def first_correspondence(source, reference):
    """Global column naming diagnostic only; never changes data or initialization."""
    source, target = np.asarray(source), np.asarray(reference["source_rows"])
    require(
        source.shape == target.shape == (3, 3)
        and source.dtype.kind in "iu"
        and target.dtype.kind in "iu",
        "invalid source reference",
    )
    matches = [list(p) for p in permutations(range(3)) if np.array_equal(source[list(p)], target)]
    return dict(
        first_source_rows=source.tolist(),
        reference_source_rows=target.tolist(),
        seed_slot_to_output_slot_permutations=matches,
        exact_initial_correspondence_match=bool(matches),
        reference_registration_sha256=reference["sha256"],
        global_permutation_applied=False,
        shared_seed_supplied=False,
        same_initial_pose_or_first_advance_asserted=False,
        physical_ID_truth=False,
    )


def replay_arm(context, local, rig, window, penalty, seed_reference):
    """One whole-window C25 call, followed by unmodified geometry/slot repairs."""
    require(penalty in LAMBDAS, "unregistered lambda")
    args = (
        context["rays"],
        NODES,
        context["label_at"],
        context["k"],
        context["frames"],
        penalty,
        4.0,
        context["base"],
    )
    cache_diagnostics = {}
    chosen, objective, inverse, tables, margin, pruned = checked_solver.solve(
        *args, diagnostics=cache_diagnostics
    )
    xyz, residual, source, rays = c25.chosen_geometry(
        context["rays"],
        NODES,
        tables,
        inverse,
        chosen,
        context["label_at"],
        context["row_at"],
        context["frames"],
        NODES[0],
    )
    raw = storage_boundary(xyz, residual, source, rays, margin * 3)
    corrected, ray_stats = c33.repair(raw, local, rig)
    ray_fixed = storage_boundary(
        corrected["xyz"],
        corrected["residual_mm"],
        corrected["source"],
        corrected["slot_rays"],
        raw["margin_mm"],
    )
    xyz64 = ray_fixed["xyz"].astype(np.float64)
    ok = np.flatnonzero((ray_fixed["n_views_used"] >= 2) & np.isfinite(xyz64).any(axis=(1, 2)))
    permutation, joins, refusals = c30.stitch(
        xyz64, ok, c30.WINDOW, c30.MARGIN_MM, c30.GAP_FRAMES, degraded=context["degraded"]
    )
    rows = np.arange(window.size)[:, None]
    stitched = storage_boundary(
        xyz64[rows, permutation],
        ray_fixed["residual_mm"][rows, permutation],
        ray_fixed["source"][rows, permutation],
        ray_fixed["slot_rays"][rows, permutation],
        ray_fixed["margin_mm"],
    )
    xyz64 = stitched["xyz"].astype(np.float64)
    ok = np.flatnonzero((stitched["n_views_used"] >= 2) & np.isfinite(xyz64).all(axis=(1, 2)))
    windows = c34.find_windows(xyz64, ok)
    repaired = c34.apply_windows(
        dict(
            xyz=xyz64,
            residual_mm=stitched["residual_mm"],
            slot_rays=stitched["slot_rays"],
            source=stitched["source"],
        ),
        windows,
    )
    final = storage_boundary(
        repaired["xyz"],
        repaired["residual_mm"],
        repaired["source"],
        repaired["slot_rays"],
        stitched["margin_mm"],
    )
    reconstruction = Reconstruction(
        EXPERIMENT,
        tuple(VIEWS),
        final["xyz"],
        final["source"],
        final["residual_mm"],
        final["margin_mm"],
        final["n_views_used"],
    )
    label_track = np.where(
        reconstruction.resolved[:, None], np.arange(3, dtype=np.int32)[None], UNTRACKED
    ).astype(np.int32)
    view_tracks = propagate_to_views(
        reconstruction,
        label_track,
        {v: local[v].start for v in VIEWS},
        {v: len(local[v].frame_idx) for v in VIEWS},
    )
    payload = dict(
        frame_idx=window.native_frames,
        **final,
        slot_track=label_track,
        **{f"track__{v}": view_tracks[v] for v in VIEWS},
        **{f"instance_frame_idx__{v}": local[v].frame_idx + window.start for v in VIEWS},
    )
    diagnostics = dict(
        lambda_mm=penalty,
        objective=float(objective),
        solve_calls=1,
        initialized_state="unconstrained first-frame unary costs",
        transition_cache_diagnostics=cache_diagnostics,
        pruned=pruned,
        ray_rejection=ray_stats,
        joins_local=joins,
        refusals_local=refusals,
        microflip_windows_local=windows,
        diagnostic_native_offset=window.start,
        initial_C25=first_correspondence(raw["source"][0], seed_reference),
        initial_C34=first_correspondence(final["source"][0], seed_reference),
    )
    stages = dict(
        c28=dict(**raw, chosen_state=chosen),
        c33=ray_fixed,
        c30=dict(**stitched, slot_perm=permutation),
    )
    return payload, diagnostics, stages


def no_cuda():
    require(
        "torch" not in sys.modules or not sys.modules["torch"].cuda.is_initialized(),
        "CPU replay initialized CUDA",
    )


def source_binding():
    from sleap_nn.tracking.tracker import Tracker

    require(Tracker.__module__ == "sleap_nn.tracking.tracker", "foreign tracker class")
    sources = {}
    for module in tuple(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name:
            path = Path(name).resolve()
            if path.suffix == ".py" and (
                path in {Path(__file__).resolve(), Path(checked_solver.__file__).resolve()}
                or any(path.is_relative_to(root) for root in (REPO / "tools", REPO / "src"))
            ):
                sources[str(path)] = digest(path.read_bytes())
    for relative, expected in PRIMARY_SOURCE_SHA.items():
        require(
            sources.get(str((REPO / relative).resolve())) == expected,
            "frozen classical source differs: " + relative,
        )
    no_cuda()
    return dict(
        sources=sources,
        packages={
            key: importlib.metadata.version(key)
            for key in ("numpy", "scipy", "sleap-nn", "sleap-io")
        },
    )


def capture(path, expected):
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    require(
        (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        and digest(data) == expected,
        "input changed or SHA differs",
    )
    return data, dict(
        path=str(path.resolve()), sha256=expected, size_bytes=len(data), mtime_ns=after.st_mtime_ns
    )


def read_npy_rows(archive, name, lo, hi, count, tail, dtype):
    """Deserialize only selected points/scores; compressed bytes may be skipped."""
    with archive.open(name + ".npy") as stream:
        version = np.lib.format.read_magic(stream)
        require(version in ((1, 0), (2, 0)), "unsupported NPY version")
        reader = (
            np.lib.format.read_array_header_1_0
            if version == (1, 0)
            else np.lib.format.read_array_header_2_0
        )
        shape, fortran, actual = reader(stream)
        require(
            shape == (count, *tail) and not fortran and actual == np.dtype(dtype),
            "SL member shape/order/type differs",
        )
        row_bytes = int(np.prod(tail, dtype=np.int64)) * actual.itemsize
        stream.seek(lo * row_bytes, 1)
        raw = stream.read((hi - lo) * row_bytes)
        require(len(raw) == (hi - lo) * row_bytes, "short SL member")
        return np.frombuffer(raw, actual).reshape(hi - lo, *tail).copy()


def decode_window(data, view, window):
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        require(
            set(archive.files)
            == {"experiment", "view", "frame_idx", "points", "score", "n_frames", "nodes"},
            "SL archive schema differs",
        )
        require(
            str(archive["experiment"]) == EXPERIMENT
            and str(archive["view"]) == view
            and tuple(archive["nodes"].tolist()) == c01.NODES,
            "SL identity differs",
        )
        frames, size = archive["frame_idx"], int(archive["n_frames"])
    require(
        frames.dtype == np.int32
        and frames.ndim == 1
        and np.all(np.diff(frames.astype(np.int64)) >= 0)
        and np.all((frames >= 0) & (frames < size))
        and window.stop <= size,
        "SL native index differs",
    )
    lo, hi = map(int, np.searchsorted(frames, [window.start, window.stop]))
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        points = read_npy_rows(archive, "points", lo, hi, len(frames), (3, 2), np.float32)
        score = read_npy_rows(archive, "score", lo, hi, len(frames), (), np.float32)
    detections = c01.Detections(
        EXPERIMENT, view, frozen(frames[lo:hi]), frozen(points), frozen(score), window.stop
    )
    validate_detections(detections)
    return detections, dict(
        selected_flat_rows=[lo, hi],
        native_interval=[window.start, window.stop],
        full_native_index_read=True,
        outside_points_scores_deserialized=False,
        selected_signatures=detection_signatures(detections),
    )


def load_inputs(window):
    data, catalog_record = capture(INPUT_ROOT / "staging_manifest.json", CATALOG_SHA)
    catalog = json.loads(data)
    index = {}
    for row in catalog["files"]:
        name = row["relative"].replace("\\", "/")
        require(
            name not in index and not Path(name).is_absolute() and ".." not in Path(name).parts,
            "unsafe or duplicated catalog entry",
        )
        index[name] = row
    seed_bytes, seed_record = capture(SEED_REFERENCE, SEED_REFERENCE_SHA)
    registration = json.loads(seed_bytes)
    require(
        registration["schema"] == "three_body_first_frame_seed_registration_v1"
        and registration["experiment"] == EXPERIMENT
        and registration["frame"] == window.start,
        "seed comparison reference differs",
    )
    seed_rows = np.asarray(registration["source_rows"])
    require(
        seed_rows.shape == (3, 3)
        and seed_rows.dtype.kind in "iu"
        and np.all(seed_rows >= 0)
        and all(len(set(seed_rows[:, i])) == 3 for i in range(3)),
        "seed source rows malformed",
    )
    seed_reference = dict(
        path=seed_record["path"],
        sha256=seed_record["sha256"],
        frame=window.start,
        source_rows=seed_rows.tolist(),
        original_limitation=registration["initial_identity_limitation"],
    )
    records = [catalog_record, seed_record]

    def registered(name):
        row = index[name]
        path = (INPUT_ROOT / name).resolve(strict=True)
        require(
            path.is_relative_to(INPUT_ROOT.resolve())
            and path == Path(row["local"]).resolve()
            and path.stat().st_size == row["bytes"],
            "foreign registered input",
        )
        captured, record = capture(path, row["sha256"])
        records.append(record)
        return captured

    native, selected = {}, {}
    for view in VIEWS:
        data = registered(f"work/c01_s16_predictions/bottomup/{EXPERIMENT}__{view}.npz")
        native[view], selected[view] = decode_window(data, view, window)
    manifest = json.loads(registered(CALIBRATION + "/calibration_manifest.json"))
    require(
        tuple(manifest["view_order"]) == tuple(VIEWS) and len(manifest["model_files"]) == 3,
        "calibration scope differs",
    )
    for view, model in zip(VIEWS, manifest["model_files"], strict=True):
        require(
            model["view"] == view and Path(model["path"]).name == model["path"], "unsafe rig member"
        )
        require(
            digest(registered(CALIBRATION + "/" + model["path"])) == model["sha256"],
            "rig SHA differs",
        )
    rig = load_rig(INPUT_ROOT / CALIBRATION)
    return native, rig, dict(records=records, selected=selected, seed_reference=seed_reference)


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run(args):
    require(EXECUTION_READY and args.execute, "classical replay is not enabled")
    output = args.output.resolve()
    require(
        output.is_relative_to(WORK.resolve()) and not output.exists(),
        "fresh workspace output required",
    )
    output.mkdir(parents=True)
    outputs, bound_inputs = {}, {}

    def save(name, value):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            if name.endswith(".npz"):
                np.savez_compressed(stream, **value)
            else:
                stream.write(
                    (json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n").encode()
                )
        outputs[name] = dict(
            path=str(path), sha256=digest(path.read_bytes()), size_bytes=path.stat().st_size
        )

    started, status = time.time(), "failed"
    try:
        window = Window(START, STOP)
        sources = source_binding()
        bound_inputs.update(sources["sources"])
        native, rig, inputs = load_inputs(window)
        bound_inputs.update({row["path"]: row["sha256"] for row in inputs["records"]})
        plan = dict(
            schema=STAGE,
            process_interval=[START, STOP],
            process_frames=window.size,
            science_interval=[SCIENCE_START, STOP],
            lambdas=LAMBDAS,
            inputs=inputs,
            source_binding=sources,
            initial_seed_shared=False,
            initialization="unconstrained C25 first-frame costs",
            original_C25_objective_preserved=True,
            transition_cache_identity_bug_corrected=True,
            old_full_recording_path_reproduction_claimed=False,
            future_horizon_exclusive=STOP,
            time_chunking=False,
            gated_tracks_recomputed=False,
            gated_track_reason=(
                "gated IDs never feed C33/C30/C34 corrections; final C34-label retained"
            ),
            outside_observations_used=False,
            no_physical_ID_truth=True,
        )
        save("plan.json", plan)
        local, labels, details = stage_a(native, window)
        save("stage_a.json", details)
        save(
            "stage_a.npz",
            {
                f"{key}__{view}": value
                for view in VIEWS
                for key, value in (
                    ("labels", labels[view]),
                    ("native_instance_frame", native[view].frame_idx),
                )
            },
        )
        context = prepare_correspondence(local, labels, rig, window)
        results = {}
        for penalty in LAMBDAS:
            payload, diagnostics, stages = replay_arm(
                context, local, rig, window, penalty, inputs["seed_reference"]
            )
            tag = f"lam{penalty:g}"
            for name, values in stages.items():
                save(f"{tag}/{name}.npz", dict(frame_idx=window.native_frames, **values))
            save(f"{tag}/C34-label.npz", payload)
            save(f"{tag}/diagnostics.json", diagnostics)
            results[tag] = dict(
                objective=diagnostics["objective"],
                array_signatures={k: signature(v) for k, v in payload.items()},
            )
        require(source_binding() == sources, "source/runtime changed")
        for row in inputs["records"]:
            info = Path(row["path"]).stat()
            require(
                (info.st_size, info.st_mtime_ns) == (row["size_bytes"], row["mtime_ns"]),
                "input stat changed",
            )
        no_cuda()
        save(
            "summary.json",
            dict(
                status="ok",
                stage=STAGE,
                process_interval=[START, STOP],
                science_interval=[SCIENCE_START, STOP],
                all7_lambdas=True,
                stage_a_calls=3,
                c25_whole_window_calls=7,
                manual_reinitializations=0,
                shared_seed=False,
                final_future_horizon=STOP,
                results=results,
                physical_ID_accuracy_validated=False,
            ),
        )
        status = "ok"
    except BaseException as exc:
        save(
            "failure.json",
            dict(type=type(exc).__name__, message=str(exc), completed_outputs=list(outputs)),
        )
        raise
    finally:
        save(
            "replay.provenance.json",
            dict(
                stage=STAGE,
                status=status,
                seconds=time.time() - started,
                inputs=bound_inputs,
                outputs=dict(outputs),
            ),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
