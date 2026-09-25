"""Fixed window 1/5/10 sensitivity on cropped C27; no selected best setting."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
from c27_replay import compare
from run import measure

from medaka3d.aquarium import sha256
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work/clean2d_video_comparison_v1/c27_sensitivity"
REPLAY = OUT.parent / "c27_replay"
CASES = ("cg20s", "known_21232", "known_302762", "known_333619")
EXP = "20251225_K_1_2_FCMC"


def replay_window(frames, points, scores, start, stop, window):
    import sleap_io as sio
    from sleap_nn.tracking.tracker import Tracker

    tracker = Tracker.from_config(
        window_size=window,
        candidates_method="local_queues",
        features="keypoints",
        scoring_method="oks",
        scoring_reduction="mean",
        track_matching_method="hungarian",
        max_tracks=None,
        tracking_target_instance_count=None,
        min_match_points=0,
    )
    skeleton = sio.Skeleton(["mouth", "pectoral"])
    labels = np.full(len(frames), -1, dtype=np.int64)
    legacy = labels.copy()
    names = {}
    reordered = []
    for frame in range(start, stop):
        lo, hi = np.searchsorted(frames, [frame, frame + 1])
        if lo == hi:
            continue
        batch = [
            sio.PredictedInstance.from_numpy(
                points_data=points[row, :2].astype(float),
                skeleton=skeleton,
                point_scores=np.ones(2),
                score=float(scores[row]),
            )
            for row in range(lo, hi)
        ]
        rows = {id(instance): row for row, instance in enumerate(batch)}
        result = tracker.track(batch, frame_idx=frame)
        order = []
        for offset, instance in enumerate(result):
            row = rows[id(instance)]
            order.append(row)
            if instance.track is None:
                continue
            name = instance.track.name
            if name not in names:
                names[name] = len(names)
            labels[lo + row] = names[name]
            legacy[lo + offset] = names[name]
        if order != list(range(len(batch))):
            reordered.append(dict(frame=int(frame), returned_input_row_order=order))
    return labels, legacy, reordered


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    if OUT.exists():
        raise ValueError(f"Refusing to overwrite sensitivity outputs: {OUT}")
    protocol_source = ROOT / "work/custom2d_comparison_v1/analysis_v3/protocol.json"
    cases = [
        c
        for c in json.loads(protocol_source.read_text(encoding="utf-8"))["cases"]
        if c["name"] in CASES
    ]
    OUT.mkdir(parents=True)
    protocol = dict(
        cases=cases,
        context_frames_each_side=332,
        windows=[1, 5, 10],
        baseline="Existing cropped window5, row-faithful readout, same current installation",
        replay_window5_input_directory=str(REPLAY.relative_to(ROOT)),
        changed_parameter_only="Tracker.from_config(window_size), last observations per track",
        remaining_config=dict(
            candidates_method="local_queues",
            features="keypoints",
            scoring_method="oks",
            scoring_reduction="mean",
            track_matching_method="hungarian",
            max_tracks=None,
            tracking_target_instance_count=None,
            min_match_points=0,
        ),
        readout="Original input row by instance object identity; legacy also recorded",
        metric_scope="Core only; adjacent links exclude first native core frame",
        metrics="Same-lab run.measure; same raw C01 detections, no pose or ID ground truth",
        selection="All fixed choices reported; no best setting selected",
        interpretation="Fragmentation/motion diagnostics, not biological identity accuracy",
        code_sha256=sha256(Path(__file__)),
        shared_measure_sha256=sha256(Path(__file__).with_name("run.py")),
    )
    # Freeze choices before loading outcomes or running any variant.
    write_json(OUT / "protocol.json", protocol)
    records = []
    with provenance("c27_cropped_window_sensitivity", OUT) as rec:
        for name, path in [
            ("protocol", OUT / "protocol.json"),
            ("code", Path(__file__)),
            ("measure", Path(__file__).with_name("run.py")),
            ("replay_comparison", Path(__file__).with_name("c27_replay.py")),
            ("case_ranges", protocol_source),
        ]:
            rec.input(name, path, sha256=sha256(path))
        import sleap_nn.tracking.tracker as installed

        package = Path(installed.__file__).parent
        for relative in ("tracker.py", "candidates/local_queues.py", "utils.py"):
            path = package / relative
            rec.input("installed_" + relative, path, sha256=sha256(path))
        raw = {}
        for view in VIEWS:
            path = ROOT / "work/generative_tracking_inputs/work/c01_s16_predictions/bottomup"
            path /= f"{EXP}__{view}.npz"
            with np.load(path, allow_pickle=False) as z:
                raw[view] = {k: z[k] for k in ("frame_idx", "points", "score")}
            rec.input("untracked_" + view, path, sha256=sha256(path))
        for case in cases:
            start, stop = case["start"] - 332, case["stop"] + 332
            dest = OUT / case["name"]
            dest.mkdir()
            for view in VIEWS:
                data = raw[view]
                lo, hi = np.searchsorted(data["frame_idx"], [start, stop])
                frames, points, scores = (data[k][lo:hi] for k in ("frame_idx", "points", "score"))
                basepath = REPLAY / case["name"] / f"{view}.npz"
                with np.load(basepath, allow_pickle=False) as z:
                    assert np.array_equal(frames, z["frame_idx"])
                    assert np.array_equal(np.arange(lo, hi), z["global_rows"])
                    baseline = z["row_faithful_labels"]
                rec.input(case["name"] + "_" + view + "_window5", basepath, sha256=sha256(basepath))
                for window in (5, 1, 10):
                    began = time.perf_counter()
                    if window == 5:
                        labels = baseline
                        with np.load(basepath, allow_pickle=False) as z:
                            legacy = z["legacy_labels"]
                        original = json.loads(
                            basepath.with_suffix(".json").read_text(encoding="utf-8")
                        )
                        reordered = original["reordered_returns"]
                        array_path = basepath
                    else:
                        labels, legacy, reordered = replay_window(
                            frames, points, scores, start, stop, window
                        )
                        array_path = dest / f"{view}__window{window}.npz"
                        np.savez_compressed(
                            array_path,
                            frame_idx=frames,
                            global_rows=np.arange(lo, hi),
                            row_faithful_labels=labels,
                            legacy_labels=legacy,
                        )
                        rec.output(
                            case["name"] + "_" + view + f"_window{window}",
                            array_path,
                            sha256=sha256(array_path),
                        )
                    core = (frames >= case["start"]) & (frames < case["stop"])
                    stats, _ = measure(frames, points, labels, case["start"], case["stop"])
                    result = dict(
                        case=case["name"],
                        view=view,
                        window=window,
                        core_half_open=[case["start"], case["stop"]],
                        context_half_open=[start, stop],
                        metrics=stats,
                        # Comparison's edge scope differs: remove entry frame explicitly.
                        versus_cropped_window5=compare(
                            frames,
                            baseline,
                            labels,
                            np.arange(lo, hi),
                            case["start"] + 1,
                            case["stop"],
                        ),
                        reordered_returns=reordered,
                        readout_different_rows_core=int(
                            np.count_nonzero(core & (labels != legacy))
                        ),
                        input_or_output_npz=str(array_path.relative_to(ROOT)),
                        npz_sha256=sha256(array_path),
                        reused_baseline=window == 5,
                        elapsed_seconds=time.perf_counter() - began,
                    )
                    write_json(dest / f"{view}__window{window}.json", result)
                    records.append(result)
                    print(
                        json.dumps(
                            dict(
                                case=case["name"],
                                view=view,
                                window=window,
                                seconds=round(result["elapsed_seconds"], 2),
                                runs=stats["contiguous_runs"],
                            )
                        ),
                        flush=True,
                    )
        summary = OUT / "summary.json"
        write_json(summary, dict(status="complete", records=records, protocol=protocol))
        rec.output("summary", summary, sha256=sha256(summary))


if __name__ == "__main__":
    main()
