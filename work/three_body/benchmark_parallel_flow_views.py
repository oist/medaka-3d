"""Fixed33 CPU flow comparison against sealed original 15-field outputs.

One native decode of34 times x3 cameras;66 untimed qualification flow calls,
then198 timed flow calls in alternating paired order. A single external pool
persists throughout. Timers exclude decode, byte checks, output writes and pool
startup. Concurrent tracker/D2 load is uncontrolled, so results are fixture-only.
No tracker construction/advance/fit, GPU operation, or active integration occurs.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

_REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_REPO / "labs/20260909_three_body"), str(_REPO / "src")]

import cv2
import numpy as np
import parallel_flow_views as parallel
import prepare_bounded_mouth_fit_packets as packets

from medaka3d.provenance import provenance

checks, REPO = packets.checks, packets.REPO
ROOT = REPO / "work/three_body/bounded_mouth_local_fit_packets_33_v1"
PACKET_SHA = "c1b4fc4b0f369f7f4a376f8bcd4a5b1736ad42efdcc0101118f9a407243da080"
PACKET_SOURCE = "5094ce9b8ba5b5a7bcbcb31abd838340cfc374709d3c14c8a15e976bdf336da3"
PARALLEL_SHA = "2d45db438facb9697b7f5d2a36f96ed552171a614fd0c32e79853fda0444917c"
STAGE = "parallel_flow_fixed33_CPU_benchmark_v1"
FRAMES = tuple(range(67663, 67696))
FIELDS = tuple(f.name for f in dataclasses.fields(parallel.frozen.FlowObservations))


def equal_flow(actual, expected):
    checks.require(set(actual) == set(expected) == set(FIELDS), "all15 flow fields required")
    for key in FIELDS:
        checks.require(checks.equal_bytes(actual[key], expected[key]), "flow byte mismatch: " + key)


def call_order(cycle, index):
    return ("serial", "parallel") if (cycle + index) % 2 == 0 else ("parallel", "serial")


def summarize(trials, frames=FRAMES, cycles=3):
    expected = [
        (c, n, m) for c in range(cycles) for i, n in enumerate(frames) for m in call_order(c, i)
    ]
    actual = [(r["cycle"], r["frame_idx"], r["method"]) for r in trials]
    checks.require(actual == expected, "timed paired schedule incomplete or reordered")
    checks.require(
        all(np.isfinite(r["seconds"]) and r["seconds"] > 0 for r in trials), "invalid timing"
    )
    table = {(r["cycle"], r["frame_idx"], r["method"]): r["seconds"] for r in trials}
    ratios = [
        table[c, n, "serial"] / table[c, n, "parallel"] for c in range(cycles) for n in frames
    ]
    methods = {}
    for method in ("serial", "parallel"):
        values = [r["seconds"] for r in trials if r["method"] == method]
        methods[method] = dict(
            calls=len(values),
            total_seconds=sum(values),
            median_seconds=float(np.median(values)),
            p10_p50_p90_seconds=np.percentile(values, [10, 50, 90]).tolist(),
        )
    return dict(
        methods=methods,
        paired_serial_over_parallel_p10_p50_p90=np.percentile(ratios, [10, 50, 90]).tolist(),
        paired_ratios=ratios,
    )


def member(name, hashes):
    path = ROOT / name
    checks.require(
        str(path) in hashes and checks.sha(path) == hashes[str(path)],
        "unsealed packet member: " + name,
    )
    return path


def own_flow_frame(frame):
    fields = {}
    for field in ("images_bgr", "masks"):
        fields[field] = {}
        for view, value in getattr(frame, field).items():
            if value is not None:
                value = value.copy()
                value.setflags(write=False)
            fields[field][view] = value
    return SimpleNamespace(frame_idx=int(frame.frame_idx), **fields)


def frame_signature(frame):
    return packets.signature(
        {
            field + "_" + view: value
            for field in ("images_bgr", "masks")
            for view, value in getattr(frame, field).items()
            if value is not None
        }
    )


def read34(reader, *, start=FRAMES[0] - 1, stop=FRAMES[-1] + 1):
    frames, rows = {}, []
    iterator = packets.iter_from(reader, start)
    try:
        for native in range(start, stop):
            frame = next(iterator)
            checks.require(frame.frame_idx == native, "decoded native order differs")
            rows.extend(packets.observation_evidence(frame, reader))
            frames[native] = own_flow_frame(frame)
    finally:
        iterator.close()
    return frames, rows


def fixture(hashes):
    packets.prior.bind(ROOT / "preparation.provenance.json", PACKET_SHA, hashes)
    producer = checks.verify(ROOT, "preparation.provenance.json", packets.STAGE, hashes)
    meta = checks.read(member("metadata.json", hashes))
    checks.require(
        meta["frames"] == list(FRAMES) and meta["raw_interval"] == [67662, 67696],
        "fixed native context differs",
    )
    for path, digest in list(hashes.items()):
        if path in producer["inputs"]:
            checks.require(
                producer["inputs"][path]["sha256"] == digest, "historical source differs"
            )
    audit_path = packets.prior.AUDIT / "audit.provenance.json"
    packets.prior.audit_input(producer, audit_path, hashes)
    checks.require(hashes[str(audit_path)] == packets.prior.AUDIT_SHA, "parent audit changed")
    audit = checks.read(audit_path)
    binding = meta["checkpoint_binding"]
    for path, digest in binding["original_binding"]["inputs"]["seed"][
        "SL_and_calibration_sha256"
    ].items():
        packets.prior.audit_input(producer, path, hashes)
        checks.require(hashes[str(Path(path).resolve())] == digest, "SL/calibration differs")
    rig = packets.load_rig(meta["calibration_directory"])
    data = {}
    for native in FRAMES:
        with np.load(member(f"f{native}_packet.npz", hashes), allow_pickle=False) as z:
            golden = {k: z["flow_" + k].copy() for k in FIELDS}
            pose = z["parent_pose"][0].copy()
        checks.require(
            int(golden["frame_idx"]) == native and int(golden["previous_frame_idx"]) == native - 1,
            "golden frame differs",
        )
        pose.setflags(write=False)
        for a in golden.values():
            a.setflags(write=False)
        data[native] = dict(pose=pose, golden=golden)
    reader, selected = packets.get_reader(binding, audit, hashes)
    checks.require(selected == meta["source_chunk"], "source chunk differs from packet")
    frames, rows = read34(reader)
    expected = checks.read(member("native_observation_evidence.json", hashes))
    checks.require(rows == expected, "native BGR/mask/SL/background evidence differs from packet")
    return meta, rig, data, frames, rows


def start_pool():
    started = time.perf_counter()
    executor = ThreadPoolExecutor(max_workers=3)
    try:
        barrier = threading.Barrier(3, timeout=30)

        def worker():
            barrier.wait()
            return threading.get_ident()

        futures = [executor.submit(worker) for _ in range(3)]
        identities = [f.result() for f in futures]
        checks.require(len(set(identities)) == 3, "three distinct workers required")
        return executor, time.perf_counter() - started, identities
    except BaseException:
        executor.shutdown(wait=True)
        raise


def run(output):
    output = Path(output).resolve()
    checks.require(output.is_relative_to(REPO / "work") and not output.exists(), "new work output")
    checks.require(checks.sha(parallel.__file__) == PARALLEL_SHA, "parallel helper changed")
    checks.require(checks.sha(packets.__file__) == PACKET_SOURCE, "packet reader helper changed")
    checks.require(
        checks.sha(parallel.frozen.__file__) == parallel.FROZEN_MOTION_SHA256,
        "frozen motion changed",
    )
    code = packets.comparison.source_registry.local_code()
    hashes, outputs = dict(code), {}
    output.mkdir(parents=True)
    before_threads = packets.torch.get_num_threads(), cv2.getNumThreads()
    executor = None
    calls, stored, trials = [], [], []
    summary = dict(status="incomplete", GPU_used=False, integrated_into_tracker=False)
    with provenance(STAGE, output, name="benchmark") as record:

        def emit(name, value=None):
            path = output / name
            if value is not None:
                path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
            digest = checks.sha(path)
            outputs[str(path)] = digest
            record.output(name, path, sha256=digest)

        try:
            for path, digest in code.items():
                target = output / "code_snapshot" / Path(path).relative_to(REPO)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(Path(path).read_bytes())
                checks.require(checks.sha(target) == digest, "source snapshot changed")
                emit(str(target.relative_to(output)))
            emit(
                "plan.json",
                dict(
                    frames=FRAMES,
                    raw_interval=[67662, 67696],
                    views=list(packets.VIEWS),
                    cycles=3,
                    timed_calls=198,
                    qualification_calls=66,
                    order_rule="serial first iff (cycle + pair_index) is even",
                    packet_sha256=PACKET_SHA,
                    parallel_sha256=PARALLEL_SHA,
                    workers=3,
                    thresholds_changed=False,
                    raw_decodes=102,
                    concurrent_live_tracker_and_D2=True,
                    active_processes_not_polled=True,
                ),
            )
            prepare_start = time.perf_counter()
            meta, rig, data, frames, raw_rows = fixture(hashes)
            packets.torch.set_num_threads(meta["torch_cpu_threads"])
            cv2.setNumThreads(meta["opencv_threads"])
            checks.require(cv2.getNumThreads() == meta["opencv_threads"], "OpenCV threads differ")
            runtime = dict(
                numpy=np.__version__,
                scipy=packets.scipy.__version__,
                opencv=cv2.__version__,
                torch=str(packets.torch.__version__),
                cuda=packets.torch.version.cuda,
            )
            checks.require(runtime == meta["runtime"], "original installed runtime differs")
            emit(
                "metadata.json",
                dict(
                    packet_metadata=meta,
                    runtime=runtime,
                    python=sys.version,
                    opencv_threads=cv2.getNumThreads(),
                    torch_cpu_threads=packets.torch.get_num_threads(),
                ),
            )
            emit("native_observation_evidence.json", raw_rows)
            signatures = {n: frame_signature(f) for n, f in frames.items()}
            pose_signatures = {n: packets.signature({"pose": d["pose"]}) for n, d in data.items()}
            preparation_seconds = time.perf_counter() - prepare_start
            executor, cold_seconds, thread_ids = start_pool()
            config = parallel.frozen.FlowConfig()

            def invoke(native, method, phase, cycle=-1, position=-1):
                checks.require(
                    cv2.getNumThreads() == meta["opencv_threads"], "thread setting changed"
                )
                previous, current, pose = frames[native - 1], frames[native], data[native]["pose"]
                start = time.perf_counter()
                if method == "serial":
                    flow = parallel.frozen.FlowObservations.from_frames(
                        previous, current, rig, pose, config
                    )
                else:
                    flow = parallel.from_frames_parallel(
                        previous, current, rig, pose, config, executor=executor
                    )
                seconds = time.perf_counter() - start
                actual = {k: np.asarray(getattr(flow, k)).copy() for k in FIELDS}
                row = dict(
                    phase=phase,
                    cycle=cycle,
                    frame_idx=native,
                    method=method,
                    position=position,
                    seconds=seconds,
                    all15_bytes_exact=False,
                )
                calls.append(row)
                try:
                    equal_flow(actual, data[native]["golden"])
                    for frame in (previous, current):
                        checks.require(
                            frame_signature(frame) == signatures[frame.frame_idx],
                            "flow mutated immutable raw/mask input",
                        )
                    checks.require(
                        packets.signature({"pose": pose}) == pose_signatures[native],
                        "flow mutated original parent pose",
                    )
                except BaseException:
                    np.savez_compressed(output / "failed_flow.npz", **actual)
                    emit("failed_flow.npz")
                    raise
                row["all15_bytes_exact"] = True
                stored.append(actual)
                if phase == "timed":
                    trials.append(row)

            for native in FRAMES:
                for method in ("serial", "parallel"):
                    invoke(native, method, "qualification")
            print(
                json.dumps(dict(event="all33_qualified", fields=15, all_bytes_exact=True)),
                flush=True,
            )
            for cycle in range(3):
                for i, native in enumerate(FRAMES):
                    for position, method in enumerate(call_order(cycle, i)):
                        invoke(native, method, "timed", cycle, position)
                print(
                    json.dumps(dict(event="cycle_complete", cycle=cycle, timed_calls=len(trials))),
                    flush=True,
                )
            summary = dict(
                schema=STAGE,
                status="ok",
                frames=FRAMES,
                raw_times=34,
                raw_images=102,
                qualification_calls=66,
                timed_calls=198,
                total_flow_calls=len(calls),
                all15_bytes_exact=True,
                input_buffers_unchanged=True,
                preparation_seconds=preparation_seconds,
                cold_pool_start_seconds=cold_seconds,
                worker_thread_ids=thread_ids,
                opencv_threads=meta["opencv_threads"],
                flow_config=dataclasses.asdict(config),
                **summarize(trials),
                GPU_used=False,
                fit_calls=0,
                advance_calls=0,
                active_integration=False,
                concurrent_tracker_D2_CPU_load_uncontrolled=True,
                timing_scope="complete flow call only; no decode/hash/output/startup",
                interpretation=(
                    "same fixed33 pairs repeated; no independent timing replicates "
                    "or end-to-end speed claim"
                ),
            )
            timing = summary["methods"]
            (output / "RESULT.md").write_text(
                "# 固定33組・視点並行flowのCPU比較\n\n"
                "[67663,67696) の同じ元親poseと34時刻・各3視点を使用。"
                "全264呼出しで保存flow15項目のdtype/shape/bytesが完全一致した。\n\n"
                f"逐次median {timing['serial']['median_seconds'] * 1000:.3f} ms、"
                f"並行median {timing['parallel']['median_seconds'] * 1000:.3f} ms。"
                "各native/反復で対応する逐次÷並行のmedianは "
                f"{summary['paired_serial_over_parallel_p10_p50_p90'][1]:.4f}。\n\n"
                "GPU trackerとD2が別processで稼働中の負荷を制御していない。"
                "反復3回は同じ33組で、独立な標本や追跡全体の速度測定ではない。"
                "復号・SHA比較・保存・pool開始費はflow timerから除いた。"
                "旧source、閾値、現行trackerへの導入、GPU計算は変更していない。\n",
                encoding="utf-8",
            )
            emit("RESULT.md")
        except BaseException as error:
            summary.update(status="failed", error=repr(error), completed_flow_calls=len(calls))
            raise
        finally:
            if executor is not None:
                executor.shutdown(wait=True)
            packets.torch.set_num_threads(before_threads[0])
            cv2.setNumThreads(before_threads[1])
            checks.require(
                (packets.torch.get_num_threads(), cv2.getNumThreads()) == before_threads,
                "caller thread settings not restored",
            )
            if stored:
                np.savez_compressed(
                    output / "all_call_outputs.npz",
                    **{k: np.stack([r[k] for r in stored]) for k in FIELDS},
                )
                emit("all_call_outputs.npz")
            emit("calls.json", calls)
            emit("summary.json", summary)
            checks.require(
                code == packets.comparison.source_registry.local_code(), "loaded source changed"
            )
            for path, digest in hashes.items():
                record.input(path, path, sha256=digest)
            packets.final_guard(hashes, outputs)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output), indent=2))
