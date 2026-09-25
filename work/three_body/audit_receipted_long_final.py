"""Fixed completed-30-minute technical audit caller; CPU, no tracking or process actions.

Final-owner exit evidence must pass before any terminal-array loader is called.
This module does not create that evidence. Missing owner-verifier integration fails
closed. Existing helpers retain their own final_audit_complete=False assertions.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import audit_long_final as support_api
import long_final_boundary_observations as boundary
import long_final_d2_assembly as d2
import long_final_native_samples as native
import long_final_native_spool as spool
import long_final_receipted_child_chain as chain_api
import numpy as np
import receipted_final_inputs as adapters
import receipted_final_report as report

REPO, WORK, VIEWS = adapters.REPO, adapters.WORK, adapters.VIEWS
FIRST, STOP, BLOCK, COUNT = 49800, 348600, 1660, 298800
STAGE = "receipted_long_final_technical_audit_v1"
require, equal = adapters.require, adapters.equal
PINS = {
    WORK / "LONG_FINAL_CLI_INTEGRATION_PLAN.md": (
        "1e475e6f5a3fa5c4cf7ea159127c45fe2171078d652d4ae3549c4616df9afe75"
    ),
    WORK / "RECEIPTED_FINAL_AUDIT_EXECUTION_PLAN.md": (
        "656ac49e92debedf768f1716e0b2d5a1e12c5f0c92a4d7b2846ccbbfef4764b3"
    ),
    WORK / "LONG_FINAL_VALIDATION_PLAN.md": (
        "a3cc8deffdf9b18db3a34ed3ee4c1112f406b5aa69ebe616e9b3fe38010a221d"
    ),
    Path(chain_api.__file__): "fa5a6b13b63ceec32b24abbdaa5ee6ca8ac464c0fc9be297feae15020546cdfb",
    Path(d2.__file__): "b07fd0540d31f48fec82277654e4c61ed02ab50488c24b73f77877357e54f233",
    Path(native.__file__): "f9c82e6fc7150edd20f424dd9fa75fda2ef321ad10f41076744ef7a7689e7791",
    Path(spool.__file__): "7ad2185f34c21877815818b397ab31c6810527c5c38d1fd1f65e1697a71258a2",
    Path(boundary.__file__): "da2c89ba92dc3d1450c79864b4f503d9e1268f5fb3ffeaec7ffce83ea4fcbcbe",
    Path(support_api.__file__): "9140be0d5401228f38fefa04b1d4d4e68ea672ab8a0793130fbd62952c3e61eb",
    Path(report.__file__): "fb441f3281e2a33b3be603f24ba38a3b7c642faf386b566a69b2a88cacfbe464",
    Path(adapters.__file__): "b7d2cfae0bb119e2c1e31f23a807a5319731adac719945620574f96a22556fd4",
    WORK / "receipted_final_owner_exit.py": (
        "7130fbea0c9d0b3481b3b001bcc059877ffde285cc9c8b1df5ce6d5fb0d636d4"
    ),
}


def now():
    return datetime.now(UTC).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while data := f.read(1024 * 1024):
            h.update(data)
    return h.hexdigest()


def canonical(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")


def valid_sha(value):
    require(
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
        "explicit lowercase SHA256 required",
    )
    return value


class Outputs:
    """Owned new output directory; no implicit overwrite, deletion, or pruning."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        require(
            self.root.is_relative_to(REPO) and not self.root.exists(),
            "new workspace output required",
        )
        self.root.mkdir(parents=True)
        self.records = {}

    def path(self, name):
        p = self.root / name
        require(p.resolve().is_relative_to(self.root) and p != self.root, "output escaped root")
        return p

    def add(self, name, *, expected=None, arrays=None):
        require(name not in self.records, "duplicate output role: " + name)
        p = self.path(name)
        digest = sha(p)
        require(expected is None or digest == expected, "new output digest differs")
        row = dict(path=str(p), sha256=digest, size_bytes=p.stat().st_size)
        if arrays is not None:
            row["arrays"] = arrays
        self.records[name] = row
        return dict(path=name, **{k: v for k, v in row.items() if k != "path"})

    def bytes(self, name, data):
        p = self.path(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("xb") as f:
            f.write(data)
        return self.add(name)

    def json(self, name, data):
        return self.bytes(name, canonical(data))

    def npz(self, name, arrays):
        p = self.path(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        signatures = spool.signatures(arrays)
        with p.open("xb") as f:
            np.savez_compressed(f, **arrays)
        with np.load(p, allow_pickle=False) as saved:
            require(
                len(saved.files) == len(arrays)
                and set(saved.files) == set(arrays)
                and spool.signatures({k: saved[k] for k in saved.files}) == signatures,
                "saved array roundtrip differs",
            )
        require(spool.signatures(arrays) == signatures, "save mutated caller arrays")
        return self.add(name, arrays=signatures)

    def guard(self):
        actual = {
            p.relative_to(self.root).as_posix()
            for p in self.root.rglob("*")
            if p.is_file() and p != self.root / "audit.provenance.json"
        }
        require(actual == set(self.records), "audit output membership differs")
        for name, row in self.records.items():
            p = self.path(name)
            require(
                p.stat().st_size == row["size_bytes"] and sha(p) == row["sha256"],
                "audit output changed: " + name,
            )


def source_paths():
    paths = {
        Path(__file__).resolve(),
        Path(adapters.__file__).resolve(),
        Path(report.__file__).resolve(),
    }
    for module in tuple(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name:
            p = Path(name).resolve()
            if p.suffix == ".py" and (
                p.parent == WORK
                or p.is_relative_to(REPO / "src")
                or p.is_relative_to(REPO / "labs/20260909_three_body")
            ):
                paths.add(p)
    return paths


def bind_sources(evidence, outputs):
    for path, digest in PINS.items():
        evidence.add(path, digest)
    sources = {}
    for path in sorted(source_paths()):
        digest = evidence.add(path)
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == digest, "source changed before snapshot")
        name = "code_snapshot/" + path.relative_to(REPO).as_posix()
        outputs.bytes(name, raw)
        sources[str(path)] = digest
    for path, digest in PINS.items():
        if path.suffix == ".md":
            outputs.bytes("plan_snapshot/" + path.name, path.read_bytes())
            require(
                sha(outputs.path("plan_snapshot/" + path.name)) == digest, "plan snapshot changed"
            )
    return sources


def source_guard(sources):
    require(
        source_paths() == {Path(p) for p in sources},
        "late local import was not snapshotted before use",
    )
    for name, digest in sources.items():
        require(sha(name) == digest, "executed source changed")


def owner_verifier():
    """No stub, fallback, process query or self-declared completion receipt."""
    path = WORK / "receipted_final_owner_exit.py"
    require(path.is_file() and sha(path) == PINS[path], "reviewed final owner verifier unavailable")
    module = importlib.import_module("receipted_final_owner_exit")
    require(
        Path(module.__file__).resolve() == WORK / "receipted_final_owner_exit.py",
        "owner verifier loaded from foreign path",
    )
    function = getattr(module, "verify_final_owner_exit", None)
    require(callable(function), "final owner verifier API is not connected")
    return function


def terminal_scope(completed):
    c, m = completed["chain"], completed["checkpoint_metadata"]
    require(
        completed["final_audit_complete"] is False
        and c["final_audit_complete"] is False
        and c["chain_version"] == "old_prefix_then_receipted_child_v1"
        and c["tracking_processes"] == 33
        and c["checkpoint_restores"] == 32
        and m["processed_frames"] == COUNT
        and m["previous_frame"] == STOP - 1
        and m["initialize_count"] == 1
        and m["restore_count"] == 32
        and len(completed["checkpoint"]) == 39
        and len(completed["candidate"]) == 21,
        "complete terminal scope differs",
    )


def support_totals(result, block_records, candidate_counts):
    expected = dict(
        native_frames=COUNT,
        node_denominator=COUNT * 9,
        pectoral_denominator=COUNT * 3,
        maximum_view_node_pairs=COUNT * 27,
        block_frames=BLOCK,
        blocks=180,
        frame_range=[FIRST, STOP],
        ray_gate_mm=2.0,
        native_pixel_gate=12.0,
        all_sources_and_observed_flags_pass=True,
        temporal_subsampling=False,
    )
    for key, value in expected.items():
        equal(result[key], value, "support denominator differs: " + key)
    require(len(block_records) == 180, "missing support callback")
    for key in ("source_rows_checked", "observed_nodes_checked"):
        require(sum(r[key] for r in block_records) == result[key], "support callback totals differ")
    require(
        result["observed_nodes_checked"]
        == sum(sum(r) for r in candidate_counts["direct_node_counts_fish_node"]),
        "support count differs from final candidate",
    )


def run(args):
    outputs = Outputs(args.output)
    started, timer = now(), time.perf_counter()
    evidence, timings, sources = support_api.Digests(), {}, {}
    current_stage, writer = "entry", None
    progress = {}

    def stage(name, function):
        nonlocal current_stage
        current_stage = name
        t = time.perf_counter()
        print(json.dumps(dict(event="final_audit_stage_started", stage=name)), flush=True)
        value = function()
        timings[name] = time.perf_counter() - t
        outputs.json(
            "stages/" + name + ".json", dict(stage=name, status="ok", seconds=timings[name])
        )
        return value

    try:
        for value in (
            args.expected_supervisor_sha256,
            args.expected_owner_exit_sha256,
            args.expected_d2_invocation_sha256,
        ):
            valid_sha(value)
        # Import the real verifier before snapshotting, but do not invoke any owner action.
        verify_owner = owner_verifier()
        sources = bind_sources(evidence, outputs)
        outputs.json(
            "registration.json",
            dict(
                schema=STAGE,
                arguments=vars(args),
                started_at=started,
                native_interval=[FIRST, STOP],
                block_frames=BLOCK,
                samples=native.fixed_sample_frames().tolist(),
                source_hashes=sources,
                independent_ID_truth=False,
                tracking_fit_detector_GPU_process_actions=False,
            ),
        )
        root = Path(args.supervisor).resolve(strict=True)
        header = stage(
            "completion_header",
            lambda: adapters.completed_header(root, args.expected_supervisor_sha256, evidence),
        )
        owner = stage(
            "owner_exit",
            lambda: verify_owner(
                args.owner_exit_evidence,
                args.expected_owner_exit_sha256,
                root,
                header,
                audit_started_at=started,
                evidence=evidence,
            ),
        )
        outputs.json("owner_exit_verification.json", owner)
        completed = stage(
            "completed_chain_terminal", lambda: chain_api.load_completed_final(root, evidence)
        )
        terminal_scope(completed)
        chain = completed["chain"]
        input_signatures = dict(
            candidate=spool.signatures(completed["candidate"]),
            checkpoint=spool.signatures(completed["checkpoint"]),
        )
        outputs.json("terminal_array_signatures.json", input_signatures)
        outputs.json(
            "chain.json",
            dict(
                ancestors=chain["ancestors"],
                phases=chain["phases"],
                final_run=str(chain["final_run"]),
                counts=completed["counts"],
                exact_backtrace=completed["exact_backtrace"],
                full_history_proofs_reused=True,
                historical_prefix_replay_repeated=False,
            ),
        )
        context = stage(
            "native_inputs", lambda: chain_api.load_native_sample_inputs(chain, evidence)
        )
        geometry = stage(
            "geometry_backgrounds",
            lambda: adapters.bind_geometry_and_backgrounds(completed, context, evidence),
        )
        outputs.json("geometry_backgrounds.json", geometry["proof"])
        refs = stage("endpoint_references", lambda: adapters.make_endpoint_refs(chain, evidence))
        outputs.json("endpoint_references.json", refs)
        consumed = adapters.named_json(
            chain["final_record"],
            "consumed_chunks",
            chain["final_run"] / "consumed_chunks.json",
            evidence,
        )
        assembly = stage(
            "D2_metadata",
            lambda: d2.load_completed_assembly(
                args.d2_invocation_seal,
                args.expected_d2_invocation_sha256,
                chain,
                consumed,
                evidence,
            ),
        )
        outputs.json("D2_metadata.json", assembly["summary"])
        d2_rows = []

        def d2_row(row):
            i = len(d2_rows)
            progress.update(view=row.get("view"), chunk=row.get("index"))
            require(
                i < 540
                and row["view"] == VIEWS[i // 180]
                and row["index"] == i % 180
                and row["start"] == FIRST + (i % 180) * BLOCK
                and row["stop"] == FIRST + (i % 180 + 1) * BLOCK
                and row["frames"] == BLOCK
                and row["exact_fields"] == 13,
                "D2 callback scope/order differs",
            )
            outputs.json(f"D2_rows/{i:03d}.json", row)
            d2_rows.append(row)

        assembled = stage("D2_packed", lambda: d2.verify_packed_assembly(assembly, on_chunk=d2_row))
        require(
            len(d2_rows) == 540
            and assembled["chunk_view_comparisons"] == 540
            and assembled["logical_field_comparisons"] == 7020
            and assembled["packed_arrays_verified"] is True,
            "D2 final totals differ",
        )
        outputs.json("D2_comparison.json", assembled)
        blocks = []

        def support_block(row, data):
            i = len(blocks)
            progress.update(support_block=i, native_interval=row.get("frame_range"))
            lo, hi = i * BLOCK, (i + 1) * BLOCK
            require(
                i < 180
                and row["start_index"] == lo
                and row["stop_index"] == hi
                and row["frame_range"] == [FIRST + lo, FIRST + hi],
                "support callback range/order differs",
            )
            require(
                set(data) == {"frame_idx", "valid_source", "ray_distance_mm", "native_pixel_error"}
                and data["frame_idx"].dtype == np.int64
                and np.array_equal(data["frame_idx"], np.arange(FIRST + lo, FIRST + hi))
                and data["valid_source"].dtype == bool
                and data["valid_source"].shape == (BLOCK, 3, 3, 3),
                "support callback arrays differ",
            )
            require(
                int(data["valid_source"].sum()) == row["source_rows_checked"],
                "support callback source count differs",
            )
            saved = outputs.npz(f"support/block_{i:03d}.npz", data)
            blocks.append(dict(**row, **saved))
            outputs.json(f"support/block_{i:03d}.json", blocks[-1])

        supported = stage(
            "all_support",
            lambda: support_api.verify_support_partitioned(
                completed["candidate"],
                context.nodes,
                geometry["rig"],
                block_frames=BLOCK,
                on_block=support_block,
            ),
        )
        support_totals(supported, blocks, completed["counts"])
        outputs.json("support_index.json", blocks)
        outputs.json("support_summary.json", supported)
        spool_bindings = dict(
            supervisor_sha256=args.expected_supervisor_sha256,
            owner_exit_sha256=args.expected_owner_exit_sha256,
            final_candidate_sha256=chain["final_record"]["outputs"]["candidate"]["sha256"],
            source_hashes=sources,
            native_interval=[FIRST, STOP],
        )
        writer = spool.NativeSpool(outputs.root / "native_spool", spool_bindings)
        samples = stage(
            "native_samples", lambda: native.read_native_samples(context, on_sample=writer)
        )
        manifest = stage("spool_finish", writer.finish)
        require(
            len(samples) == 1629
            and manifest["callback_records"] == 1629
            and manifest["saved_views"] == 1077,
            "native/spool denominator differs",
        )
        for name, record in manifest["outputs"].items():
            outputs.add(
                "native_spool/" + name, expected=record["sha256"], arrays=record.get("arrays")
            )
        spool_record = outputs.add("native_spool/spool.manifest.json")
        outputs.json("native_samples.json", samples)
        reader = spool.SpoolReader(
            outputs.root / "native_spool",
            expected_manifest_sha256=spool_record["sha256"],
            expected_bindings=spool_bindings,
        )
        reader.require_final_scope()
        flows = []

        def flow_row(row, arrays):
            i = len(flows)
            progress.update(flow_boundary=row.get("frame_idx"))
            require(
                i < 179
                and row["frame_idx"] == FIRST + (i + 1) * BLOCK
                and row["previous_frame"] == row["frame_idx"] - 1,
                "flow callback order differs",
            )
            saved = outputs.npz(f"flow/boundary_{row['frame_idx']:06d}.npz", arrays)
            flows.append(dict(**row, payload=saved))
            outputs.json(f"flow/boundary_{row['frame_idx']:06d}.json", flows[-1])

        flow = stage(
            "boundary_flow",
            lambda: boundary.verify_flow_boundaries(
                reader,
                completed["checkpoint"],
                completed["checkpoint_metadata"],
                geometry["rig"],
                backgrounds=geometry["backgrounds"],
                flow_config=geometry["protocol"]["flow_config"],
                opencv_threads=chain["binding"]["opencv_threads"],
                restore_starts=[r["stop"] for r in refs[:-1]],
                on_flow=flow_row,
            ),
        )
        require(
            len(flows) == 179
            and flow["restored_boundaries"] == 32
            and flow["same_process_reader_boundaries"] == 147,
            "flow final denominator differs",
        )
        outputs.json("flow_summary.json", flow)
        previous = stage(
            "checkpoint_previous",
            lambda: boundary.verify_previous_observations(
                reader,
                refs,
                completed["checkpoint_metadata"]["context"],
                backgrounds=geometry["backgrounds"],
                encoded_background_records=geometry["encoded_background_records"],
                evidence=evidence,
            ),
        )
        require(
            previous["count"] == 33 and len(previous["checkpoints"]) == 33,
            "previous denominator differs",
        )
        outputs.json("checkpoint_previous.json", previous)
        metrics = stage(
            "candidate_metrics", lambda: report.candidate_metrics(completed["candidate"])
        )
        residuals = stage("residual_metrics", lambda: report.residual_metrics(blocks, outputs.root))
        equal(metrics["direct_counts"], completed["counts"], "reported candidate counts differ")
        require(
            residuals["native_frames"] == COUNT
            and residuals["valid_source_count"] == supported["source_rows_checked"],
            "reported residual denominator differs",
        )
        for key, maximum in (
            ("ray_distance_mm", "maximum_supported_ray_distance_mm"),
            ("native_pixel_error", "maximum_supported_pixel_error"),
        ):
            stats = residuals["metrics"][key]
            require(
                stats["nonfinite_supported_count"] == 0
                and stats["finite_supported_count"] == supported["source_rows_checked"]
                and stats["max"] == supported[maximum],
                "reported residual finite denominator/maximum differs",
            )
        outputs.json("candidate_metrics.json", metrics)
        outputs.json("residual_metrics.json", residuals)

        def final_guards():
            equal(
                dict(
                    candidate=spool.signatures(completed["candidate"]),
                    checkpoint=spool.signatures(completed["checkpoint"]),
                ),
                input_signatures,
                "terminal arrays mutated during audit",
            )
            reader.verify_all()
            source_guard(sources)
            evidence.final_guard()
            outputs.guard()

        stage("final_guards", final_guards)
        details = dict(
            native_interval=[FIRST, STOP],
            native_frames=COUNT,
            chunks=180,
            tracking_processes=33,
            logical_initialization_count=1,
            checkpoint_restores=32,
            external_pose_resets=0,
            terminal_arrays=39,
            exact_backtrace_arrays=21,
            support=supported,
            counts=completed["counts"],
            sampled_native_frames=543,
            requested_decoded_images=1629,
            spooled_images=1077,
            boundary_flow_checks=179,
            restore_boundaries=32,
            reader_boundaries=147,
            previous_observations=33,
            D2_view_chunk_comparisons=540,
            D2_logical_field_comparisons=7020,
            independent_ID_truth=False,
            single_process_numeric_equivalence=False,
            tracker_initialization_restore_advance_fit_GPU_or_process_actions=False,
            codec_internal_seek_decode_work_measured=False,
            frozen_helper_completion_flags_unchanged=True,
            timings_seconds=timings,
            total_before_summary_seconds=time.perf_counter() - timer,
        )
        outputs.json(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                technical_completion=True,
                final_audit_complete=True,
                **details,
            ),
        )
        text = (
            "# Completed 30-minute technical audit\n\n"
            "All 298800 native frames, 33 processes (init1/restores32/reset0), terminal39/21, "
            "180 support blocks, 540 detector chunk/view comparisons, 543 native samples, "
            "179 flow boundaries and 33 previous observations passed.\n\n"
            "Support uses the unchanged 2mm / 12px gates. Missing and review entries remain "
            "in the denominators. Counts and all gap intervals are in candidate_metrics.json. "
            "This is technical completion and conditional availability; biological identity "
            "accuracy and single-process numerical equivalence are not established.\n"
        )
        outputs.bytes("RESULT.md", text.encode())
        outputs.guard()
        source_guard(sources)
        seal = dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=now(),
            technical_completion=True,
            final_audit_complete=True,
            inputs=evidence.paths,
            outputs=outputs.records,
            source_hashes=sources,
        )
        seal_path = outputs.root / "audit.provenance.json"
        raw = canonical(seal)
        with seal_path.open("xb") as f:
            f.write(raw)
        require(seal_path.read_bytes() == raw, "final seal roundtrip differs")
        outputs.guard()
        print(
            json.dumps(
                dict(
                    status="ok",
                    final_audit_complete=True,
                    outputs=len(outputs.records),
                    seal_sha256=sha(seal_path),
                )
            ),
            flush=True,
        )
        return seal
    except BaseException as error:
        abort_error = None
        if writer is not None:
            try:
                writer.abort(error)
            except BaseException as aborted:
                abort_error = dict(type=type(aborted).__name__, error=str(aborted))
        failed = dict(
            stage=STAGE,
            status="failed",
            started_at=started,
            finished_at=now(),
            failure_stage=current_stage,
            error_type=type(error).__name__,
            error=str(error),
            final_audit_complete=False,
            technical_completion=False,
            inputs=evidence.paths,
            known_outputs=outputs.records,
            source_hashes=sources,
            completed_stage_timings=timings,
            retained_partial_outputs=True,
            failure_context=progress,
            spool_abort_error=abort_error,
            partial_file_paths=[
                p.relative_to(outputs.root).as_posix()
                for p in outputs.root.rglob("*")
                if p.is_file()
            ],
            partial_files_not_promoted_to_verified_outputs=True,
        )
        # An exception after the final write must never leave an authoritative ok seal.
        (outputs.root / "audit.provenance.json").write_bytes(canonical(failed))
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "supervisor",
        "expected-supervisor-sha256",
        "owner-exit-evidence",
        "expected-owner-exit-sha256",
        "d2-invocation-seal",
        "expected-d2-invocation-sha256",
        "output",
    ):
        p.add_argument("--" + name, required=True)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
