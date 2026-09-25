"""New fixed chest+mouth support-only capacity check on sealed same-minute raw21.

Reuse the completed motion-only caller's input bindings and pure I/O. Read only
the successful solution's raw frame_idx and observed arrays; inspect all21 NPY
headers. Do not read fitted positions, graph arrays, point evidence or images.
No DP scoring, new geometry, GPU, source mutation, node/weight/cap search.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import count_identity_node_memory_states as counter
import run_tail_minute_identity as previous

WORK, REPO = previous.WORK, previous.REPO
START, STOP = previous.START, previous.STOP
METHODS, NODE_INDICES = previous.METHODS, (1, 0)
STAGE = "tail_minute_mouth_memory_reachability_v1"
PRIOR = WORK / "tail_minute_identity_motion_v1"
PRIOR_SHA = "bee96d995244cdd783bfa4da228d9090873f1c153f2f4bd5c8587fd4f7111f87"
PREVIOUS_SHA = "738622b1fba155f3bb92ccf9c42425c7d685da8830310efd8332864a76c0d7d4"
# Filled after the new pure counter's synthetic qualification, before execution.
COUNTER_SHA = "2994b164693d769e61028be7bf882c683a6e2d7c0a7edf738dcf5c43c2c2bdde"
require = previous.require


def load_inputs(evidence):
    """Small JSON/source only; raw support payloads are read in run, not here."""
    evidence.read(WORK / "run_tail_minute_identity.py", PREVIOUS_SHA)
    evidence.read(WORK / "count_identity_node_memory_states.py", COUNTER_SHA)
    source_sha = previous.file_sha(__file__)
    evidence.read(Path(__file__), source_sha)
    binding, comparison, _, _ = previous.load_inputs(evidence)
    sealed = previous.seal_read(
        PRIOR, "identity.provenance.json", previous.STAGES["solve"], PRIOR_SHA, evidence
    )
    plan = json.loads(previous.named_read(sealed, PRIOR, "plan.json", evidence))
    summary = json.loads(previous.named_read(sealed, PRIOR, "summary.json", evidence))
    require(
        plan["schema"] == summary["schema"] == previous.STAGES["solve"]
        and summary["status"] == "ok"
        and plan["binding"] == binding
        and summary["native_interval"] == [START, STOP]
        and summary["native_frames"] == STOP - START
        and summary["full_DP_executed"] is True
        and summary["raw_candidate21_modified"] is False
        and summary["sources_support_review_modified"] is False,
        "successful unchanged same-minute raw21 export required",
    )
    previous.input_member(sealed, WORK / "run_tail_minute_identity.py", PREVIOUS_SHA)
    for spec in previous.RUNS.values():
        previous.input_member(sealed, spec["root"] / "candidate.npz", spec["candidate"])
    references = {
        m: previous.output_member(sealed, PRIOR, m + "/raw_candidate21.npz") for m in METHODS
    }
    new_plan = dict(
        schema=STAGE,
        prior_identity_sha256=PRIOR_SHA,
        prior_input_binding=binding,
        native_interval=[START, STOP],
        methods=list(METHODS),
        node_indices=list(NODE_INDICES),
        anatomical_order=["chest", "mouth"],
        event_encoding="3*frame_offset+raw_slot",
        state_memory_axes="identity major then node in [1,0] order",
        one_shared_permutation_for_all_nodes=True,
        raw_candidate_references=references,
        caps=dict(
            maximum_layer_states=100000, maximum_total_states=1000000, maximum_total_edges=6000000
        ),
        future_cost="sum chest and mouth last-direct costs; each 0.2/3, scale0.8mm, 3log(delta)",
        unary=0,
        source_pins={
            "run_tail_minute_identity.py": PREVIOUS_SHA,
            "count_identity_node_memory_states.py": COUNTER_SHA,
            Path(__file__).name: source_sha,
        },
        geometry_and_ID_truth="fixed saved geometry; no new biological truth",
        support_scope="observed[:,:, [1,0]]; no positions or DP scores read",
    )
    return sealed, new_plan, comparison


def run(args):
    evidence, output = previous.Evidence(), previous.Output(args.output)
    began = datetime.now(UTC).isoformat()
    try:
        sealed, plan, comparison = load_inputs(evidence)
        output.save("plan.json", plan)
        for path, digest in evidence.inputs.items():
            p = Path(path)
            if p.suffix == ".py":
                target = output.root / "code_snapshot" / p.relative_to(REPO)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(evidence.read(p, digest))
                output.outputs[target.relative_to(output.root).as_posix()] = dict(
                    path=str(target), sha256=digest, size_bytes=target.stat().st_size
                )
        reports = {}
        for method in METHODS:
            raw = previous.named_read(sealed, PRIOR, method + "/raw_candidate21.npz", evidence)
            candidate, headers = previous.read_candidate(
                raw, support_only=True, first=START, start=START, stop=STOP
            )
            frames = candidate["frame_idx"]
            # Advanced indexing is last-axis only: T,raw-slot,K, not K,T,slot.
            support = candidate["observed"][:, :, NODE_INDICES].copy()
            require(support.shape == (STOP - START, 3, 2), "fixed MP support axes required")
            require(support[0].all(), "all initial two-node/three-ID cells must be direct")
            require(
                candidate["observed"].sum(axis=0).tolist()
                == comparison["methods"][method]["direct_by_slot_node"],
                "raw observed totals differ from four-way evidence",
            )
            before = previous.signature(support)
            started = time.perf_counter()
            result = counter.reachable_counts(frames, support, counter.Caps())
            elapsed = time.perf_counter() - started
            require(previous.signature(support) == before, "counter mutated support")
            output.save(method + "/headers.json", headers)
            output.save(
                method + "/support.npz", dict(frame_idx=frames, direct_nodes=support), arrays=True
            )
            output.save(
                method + "/counts.npz",
                dict(
                    frame_idx=result.frame_idx,
                    state_counts=result.state_counts,
                    edge_counts=result.edge_counts,
                ),
                arrays=True,
            )
            reports[method] = dict(
                completed=result.completed,
                stop=result.stop,
                counted_native_frames=len(result.frame_idx),
                total_states=result.total_states,
                total_edges=result.total_edges,
                maximum_layer_states=int(result.state_counts.max()),
                direct_by_slot_node=support.sum(axis=0).tolist(),
                all_six_direct_native=int(support.all(axis=(1, 2)).sum()),
                seconds=elapsed,
                decoded_raw21_fields=["frame_idx", "observed"],
                node_indices=list(NODE_INDICES),
            )
            print(json.dumps(dict(method=method, report=reports[method])), flush=True)
        output.save(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                native_interval=[START, STOP],
                native_frames=STOP - START,
                methods=reports,
                all_within_fixed_caps=all(r["completed"] for r in reports.values()),
                graph_scored=False,
                positions_read=False,
                cap_or_weight_changed=False,
                node_indices=list(NODE_INDICES),
                initial_all_six_direct=True,
                DP_fit_GPU_executed=False,
                candidate_modified=False,
            ),
        )
        evidence.guard()
        output.guard()
        record = dict(
            stage=STAGE,
            status="ok",
            started_at=began,
            finished_at=datetime.now(UTC).isoformat(),
            inputs=evidence.inputs,
            outputs=output.outputs,
        )
    except BaseException as error:
        record = dict(
            stage=STAGE,
            status="failed",
            started_at=began,
            finished_at=datetime.now(UTC).isoformat(),
            inputs=evidence.inputs,
            outputs=output.outputs,
            error=repr(error),
        )
        (output.root / "preflight.provenance.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        raise
    p = output.root / "preflight.provenance.json"
    p.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
    print(
        json.dumps(
            dict(status="ok", seal_sha256=previous.file_sha(p), outputs=len(output.outputs))
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    if args.metadata_only:
        evidence = previous.Evidence()
        _, plan, _ = load_inputs(evidence)
        evidence.guard()
        print(
            json.dumps(
                dict(
                    metadata_only=True,
                    inputs=len(evidence.inputs),
                    node_indices=plan["node_indices"],
                )
            )
        )
    else:
        parser.error("--output required") if args.output is None else run(args)


if __name__ == "__main__":
    main()
