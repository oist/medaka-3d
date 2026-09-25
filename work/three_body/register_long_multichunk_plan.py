"""Register the first same-process four-chunk extension of the long tracker."""

import json
from pathlib import Path

from register_long_boundary_plan import LAB, ROOT, sha

from medaka3d.provenance import provenance


def main():
    prior = ROOT / "work/three_body/long_continuation_through53120_v1"
    seal_path = prior / "continuation.provenance.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal["status"] != "ok" or seal["stage"] != "three_body_long_continuation":
        raise ValueError("parent must be successfully sealed")
    evidence = {str(seal_path): sha(seal_path)}
    for role, filename in (
        ("summary", "summary.json"), ("candidate", "candidate.npz"),
        ("consumed_chunks", "consumed_chunks.json"),
        ("checkpoint_manifest", "checkpoint/manifest.json"),
        (str(prior / "checkpoint/arrays.npz"), "checkpoint/arrays.npz"),
        (str(prior / "checkpoint/metadata.json"), "checkpoint/metadata.json"),
    ):
        item = seal["outputs"][role]
        path = prior / filename
        if Path(item["path"]).resolve() != path or sha(path) != item["sha256"]:
            raise ValueError("parent sealed output differs")
        evidence[str(path)] = item["sha256"]
    for filename, digest in seal["params"]["bindings"]["code_sha256"].items():
        if sha(filename) != digest:
            raise ValueError("frozen tracking source differs")
        evidence[filename] = digest
    for path in (Path(__file__).resolve(), LAB / "long_input_protocol.json"):
        evidence[str(path)] = sha(path)
    summary = json.loads((prior / "summary.json").read_text(encoding="utf-8"))
    if (summary["processed_interval"] != [49800, 53120]
            or summary["logical_initialization_count"] != 1
            or summary["checkpoint_restore_count"] != 1
            or summary["external_pose_reset_count"] != 0):
        raise ValueError("parent lifecycle differs")
    plan = dict(
        schema="three_body_long_multichunk_execution_plan_v1",
        full_target_interval=[49800, 348600],
        parent_phase=str(prior),
        parent_processed_interval=[49800, 53120],
        next_native_interval=[53120, 59760],
        output_interval=[49800, 59760],
        new_input_intervals=[[53120 + 1660*i, 54780 + 1660*i] for i in range(4)],
        frozen_input_and_code_binding=seal["params"]["bindings"],
        logical_initializations_expected=1,
        checkpoint_restores_expected=2,
        external_pose_resets_expected=0,
        parent_history_rows_expected=19913,
        boundary_audit_required_before_tracking="long_continuation_through53120_independent_audit_v1",
        execution="Load the four successfully sealed new chunks in one process, "
                  "restoring the full parent state once. Continue all six active "
                  "hypotheses across each input-file boundary, keeping all history.",
        validation=[
            "Past consumed chunk records and all 19 ordered history fields remain byte-exact prefixes.",
            "Native frames are exactly 49800 through 59759; no temporal subsampling.",
            "All four new chunk boundaries preserve the actual previous observation and tracker state.",
            "All 39 checkpoint arrays round-trip exactly; final 21 arrays match a full-history backtrace.",
            "Final direct source correspondences satisfy the inherited 2mm ray and 12px projection gates.",
            "Report direct and accepted coordinates separately from independent biological identity accuracy.",
        ],
        limits=[
            "The initial correspondence remains the same optimistic registered seed.",
            "This minute is an intermediate prefix of the full thirty-minute objective.",
            "Provisional past winning paths may change with later evidence.",
            "All-three human identity truth has not been supplied; zero swaps cannot be certified.",
            "No new uninterrupted 9960-frame counterfactual run is assumed.",
        ],
    )
    output = ROOT / "work/three_body/long_multichunk_minute_plan_v1"
    output.mkdir(exist_ok=False)
    with provenance("register_long_multichunk_execution", output, name="registration") as record:
        for filename, digest in evidence.items():
            record.input(filename, filename, sha256=digest)
        target = output / "plan.json"
        target.write_text(json.dumps(plan, indent=2, allow_nan=False), encoding="utf-8")
        record.output("plan", target, sha256=sha(target))
        for filename, digest in evidence.items():
            if sha(filename) != digest:
                raise ValueError("registration input changed")
        record.param("tracking_performed", False)
    print(json.dumps({"plan": str(target), "sha256": sha(target)}))


if __name__ == "__main__":
    main()
