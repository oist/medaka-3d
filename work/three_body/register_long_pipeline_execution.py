"""Bind measured concurrency gates before the first overlapping long phase."""

import json
from pathlib import Path

from register_long_boundary_plan import ROOT, sha

from medaka3d.provenance import provenance


def main():
    work = ROOT / "work/three_body"
    inputs = {}
    for relative, filename, stage in (
        ("long_continuation_through53120_independent_audit_v1", "audit.provenance.json",
         "long_continuation_boundary_independent_audit"),
        ("continuation_transient_200_concurrent_comparison_v1", "audit.provenance.json",
         None),
        ("concurrent_d2_solo_replay_1810_v1", "replay.provenance.json",
         "concurrent_d2_solo_replay"),
    ):
        root = work / relative
        path = root / filename
        seal = json.loads(path.read_text(encoding="utf-8"))
        if seal["status"] != "ok" or (stage is not None and seal["stage"] != stage):
            raise ValueError("pipeline prerequisite is not successfully sealed")
        inputs[str(path)] = sha(path)
        for item in seal["outputs"].values():
            if sha(item["path"]) != item["sha256"]:
                raise ValueError("prerequisite output differs from its seal")
            inputs[item["path"]] = item["sha256"]
    replay = json.loads((work / "concurrent_d2_solo_replay_1810_v1/summary.json")
                        .read_text(encoding="utf-8"))
    if (not replay["all_predictions_byte_exact"] or not replay["raw_images_exact"]
            or replay["image_count"] != 1810
            or not all(v["same_registered_background_SHA"]
                       for v in replay["comparisons"].values())):
        raise ValueError("concurrent detector numerical/input gate failed")
    for path in (Path(__file__).resolve(), work / "LONG_PIPELINE_EXECUTION_PLAN.md",
                 work / "long_multichunk_minute_plan_v1/plan.json",
                 work / "long_multichunk_minute_plan_v1/registration.provenance.json"):
        inputs[str(path)] = sha(path)
    stages = work / "inputs/long_source_masks_49800-348600_v1/chunks"
    for start in range(49800, 59760, 1660):
        path = stages / f"f{start:06d}-{start+1660:06d}/attempt_0001/staging.provenance.json"
        seal = json.loads(path.read_text(encoding="utf-8"))
        if seal["status"] != "ok" or seal["stage"] != "prepare_independent_mask_chunk":
            raise ValueError("all six initial input chunks must be successful")
        inputs[str(path)] = sha(path)
    output = work / "long_pipeline_execution_registration_v1"
    output.mkdir(exist_ok=False)
    with provenance("register_first_long_pipeline_execution", output, name="registration") as rec:
        for path, digest in inputs.items():
            rec.input(path, path, sha256=digest)
        plan = dict(
            execution_plan_sha256=inputs[str(work / "LONG_PIPELINE_EXECUTION_PLAN.md")],
            tracking_parent="long_continuation_through53120_v1",
            tracking_output="long_continuation_through59760_v1",
            next_tracking_interval=[53120, 59760],
            next_detector_interval=[59760, 66400],
            numerical_gates_passed=True, biological_identity_truth=False,
            all_three_ID_goal_complete=False, processes_started=False,
        )
        path = output / "plan.json"
        path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        rec.output("plan", path, sha256=sha(path))
        for path, digest in inputs.items():
            if sha(path) != digest:
                raise ValueError("pipeline evidence changed during registration")
    print(json.dumps(dict(output=str(output), plan_sha256=sha(output / "plan.json"))))


if __name__ == "__main__":
    main()
