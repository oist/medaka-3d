"""Bind the next real chunk-boundary test to the completed first full state."""

import hashlib
import json
from pathlib import Path

from medaka3d.provenance import provenance

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "labs/20260909_three_body"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    prior = ROOT / "work/three_body/long_continuation_first1660_v1"
    path = prior / "continuation.provenance.json"
    seal = json.loads(path.read_text(encoding="utf-8"))
    if seal["status"] != "ok" or seal["stage"] != "three_body_long_continuation":
        raise ValueError("parent phase is not successfully sealed")
    evidence = {str(path): sha(path)}
    for role, filename in (
        ("summary", "summary.json"), ("candidate", "candidate.npz"),
        ("consumed_chunks", "consumed_chunks.json"),
        ("checkpoint_manifest", "checkpoint/manifest.json"),
    ):
        item = seal["outputs"][role]
        target = prior / filename
        if Path(item["path"]).resolve() != target or sha(target) != item["sha256"]:
            raise ValueError("parent output differs from its seal")
        evidence[str(target)] = item["sha256"]
    for target in (
        prior / "checkpoint/arrays.npz", prior / "checkpoint/metadata.json",
        LAB / "run_long_continuation.py", LAB / "long_tracking_inputs.py",
        LAB / "long_tracking_protocol.json", LAB / "long_input_protocol.json", Path(__file__),
    ):
        evidence[str(target)] = sha(target)
    plan = {
        "schema": "three_body_long_boundary_execution_plan_v1",
        "full_target_interval": [49800, 348600],
        "completed_parent_interval": [49800, 51460],
        "next_native_interval": [51460, 53120],
        "next_output_interval": [49800, 53120],
        "parent_phase": str(prior),
        "frozen_input_and_code_binding": seal["params"]["bindings"],
        "checks": [
            "All previous consumed chunk records are an identical prefix of the new records.",
            "The first new observation is 51460 with the checkpoint previous observation 51459.",
            "One logical initialization, one state restore and zero external pose resets.",
            "All original ordered beam/history rows are retained byte-exact before the new rows.",
            "All final state, direct node support and provisional output arrays pass validation.",
            "A later winning backtrace may revise the provisional first-ten-second result.",
        ],
        "next_scale": "After the actual boundary works, group multiple completed chunks into each phase to reduce repeated prefix validation; retain the full thirty-minute objective.",
        "limitations": [
            "This is operational evidence, not a biological identity accuracy measurement.",
            "All-three human correspondence is still required to evaluate and correct identity.",
            "No unbroken 3320-frame reference run is assumed or claimed in this boundary test.",
            "Twenty seconds is an intermediate prefix, not a redefinition of the full goal.",
        ],
    }
    output = ROOT / "work/three_body/long_boundary_resume_plan_v1"
    output.mkdir(exist_ok=False)
    with provenance("register_long_boundary_execution", output, name="registration") as record:
        for name, digest in evidence.items():
            record.input(name, name, sha256=digest)
        target = output / "plan.json"
        target.write_text(json.dumps(plan, indent=2, allow_nan=False), encoding="utf-8")
        record.output("plan", target, sha256=sha(target))
        for name, digest in evidence.items():
            if sha(name) != digest:
                raise ValueError("boundary evidence changed during registration")
        record.param("tracking_performed", False)
    print(json.dumps({"plan": str(target), "sha256": sha(target)}))


if __name__ == "__main__":
    main()
