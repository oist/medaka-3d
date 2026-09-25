"""Seal the bounded GPU co-execution plan before launching the short probe."""

import json
from pathlib import Path

from register_long_boundary_plan import ROOT, sha

from medaka3d.provenance import provenance


def main():
    prior = ROOT / "work/three_body/continuation_transient_200_single_v1"
    evidence = {}
    seal_path = prior / "continuation.provenance.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal["status"] != "ok":
        raise ValueError("solo reference must be successful")
    evidence[str(seal_path)] = sha(seal_path)
    for entry in seal["outputs"].values():
        path = entry["path"]
        if sha(path) != entry["sha256"]:
            raise ValueError("solo reference output changed")
        evidence[path] = entry["sha256"]
    for path in (Path(__file__).resolve(), ROOT / "work/three_body/CONCURRENT_GPU_PROBE_PROTOCOL.md"):
        evidence[str(path)] = sha(path)
    output = ROOT / "work/three_body/concurrent_gpu_probe_plan_v1"
    output.mkdir(exist_ok=False)
    with provenance("register_concurrent_gpu_probe", output, name="registration") as record:
        for path, digest in evidence.items():
            record.input(path, path, sha256=digest)
        target = output / "plan.json"
        plan = dict(protocol_sha256=evidence[str(ROOT / "work/three_body/CONCURRENT_GPU_PROBE_PROTOCOL.md")],
                    solo_reference=str(prior),
                    input_and_code_binding=seal["params"]["bindings"],
                    concurrent_D2_root_session=22901,
                    output="work/three_body/continuation_transient_200_concurrent_v1",
                    tracking_performed=False)
        target.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        record.output("plan", target, sha256=sha(target))
        for path, digest in evidence.items():
            if sha(path) != digest:
                raise ValueError("input changed during registration")
    print(json.dumps({"plan": str(target), "sha256": sha(target)}))


if __name__ == "__main__":
    main()
