"""Seal the CPU execution report without modifying the underlying diagnostic."""
from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from medaka3d.provenance import provenance  # noqa: E402


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    cpu = REPO / "work/three_body/detector_cpu_execution_profile_v1"
    batch = REPO / "work/three_body/detectron2_batch_profile"
    out = cpu / "report_v1"
    assert not (out / "report.provenance.json").exists()
    inputs = {}
    for folder in (cpu, batch):
        path = folder / "profile.provenance.json"
        seal = json.loads(path.read_text())
        assert seal["status"] == "ok"
        inputs[str(path)] = sha(path)
        for entry in seal["outputs"].values():
            path = Path(entry["path"])
            assert sha(path) == entry["sha256"]
            inputs[str(path)] = entry["sha256"]
    own_code = [Path(__file__), REPO / "src/medaka3d/provenance.py"]
    inputs.update({str(path): sha(path) for path in own_code})
    values = json.loads((cpu / "results.json").read_text())
    means = {k: statistics.mean(x["append_s"] for x in v)
             for k, v in values["repeats"].items()}
    saving = (means["original"] - means["rect_buffer_hash"]) * 4980 / 180
    result = dict(
        scope="report of sealed CPU execution diagnostic; no new inference",
        mean_append_seconds_per_180_images=means,
        append_speed_ratio=means["original"] / means["rect_buffer_hash"],
        sample_extrapolated_saving_per_chunk_s=saving,
        sample_extrapolated_fraction_of_old_chunk=saving / values["full_chunk_wall_s"],
        end_to_end_speedup_measured=False,
        predictor_or_gpu_executed=False,
        underlying_seals_and_all_recorded_outputs_verified=True,
    )
    with provenance("report_detector_cpu_execution", out, name="report") as record:
        for path, digest in inputs.items():
            record.input(path, path, sha256=digest)
        for source in own_code:
            target = out / "code_snapshot" / source.relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(source.read_bytes())
            record.output(target.relative_to(out).as_posix(), target, sha256=sha(target))
        output = out / "summary.json"
        with output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        for path in (output, out / "RESULT.md"):
            record.output(path.name, path, sha256=sha(path))
        assert all(sha(path) == digest for path, digest in inputs.items())
        assert all(sha(entry["path"]) == entry["sha256"] for entry in record.outputs.values())
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
