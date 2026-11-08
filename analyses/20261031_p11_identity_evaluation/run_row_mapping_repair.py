"""Run the isolated row-mapping repair and compare it to the pinned r2 baseline."""

import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import audit_results
import run_pipeline

from medaka3d.provenance import provenance

HERE = Path(__file__).resolve().parent


def main():
    os.environ["MEDAKA3D_P11_CONFIG"] = str(HERE / "config_row_mapping_v1.toml")
    cfg, paths, root, output = run_pipeline.settings()
    comparison = cfg["comparison"]
    baseline = root / cfg["output"]["relative_to_data_parent"] / comparison["baseline_run"]
    if baseline.resolve() == output.resolve() or output.exists():
        raise FileExistsError("The repair needs a new run directory; the baseline is immutable")
    for filename, expected in comparison["baseline_hashes"].items():
        if run_pipeline.sha256(baseline / filename) != expected:
            raise ValueError(f"Pinned baseline checksum mismatch: {filename}")
    baseline_inputs = json.loads((baseline / "inputs.json").read_text())
    manifest = json.loads((baseline / "artifact_manifest.json").read_text())
    # Verify the frozen artifacts that will be read in the comparison, rather
    # than assuming a manifest's presence establishes their current contents.
    for relative, record in manifest.items():
        normalized = relative.replace("\\", "/")
        selected = normalized.startswith(
            ("scoring_rerun_v1/", "pipeline/c27_uncapped_tracklets_any/", "pipeline/c34_microflip/")
        )
        if selected and run_pipeline.sha256(baseline / normalized) != record["sha256"]:
            raise ValueError(f"Baseline artifact changed: {relative}")
    run_pipeline.prepare()
    inputs = json.loads((output / "inputs.json").read_text())
    for key, record in baseline_inputs.items():
        for field in ("sha256", "cache_sha256", "slp_sha256"):
            if field in record and inputs[key][field] != record[field]:
                raise ValueError(f"Repair input differs from baseline: {key}/{field}")
    for script in (Path(__file__), HERE / "evaluate.py", HERE / "audit_results.py"):
        shutil.copy2(script, output / "source" / script.name)
    with provenance("p11_row_mapping_repair_comparison", output) as rec:
        rec.param("baseline_run", comparison["baseline_run"])
        rec.param("configuration", cfg)
        rec.input("comparison_code", Path(__file__), sha256=run_pipeline.sha256(Path(__file__)))
        rec.param("scope", "Adapter source-row repair only; no tracking thresholds changed")
        run_pipeline.run()
        subprocess.run(
            [sys.executable, str(HERE / "evaluate.py"), "--source", "rerun", "--version", "v1"],
            cwd=run_pipeline.REPO,
            check=True,
        )
        score = output / cfg["evaluation"]["primary_scoring_directory"]
        baseline_score = baseline / cfg["evaluation"]["primary_scoring_directory"]
        report = audit_results.audit_scoring(score, cfg, Path(inputs["review"]["backup"]))
        baseline_paths = replace(paths, work_dir=baseline / "pipeline")
        raw, aligned = audit_results.compare_runs(
            cfg, baseline_paths, output, score, previous_score=baseline_score
        )
        report["previous_run_array_comparison"] = raw
        report["anchor_aligned_comparison"] = aligned
        report["c27_previous_comparison"] = audit_results.compare_c27(cfg, baseline_paths, output)
        report["baseline_run"] = comparison["baseline_run"]
        report["audit_source_sha256"] = run_pipeline.sha256(Path(audit_results.__file__))
        run_pipeline.write_json(score / "validation.json", report)
        rec.output("validation", score / "validation.json")
        rec.param("independent_scoring_checks_passed", True)
    print(f"Completed row-mapping repair comparison: {output}", flush=True)


if __name__ == "__main__":
    main()
