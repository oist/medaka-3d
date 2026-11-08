"""Freeze the reviewed geometry-risk handoff without overwriting experiment files."""

import json
import os
import shutil
import tomllib
from pathlib import Path

import nbformat
import pandas as pd
from run_pipeline import HERE, REPO, settings, sha256, write_json

from medaka3d.provenance import provenance


def main():
    policy = tomllib.loads((HERE / "config_geometry_risk_v3.toml").read_text())
    os.environ["MEDAKA3D_P11_CONFIG"] = str(HERE / policy["source_config"])
    _, _, _, source = settings()
    run = source.parent / policy["run_id"]
    output = run / "handoff_v1"
    output.mkdir(exist_ok=False)
    validation = json.loads((run / "audit_v1/validation.json").read_text())
    notebook_path = HERE / "geometry_motion_review.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    code = [c for c in notebook.cells if c.cell_type == "code"]
    assert [c.execution_count for c in code] == list(range(1, len(code) + 1))
    assert not any(o.output_type == "error" for c in code for o in c.outputs)
    images = sum("image/png" in o.get("data", {}) for c in code for o in c.outputs)
    assert images == 2
    assert sha256(notebook_path) == sha256(run / "report_v1" / notebook_path.name)
    for name, expected in (("warning_intervals.csv", 129), ("lower_risk_intervals.csv", 130)):
        assert len(pd.read_csv(run / "report_v1" / name)) == expected
    html = (run / "report_v1/review_queue.html").read_text(encoding="utf-8")
    assert "WARNING_TABLE" not in html and "CLEAN_TABLE" not in html
    assert html.count("<tbody>") == 2
    inputs = json.loads((source / "inputs.json").read_text())
    for key in ("path", "backup"):
        assert sha256(Path(inputs["review"][key])) == inputs["review"]["sha256"]
    sources = [
        *HERE.glob("*geometry*risk*.py"),
        *HERE.glob("config_geometry_risk_v*.toml"),
        HERE / "GEOMETRY_MOTION_RESULT.md",
        HERE / "DATA_DICTIONARY.md",
        HERE / "README.md",
        HERE / "ROW_MAPPING_REPAIR.md",
        HERE / policy["source_config"],
        HERE / "run_pipeline.py",
        HERE / "evaluate.py",
        notebook_path,
        REPO / "src/medaka3d/eval/flip_risk.py",
        REPO / "tests/test_flip_risk.py",
        REPO / "uv.lock",
        REPO / "pyproject.toml",
        *(REPO / "labs/20261104_geometry_flip_risk").glob("*.py"),
        *(REPO / "labs/20261104_geometry_flip_risk").glob("*.md"),
    ]
    with provenance("p11_geometry_risk_handoff", output) as rec:
        for path in sources:
            target = output / "source" / path.relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            rec.input(path.relative_to(REPO).as_posix(), path, sha256=sha256(path))
        report = HERE / "GEOMETRY_MOTION_RESULT.md"
        shutil.copy2(report, run / "report_v1" / report.name)
        checks = {
            "experiment_status": json.loads((run / "status.json").read_text()),
            "audited_comparison_artifacts": validation["manifest_artifacts_verified"],
            "fresh_kernel_code_cells": len(code),
            "embedded_png_figures": images,
            "figures_visually_reviewed": ["comparison.png", "timeline.png"],
            "original_manual_review_and_curated_backup_unchanged": True,
            "focused_tests_passed": 33,
            "new_lab_and_isolation_checks_passed": 3,
            "unrelated_lab_failures": ["20260908_mask_volume", "20260909_three_body"],
            "full_suite_run": False,
            "appearance_classifier_used": False,
            "automatic_method_promoted": False,
            "limitations": "One recording; retrospective warning design; reference shares geometry",
        }
        write_json(output / "checks.json", checks)
        rec.output("checks", output / "checks.json", sha256=sha256(output / "checks.json"))
    files = {
        path.relative_to(run).as_posix(): {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in sorted(run.rglob("*"))
        if path.is_file()
    }
    write_json(output / "artifact_manifest.json", files)
    print(json.dumps(checks, indent=2), flush=True)
    print(run / "report_v1/review_queue.html", flush=True)


if __name__ == "__main__":
    main()
