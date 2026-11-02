"""Publish local run summaries and source manifests beside the requested outputs."""

import importlib.metadata
import json
import platform
import shutil
from pathlib import Path

import nbformat
import pandas as pd
from run_pipeline import HERE, REPO, settings, sha256, write_json

from medaka3d.provenance import provenance


def main():
    config, _, _, output = settings()
    score = output / config["evaluation"]["primary_scoring_directory"]
    state = json.loads((output / "run_status.json").read_text())
    audit = json.loads((score / "validation.json").read_text())
    inputs = json.loads((output / "inputs.json").read_text())
    assert state["status"] == "complete"
    assert audit["summary_reconciled"]
    assert all(arm["discrepancies"] == 0 for arm in audit["arms"].values())
    assert sha256(Path(inputs["review"]["path"])) == inputs["review"]["sha256"]
    notebook = nbformat.read(output / "analysis.ipynb", as_version=4)
    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    assert all(c.execution_count is not None for c in code_cells)
    assert not any(o.output_type == "error" for c in code_cells for o in c.outputs)
    summary = pd.read_csv(score / "identity_summary.csv")
    primary = summary[summary.match_distance_mm.eq(config["evaluation"]["max_match_distance_mm"])]
    arms = json.loads((score / "arm_results.json").read_text())
    combined = primary.groupby("lambda_mm").sum(numeric_only=True)
    lines = [
        "Completed the full native-frame C27 -> C28 -> C33 -> C30 v4 -> C34 chain",
        "for all seven inherited settings, followed by reviewed-ID scoring.",
        "No new pose inference or appearance model was run. No lambda is promoted.",
        "",
        "| Lambda (mm) | ID agreement on scored time | Scored / expected time | "
        "All three correct / recording time |",
        "| --- | --- | --- | --- |",
    ]
    for lam, row in combined.iterrows():
        scored = row.correct_id_seconds + row.wrong_id_seconds
        joint = arms[f"{lam:g}"]["joint_correct_fraction"]
        lines.append(
            f"| {lam:g} | {row.correct_id_seconds / scored:.2%} | "
            f"{scored / row.expected_seconds:.2%} | {joint:.2%} |"
        )
    shown = config["evaluation"]["display_lambda"]
    individual = primary[primary.lambda_mm.eq(shown)].set_index("reference_label")
    lines.extend(
        [
            "",
            f"The historically displayed lambda {shown:g} is shown for continuity,",
            "not selected using this reference. Its individual results are:",
            "",
            "| Reviewed identity | Agreement on scored time | Scored / expected | "
            "Correct / expected | Longest observed disagreement |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for label in ["F", "M1", "M2"]:
        r = individual.loc[label]
        lines.append(
            f"| {label} | {r.id_agreement_on_scored_time:.2%} | "
            f"{r.scored_time_fraction:.2%} | {r.correct_id_time_fraction:.2%} | "
            f"{r.wrong_run_max_seconds:.2f} s |"
        )
    independent = sum(a["independent_exact_coordinate_checks"] for a in audit["arms"].values())
    differences = sum(
        f["different_elements"]
        for arm in audit["previous_run_array_comparison"].values()
        for f in arm.values()
    )
    elapsed = sum(s["seconds"] for s in state["stages"])
    lines.extend(
        [
            "",
            f"Stage wall time: {elapsed / 60:.1f} minutes. Independent scalar checks:",
            f"{independent:,}, with zero discrepancies. Raw-slot fresh-versus-prior comparison:",
            f"{differences:,} differing array elements across XYZ, source rows, slot labels,",
            "residuals, and view counts, across all seven conditions.",
            "",
            "This measures agreement with supplied reviewed IDs on shared geometry.",
            "It does not establish independent 3D accuracy or generalize beyond this trio.",
        ]
    )
    lines.extend(
        [
            "",
            "Numeric slot columns can be renamed between runs. Using the original",
            "fixed pre-window anchors to align each run once into F/M1/M2 order,",
            "the post-preparation comparison gives:",
            "",
            "| Lambda | Changed matched identity (fish-s) | Changed scoring status (rows) | "
            "Max aligned XYZ difference (mm) |",
            "| --- | --- | --- | --- |",
        ]
    )
    changed_arms = []
    for tag, item in audit["anchor_aligned_comparison"].items():
        seconds = item["changed_identity_fish_seconds_both_matched"]
        count = item["changed_score_status_observations"]
        maximum = item["post_preparation_fields"]["xyz"]["max_absolute_difference"]
        lines.append(f"| {tag} | {seconds:.6f} | {count:,} | {maximum:.8g} |")
        if count or item["changed_identity_observations_both_matched"]:
            changed_arms.append(tag)
    if changed_arms:
        lines.extend(
            [
                "",
                "Differences remain after fixed-anchor alignment for lambda "
                + ", ".join(changed_arms)
                + ". Their cause has not been isolated across",
                "the execution environments. They are not attributed to "
                "a tested method improvement.",
            ]
        )
    findings = "\n".join(lines)
    with provenance("p11_result_packaging", output) as rec:
        integrity = {}
        for name, item in inputs.items():
            if isinstance(item, dict) and "path" in item and "sha256" in item:
                actual = sha256(Path(item["path"]))
                assert actual == item["sha256"], f"Pinned input changed: {name}"
                integrity[name] = {"path": item["path"], "sha256": actual, "unchanged": True}
        write_json(output / "input_integrity_final.json", integrity)
        rec.input(
            "review_unchanged", Path(inputs["review"]["path"]), sha256=inputs["review"]["sha256"]
        )
        rec.input("validation", score / "validation.json", sha256=sha256(score / "validation.json"))
        readme = HERE / "README.md"
        text = readme.read_text(encoding="utf-8")
        start, stop = "<!-- RUN_RESULTS_START -->", "<!-- RUN_RESULTS_END -->"
        before, rest = text.split(start, 1)
        _, after = rest.split(stop, 1)
        readme.write_text(f"{before}{start}\n{findings}\n{stop}{after}", encoding="utf-8")
        source = output / "analysis_source"
        source.mkdir(exist_ok=True)
        for path in HERE.iterdir():
            if path.suffix in {".py", ".md", ".toml"}:
                shutil.copy2(path, source / path.name)
        for path in [
            REPO / "src/medaka3d/eval/review_identity.py",
            REPO / "tests/test_review_identity.py",
        ]:
            shutil.copy2(path, source / path.name)
        shutil.copy2(HERE / "DATA_DICTIONARY.md", output / "DATA_DICTIONARY.md")
        package = Path(importlib.import_module("sleap_nn.tracking").__file__).parent
        runtime = output / "runtime_tracker_source"
        for path in package.rglob("*.py"):
            target = runtime / path.relative_to(package)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        write_json(
            output / "runtime_environment.json",
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in [
                        "numpy",
                        "scipy",
                        "pandas",
                        "sleap-io",
                        "sleap-nn",
                        "torch",
                        "matplotlib",
                        "h5py",
                        "nbclient",
                        "nbformat",
                        "nbconvert",
                    ]
                },
                "notebook_executed_code_cells": len(code_cells),
                "runtime_source_snapshot_time": (
                    "At result packaging; this analysis did not modify dependencies"
                ),
            },
        )
        (output / "README.md").write_text(
            "# P-1-1 tracking rerun and identity evaluation\n\n"
            "Open [analysis.html](analysis.html) for the report and figures. "
            "The executed [notebook](analysis.ipynb) retains the calculations.\n\n"
            + findings
            + "\n\n## Output guide\n\n"
            "- `scoring_rerun_v1/`: fresh scores, mismatch intervals, automatic tracks.\n"
            "- `scoring_previous_v1/`: scores of the preserved October 28 outputs.\n"
            "- `pipeline/`: full rerun arrays and intermediate stages.\n"
            "- `figures/`: standalone PNG figures.\n"
            "- `DATA_DICTIONARY.md`: columns, units, denominators, and masks.\n"
            "- `analysis_source/`: analysis code, configuration, and detailed findings.\n"
            "- `source/`: initial pipeline snapshot and per-stage execution wrappers.\n"
            "- `runtime_tracker_source/`: installed SLEAP tracking source used by this run.\n"
            "- `logs/`, `run_status.json`, and provenance sidecars: execution evidence.\n\n"
            "The automatic F/M1/M2 names are unvalidated outside the fixed anchor. "
            "Retain gaps and unresolved identities for downstream work.\n\n"
            "## Recommended next comparison\n\n"
            "The observation-only `tracker_order_audit_2000frames/` found one "
            "returned-order mismatch in the top-camera prefix, causing two labels "
            "to be written to different input rows. This unchanged baseline retains "
            "the adapter defect; the prefix audit does not quantify its later impact. "
            "An extended top-camera check through frame 51,999 found only the same "
            "early case and no recurrence in its 2,206 post-preparation frames. "
            "This does not identify the defect as the cause of the ID error. "
            "Repair and rerun it under a new version before attributing the observed "
            "error to geometry alone. Then review crossing boundaries and compare "
            "appearance links on held-out encounters, with geometry fixed.\n",
            encoding="utf-8",
        )
        manifest = {}
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.name not in {
                "artifact_manifest.json",
                "p11_result_packaging.provenance.json",
            }:
                manifest[str(path.relative_to(output))] = {
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
        write_json(output / "artifact_manifest.json", manifest)
        rec.output(
            "manifest",
            output / "artifact_manifest.json",
            sha256=sha256(output / "artifact_manifest.json"),
        )
        rec.output("readme", output / "README.md", sha256=sha256(output / "README.md"))
    print(findings)


if __name__ == "__main__":
    main()
