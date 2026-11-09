"""Verify and archive the completed repair without changing the baseline package."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import nbformat
from run_pipeline import HERE, REPO, settings, sha256, write_json

from medaka3d.provenance import provenance


def main():
    os.environ["MEDAKA3D_P11_CONFIG"] = str(HERE / "config_row_mapping_v1.toml")
    cfg, _, root, output = settings()
    baseline = root / cfg["output"]["relative_to_data_parent"] / cfg["comparison"]["baseline_run"]
    validation = json.loads(
        (output / cfg["evaluation"]["primary_scoring_directory"] / "validation.json").read_text()
    )
    if json.loads((output / "run_status.json").read_text())["status"] != "complete":
        raise ValueError("Tracking run is incomplete")
    if not validation["summary_reconciled"] or any(
        arm["discrepancies"] for arm in validation["arms"].values()
    ):
        raise ValueError("Scoring has not passed the independent audit")
    for variant in ("baseline", "repair"):
        ledger = json.loads((output / "join_ledger_v2" / variant / "summary.json").read_text())
        if len(ledger) != len(cfg["pipeline"]["lambdas"]) or not all(
            arm["c30_replay_matches_saved_permutation_and_decisions"] for arm in ledger.values()
        ):
            raise ValueError(f"Incomplete C30/ledger verification: {variant}")
    notebook = nbformat.read(output / "row_mapping_review.ipynb", as_version=4)
    cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if not cells or [cell.execution_count for cell in cells] != list(range(1, len(cells) + 1)):
        raise ValueError("The notebook was not executed top-to-bottom")
    if any(item.output_type == "error" for cell in cells for item in cell.outputs):
        raise ValueError("The notebook contains an error")
    images = sum("image/png" in item.get("data", {}) for cell in cells for item in cell.outputs)
    if not images:
        raise ValueError("The saved notebook has no rendered figure")
    if sha256(HERE / "row_mapping_review.ipynb") != sha256(output / "row_mapping_review.ipynb"):
        raise ValueError("Local and archived notebooks differ")
    old_ledger, new_ledger = (output / "join_ledger_v2" / name for name in ("baseline", "repair"))
    ledger_tables = list(old_ledger.glob("lam*/*.csv"))
    if len(ledger_tables) != 8 * len(cfg["pipeline"]["lambdas"]):
        raise ValueError("The expected ledger table set is incomplete")
    for table in ledger_tables:
        if sha256(table) != sha256(new_ledger / table.relative_to(old_ledger)):
            raise ValueError(f"Baseline/repair ledger differs: {table}")
    for name in ("c27_row_comparison/summary.json", "video_review_v1/examples.json"):
        if not (output / name).is_file():
            raise ValueError(f"Missing review artifact: {name}")
    destination = output / "handoff_source"
    destination.mkdir(exist_ok=False)
    with provenance("p11_row_mapping_handoff", output) as rec:
        for filename, expected in cfg["comparison"]["baseline_hashes"].items():
            if sha256(baseline / filename) != expected:
                raise ValueError(f"Frozen baseline changed: {filename}")
            rec.input(f"baseline_{filename}", baseline / filename, sha256=expected)
        manifest = json.loads((baseline / "artifact_manifest.json").read_text())
        verified = 0
        for relative, record in manifest.items():
            normalized = relative.replace("\\", "/")
            if normalized.startswith(
                (
                    "scoring_rerun_v1/",
                    "pipeline/c27_uncapped_tracklets_any/",
                    "pipeline/c33_ray_rejection/",
                    "pipeline/c30_slot_stitch/",
                    "pipeline/c34_microflip/",
                )
            ):
                if sha256(baseline / normalized) != record["sha256"]:
                    raise ValueError(f"Consumed baseline artifact changed: {relative}")
                verified += 1
        inputs = json.loads((output / "inputs.json").read_text())
        for role in ("path", "backup"):
            source = Path(inputs["review"][role])
            if sha256(source) != inputs["review"]["sha256"]:
                raise ValueError(f"Original or archived human reference changed: {role}")
            rec.input(f"human_review_{role}", source, sha256=inputs["review"]["sha256"])
        sources = [path for path in HERE.iterdir() if path.suffix in (".py", ".toml", ".md")]
        sources += [
            REPO / "tools/c24_uncapped_tracklets.py",
            REPO / "src/medaka3d/eval/join_ledger.py",
            REPO / "tests/test_tracker_row_mapping.py",
            REPO / "tests/test_join_ledger.py",
            REPO / "tests/test_c30_trace.py",
        ]
        source_manifest = {}
        for source in sources:
            relative = source.relative_to(REPO)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            source_manifest[relative.as_posix()] = sha256(target)
        write_json(destination / "source_manifest.json", source_manifest)
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        status = subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True)
        (destination / "working_tree_status.txt").write_text(status, encoding="utf-8")
        checks = {
            "starting_head": head,
            "uncommitted_source_manifest": "handoff_source/source_manifest.json",
            "baseline_consumed_artifacts_verified": verified,
            "human_reference_and_curated_backup_unchanged": True,
            "notebook_executed_code_cells": len(cells),
            "notebook_embedded_pngs": images,
            "all_arms_scoring_reconciled": True,
            "all_arms_c30_replay_verified_both_runs": True,
            "identical_ledger_csv_tables": len(ledger_tables),
            "physical_identity_adjudication": "not_performed",
            "appearance_or_body_length_experiment": "not_performed",
        }
        write_json(output / "handoff_checks.json", checks)
        artifacts = {}
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.name not in (
                "artifact_manifest.json",
                "p11_row_mapping_handoff.provenance.json",
            ):
                artifacts[path.relative_to(output).as_posix()] = {
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
        write_json(output / "artifact_manifest.json", artifacts)
        rec.output(
            "artifact_manifest",
            output / "artifact_manifest.json",
            sha256=sha256(output / "artifact_manifest.json"),
        )
        rec.output("handoff_checks", output / "handoff_checks.json")
        rec.param("manifest_exclusions", "The manifest itself and this in-progress provenance file")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
