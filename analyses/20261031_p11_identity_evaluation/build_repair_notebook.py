"""Execute a compact comparison notebook without changing the baseline report."""

import json
import os
import shutil
import sys
from textwrap import dedent

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from run_pipeline import HERE, REPO, settings, sha256

from medaka3d.provenance import provenance


def main():
    os.environ["MEDAKA3D_P11_CONFIG"] = str(HERE / "config_row_mapping_v1.toml")
    cfg, _, root, output = settings()
    score = output / cfg["evaluation"]["primary_scoring_directory"]
    if not (score / "validation.json").is_file():
        raise ValueError("Complete and independently audit the repair first")
    for variant in ("baseline", "repair"):
        if not (output / "join_ledger_v2" / variant / "summary.json").is_file():
            raise ValueError(f"Complete the v2 {variant} ledger first")
    target = HERE / "row_mapping_review.ipynb"
    if target.exists() or (output / "row_mapping_review.ipynb").exists():
        raise FileExistsError("The executed comparison notebook is preserved; use a new version")
    cells = []

    def md(value):
        cells.append(nbformat.v4.new_markdown_cell(dedent(value).strip()))

    def code(value):
        cells.append(nbformat.v4.new_code_cell(dedent(value).strip()))

    md("""
    # P-1-1: source-row repair and connection review

    Compare the frozen October 31 baseline with `id_row_mapping_20261102_r1`.
    Only adapter source-row assignment changes. All seven inherited lambda values
    remain visible; none is selected. Agreement is with reviewed labels on shared
    geometry, not independently certified 3D accuracy or generalization to other fish.
    """)
    code("""
    import json
    import sys
    import tomllib
    from pathlib import Path

    import matplotlib
    import matplotlib.pyplot as plt
    import pandas as pd
    from IPython.display import display

    import medaka3d
    from medaka3d import load_paths

    matplotlib.use("module://matplotlib_inline.backend_inline")
    repo = Path(medaka3d.__file__).resolve().parents[2]
    assert Path(sys.prefix).resolve() == (repo / ".venv").resolve()
    analysis = repo / "analyses/20261031_p11_identity_evaluation"
    config = tomllib.loads((analysis / "config_row_mapping_v1.toml").read_text())
    root = load_paths().raw_data_dir.parent / config["output"]["relative_to_data_parent"]
    repair = root / config["run_id"]
    baseline = root / config["comparison"]["baseline_run"]
    assert json.loads((repair / "run_status.json").read_text())["status"] == "complete"
    scoring_name = config["evaluation"]["primary_scoring_directory"]
    validation = json.loads((repair / scoring_name / "validation.json").read_text())
    assert validation["summary_reconciled"]
    assert all(v["discrepancies"] == 0 for v in validation["arms"].values())
    shown = config["evaluation"]["display_lambda"]
    figure_dir = repair / "repair_figures"
    figure_dir.mkdir(exist_ok=True)
    runs = {"Baseline": baseline, "Repair": repair}
    primary = {}
    for name, run in runs.items():
        table = pd.read_csv(run / scoring_name / "identity_summary.csv")
        primary[name] = table[table.match_distance_mm.eq(
            config["evaluation"]["max_match_distance_mm"]
        )].copy()
    print("Recording:", config["experiment"])
    print("Baseline:", baseline.name, "| Repair:", repair.name)
    print("Independent scalar checks:", sum(v["independent_exact_coordinate_checks"]
                                              for v in validation["arms"].values()))
    """)
    md("""
    ## Whole-window identity agreement

    The fixed pre-window anchor is never refitted after a crossing. Report the
    scored denominator with agreement, and preserve missing time in the complete
    expected fish-time denominator. Slot renaming is aligned once by those same anchors.
    """)
    code("""
    overview = []
    for name, table in primary.items():
        for lam, rows in table.groupby("lambda_mm"):
            correct, wrong = rows.correct_id_seconds.sum(), rows.wrong_id_seconds.sum()
            expected = rows.expected_seconds.sum()
            overview.append({
                "Run": name, "Lambda": lam,
                "Agreement on scored time (%)": 100 * correct / (correct + wrong),
                "Scored / expected (%)": 100 * (correct + wrong) / expected,
                "Correct / expected (%)": 100 * correct / expected,
            })
    overview = pd.DataFrame(overview)
    display(overview.round(4))
    comparison = pd.DataFrame({
        float(lam): {
            "Changed matched identity (fish-s)": row["changed_identity_fish_seconds_both_matched"],
            "Changed scoring status (rows)": row["changed_score_status_observations"],
            "Maximum aligned XYZ difference (mm)": row["post_preparation_fields"]["xyz"][
                "max_absolute_difference"
            ],
        } for lam, row in validation["anchor_aligned_comparison"].items()
    }).T.sort_index()
    comparison.index.name = "Lambda"
    display(comparison)
    display(pd.DataFrame(validation["c27_previous_comparison"]).T[
        ["different_labels", "unique_labels"]
    ])
    individual = pd.concat([
        rows[rows.lambda_mm.eq(shown)].assign(run=name)
        for name, rows in primary.items()
    ])
    display(individual[["run", "reference_label", "id_agreement_on_scored_time",
        "wrong_id_seconds", "scored_time_fraction", "correct_run_max_seconds"]].round(4))
    """)
    md("""
    ## Connection evidence and unassessed scope

    These are same-slot segment edges, not swap events. Both track and segment
    labels require 100 matched frames, 0.5 matched seconds, 95% matched duration,
    and no positive-duration minority label. A gap's endpoints may support a
    connection; the interior remains unassessed. The temporal candidate groups
    are unreviewed and do not establish independent encounters.
    Edge boundaries and assessable subsets differ across lambda settings, so
    these counts are not a matched-event ranking of the seven settings.
    """)
    code("""
    ledger_rows = []
    ledger_roots = {}
    for name, variant in [("Baseline", "baseline"), ("Repair", "repair")]:
        ledger_roots[name] = repair / "join_ledger_v2" / variant
        summaries = json.loads((ledger_roots[name] / "summary.json").read_text())
        for lam, row in summaries.items():
            assert row["c30_replay_matches_saved_permutation_and_decisions"]
            ledger_rows.append({
                "Run": name, "Lambda": float(lam), **row["edge_outcomes"],
                "All edges": row["edge_count"],
                "Assessed / all edges (%)": 100 * row["assessed_fraction_of_all_edges"],
                "Temporal review groups": row["candidate_group_count"],
            })
    ledger_table = pd.DataFrame(ledger_rows)
    display(ledger_table.round(3))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (name, _path) in zip(axes, ledger_roots.items(), strict=True):
        shown_rows = ledger_table[(ledger_table.Run == name) & ledger_table.Lambda.eq(shown)]
        counts = shown_rows[["correct", "wrong", "undecidable"]].iloc[0]
        bars = ax.bar(counts.index, counts, color=["#247E78", "#B4513C", "#858B91"])
        ax.bar_label(bars, padding=3)
        ax.set(title=f"{name}: lambda {shown:g}", ylabel="Candidate edges",
               ylim=(0, max(counts)*1.15))
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Connection evidence includes the unassessed denominator")
    fig.tight_layout()
    fig.savefig(figure_dir / "connection_evidence.png", dpi=150, bbox_inches="tight")
    display(fig)
    plt.close(fig)
    tracks = pd.read_csv(ledger_roots["Repair"] / f"lam{shown:g}" / "tracks.csv")
    display(tracks.groupby("label_status").agg(
        local_tracks=("local_track_id", "size"),
        observed_seconds=("observed_seconds", "sum"),
        matched_seconds=("matched_seconds", "sum"),
    ).round(3))
    support_time = []
    for name, ledger_root in ledger_roots.items():
        segments = pd.read_csv(ledger_root / f"lam{shown:g}" / "segments.csv")
        anchors = json.loads((runs[name] / scoring_name / "anchor_mappings.json").read_text())
        mapping = anchors[f"{shown:g}"]["reference_to_slot"]
        assert len(set(mapping)) == 3 and min(mapping) >= 0
        segments["correct_seconds"] = 0.0
        for label, slot in zip(("F", "M1", "M2"), mapping, strict=True):
            mask = segments.raw_slot.eq(slot)
            segments.loc[mask, "correct_seconds"] = segments.loc[mask, f"reference_{label}_seconds"]
        segments["wrong_seconds"] = segments.matched_seconds - segments.correct_seconds
        selected_primary = primary[name][primary[name].lambda_mm.eq(shown)]
        for derived, original in [("correct_seconds", "correct_id_seconds"),
                                  ("wrong_seconds", "wrong_id_seconds")]:
            assert abs(segments[derived].sum() - selected_primary[original].sum()) < 1e-6
        grouped = segments.groupby("track_label_status").agg(
            correct_fish_seconds=("correct_seconds", "sum"),
            wrong_fish_seconds=("wrong_seconds", "sum"),
        ).reset_index().assign(run=name)
        support_time.append(grouped)
    display(pd.concat(support_time, ignore_index=True).round(3))
    print("All three fish; support-category totals reconcile to the fixed-anchor scoring totals.")
    print("A pure local track can still carry the wrong trial-wide name.")
    """)
    md("""
    ## Review examples and C30 decisions

    Failed and successful connections both remain available for video review.
    The C30 replay matches the stored permutation and decision lists exactly.
    Temporal association with a final C34 edge is not causal attribution; the
    physical correctness of a velocity proposal has not been adjudicated.
    """)
    code("""
    selected = ledger_roots["Repair"] / f"lam{shown:g}"
    edges = pd.read_csv(selected / "edges.csv")
    display(edges[edges.outcome.eq("wrong")][[
        "edge_id", "raw_slot", "before_frame", "after_frame", "before_time_s", "after_time_s",
        "before_reference_label", "after_reference_label", "gap_interior_identity_status"
    ]])
    decisions = pd.read_csv(selected / "c30_decisions.csv")
    display(decisions.groupby(["kind", "decision"], dropna=False).agg(
        candidates=("c30_decision_id", "size"),
        velocity_calls=("velocity_called", "sum"),
        velocity_adoptions=("velocity_adopted", "sum"),
    ))
    changes = pd.read_csv(selected / "observed_assignment_changes.csv")
    display(changes.groupby("kind").size().rename("Observed correspondence changes"))
    wrong = pd.read_csv(selected / "wrong_identity_intervals.csv")
    top = wrong.nlargest(10, "duration_s")
    display(top[["reference_label", "start_s", "stop_s", "duration_s"]].round(3))
    print("Top ten disagreement rows, combined fish-seconds:", top.duration_s.sum())
    print("These overlapping per-fish intervals are not a count of independent crossings.")
    """)
    md("""
    ## Limits and next step

    Short, mixed, unsupported, and unobserved connections remain visible. A high
    fraction correct among a small assessed subset is not full-trial accuracy.
    Review the source videos before consolidating temporal groups into events
    or freezing a train/test split. Events sharing tracklet images must not leak
    across that split. Appearance and body length remain separate, unexecuted
    comparisons; BORIS alignment and reflection classification are unavailable.

    See `ROW_MAPPING_REPAIR.md`, the frozen configurations, stage provenance,
    `validation.json`, and the two v2 join-ledger directories for the evidence.
    """)
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {
        "display_name": "Medaka project Python",
        "language": "python",
        "name": "medaka-p11-repair",
    }
    nbformat.write(notebook, target)
    kernel_root = REPO / "work" / "p11_repair_kernel"
    kernel_dir = kernel_root / "kernels" / "medaka-p11-repair"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "Medaka project Python",
                "language": "python",
            }
        ),
        encoding="utf-8",
    )
    os.environ["JUPYTER_PATH"] = str(kernel_root)
    os.environ["JUPYTER_RUNTIME_DIR"] = str(kernel_root / "runtime")
    (kernel_root / "runtime").mkdir(exist_ok=True)
    with provenance("p11_repair_notebook", output) as rec:
        rec.input("notebook_source", target, sha256=sha256(target))
        rec.input("builder", __file__, sha256=sha256(HERE / "build_repair_notebook.py"))
        rec.input("validation", score / "validation.json", sha256=sha256(score / "validation.json"))
        baseline = (
            root / cfg["output"]["relative_to_data_parent"] / cfg["comparison"]["baseline_run"]
        )
        for variant, run in (("baseline", baseline), ("repair", output)):
            summary = run / cfg["evaluation"]["primary_scoring_directory"] / "identity_summary.csv"
            ledger = output / "join_ledger_v2" / variant / "summary.json"
            rec.input(f"{variant}_summary", summary, sha256=sha256(summary))
            rec.input(f"{variant}_ledger", ledger, sha256=sha256(ledger))
            anchor = run / cfg["evaluation"]["primary_scoring_directory"] / "anchor_mappings.json"
            rec.input(f"{variant}_anchors", anchor, sha256=sha256(anchor))
            for name in (
                "tracks",
                "segments",
                "edges",
                "c30_decisions",
                "observed_assignment_changes",
                "wrong_identity_intervals",
            ):
                table = (
                    ledger.parent / f"lam{cfg['evaluation']['display_lambda']:g}" / f"{name}.csv"
                )
                rec.input(f"{variant}_{name}", table, sha256=sha256(table))
        NotebookClient(
            notebook,
            timeout=600,
            kernel_name="medaka-p11-repair",
            resources={"metadata": {"path": str(HERE)}},
        ).execute()
        nbformat.write(notebook, target)
        shutil.copy2(target, output / target.name)
        html, _ = HTMLExporter(template_name="lab", exclude_input=True).from_notebook_node(notebook)
        (output / "row_mapping_review.html").write_text(html, encoding="utf-8")
        rec.output("notebook", output / target.name, sha256=sha256(output / target.name))
        rec.output("html", output / "row_mapping_review.html")
        rec.output("connection_evidence", output / "repair_figures" / "connection_evidence.png")
    print(output / "row_mapping_review.html", flush=True)


if __name__ == "__main__":
    main()
