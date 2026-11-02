"""Build, execute, and export the repository-required P-1-1 analysis notebook."""

import json
import os
import shutil
import subprocess
import sys
from textwrap import dedent

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from run_pipeline import HERE, REPO, settings, sha256

from medaka3d.provenance import provenance


def main():
    cfg, _, _, output = settings()
    score = output / cfg["evaluation"]["primary_scoring_directory"]
    if not (score / "validation.json").is_file():
        raise RuntimeError("Complete rerun scoring and independent audit before notebook execution")
    cells = []

    def md(text):
        cells.append(nbformat.v4.new_markdown_cell(dedent(text).strip()))

    def code(text):
        cells.append(nbformat.v4.new_code_cell(dedent(text).strip()))

    md("""
    # P-1-1: geometry tracking versus reviewed identity

    One recording, three fish, seven inherited lambda settings. These results measure
    agreement with the supplied human-reviewed F/M1/M2 labels. They do not independently
    validate 3D positions or establish an error rate for other fish groups.
    """)
    code(r"""
    import json
    import sys
    import tomllib
    from pathlib import Path

    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from IPython.display import Markdown, display

    import medaka3d
    from medaka3d import load_paths

    matplotlib.use("module://matplotlib_inline.backend_inline")
    repo = Path(medaka3d.__file__).resolve().parents[2]
    assert Path(sys.prefix).resolve() == (repo / ".venv").resolve(), (
        "Use the project virtual environment"
    )
    analysis = repo / "analyses/20261031_p11_identity_evaluation"
    config = tomllib.loads((analysis / "config.toml").read_text(encoding="utf-8-sig"))
    paths = load_paths()
    run = (
        paths.raw_data_dir.parent
        / config["output"]["relative_to_data_parent"]
        / config["run_id"]
    )
    score = run / config["evaluation"]["primary_scoring_directory"]
    assert json.loads((run / "run_status.json").read_text())["status"] == "complete"
    summary = pd.read_csv(score / "identity_summary.csv")
    inventory = json.loads((score / "reference_inventory.json").read_text())
    arm_results = json.loads((score / "arm_results.json").read_text())
    validation = json.loads((score / "validation.json").read_text())
    anchors = json.loads((score / "anchor_mappings.json").read_text())
    primary = summary[
        summary.match_distance_mm.eq(config["evaluation"]["max_match_distance_mm"])
    ].copy()
    overview = primary.groupby("lambda_mm").agg(
        correct_seconds=("correct_id_seconds", "sum"),
        wrong_seconds=("wrong_id_seconds", "sum"),
        expected_seconds=("expected_seconds", "sum"),
    )
    overview["scored_seconds"] = overview.correct_seconds + overview.wrong_seconds
    overview["agreement_percent"] = 100 * overview.correct_seconds / overview.scored_seconds
    overview["correct_time_percent"] = (
        100 * overview.correct_seconds / overview.expected_seconds
    )
    overview["scored_time_percent"] = (
        100 * overview.scored_seconds / overview.expected_seconds
    )
    overview["joint_correct_percent"] = [
        100 * arm_results[f"{x:g}"]["joint_correct_fraction"] for x in overview.index
    ]
    shown = config["evaluation"]["display_lambda"]
    shown_rows = (
        primary[primary.lambda_mm.eq(shown)]
        .set_index("reference_label")
        .loc[["F", "M1", "M2"]]
    )
    figures = run / "figures"
    figures.mkdir(exist_ok=True)
    colors = {"correct": "#2878A3", "wrong": "#C56428", "unscored": "#C3C6CA"}
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "savefig.dpi": 150,
        }
    )


    def save_figure(fig, name):
        fig.savefig(figures / name, bbox_inches="tight", facecolor="white")
        plt.show()
        plt.close(fig)


    print(
        f"Recording: {config['experiment']}; execution: {config['run_id']}; "
        f"Python {sys.version.split()[0]}"
    )
    """)
    code(r"""
    fish_lines = "; ".join(
        f"{label}: {row.id_agreement_on_scored_time:.2%}"
        for label, row in shown_rows.iterrows()
    )
    display(
        Markdown(
            "**Fresh rerun completed for all seven conditions.** "
            "The historically displayed condition "
            f"lambda={shown:g} agrees with reviewed IDs over scored time as follows: {fish_lines}. "
            "All three fish agree simultaneously over "
            f"{overview.loc[shown, 'joint_correct_percent']:.2f}% "
            f"of the full post-preparation duration. Lambda={shown:g} is shown "
            "for continuity with earlier "
            "reports, not selected as a validated optimum. All conditions are shown below."
        )
    )
    """)
    md("""
    ## Context and methods

    - Existing S14 detections were verified against all supplied per-camera SLEAP instances.
      C27, C28, C33, C30 v4, and C34 were rerun. Each camera and each Viterbi sequence is
      serial in time. Independent lambda arms run in separate processes.
    - Use native acquisition timestamps and exclude the first 300 seconds. A fixed identity
      anchor at [298, 300) seconds maps slots to reference labels once.
      Never remap after a crossing.
    - Match pectoral positions without using identity labels: mutual nearest positions within
      5 mm and a 2 mm margin against competing positions. Repeat at 2 and 10 mm for sensitivity.
    - Exclude unviewed, uncertain, bad3d, nodes_bad, interpolated, and nonfinite reference positions
      from correctness scoring. Retain their time in the full coverage denominator.
    - `watched` means displayed at any speed. `uncertain` was unused. Neither flag proves that
      every short identity exchange was resolved. The reference shares underlying geometry
      with the pipeline, so small coordinate error is not independent spatial validation.
    - Report observational agreement for one trio. No framewise independence assumption,
      biological replication claim, or population-level confidence interval is made.
    """)
    code(r"""
    display(
        pd.DataFrame(
            [
                {
                    "First scored frame": inventory["first_scored_frame"],
                    "First scored time (s)": inventory["first_scored_time_s"],
                    "Last acquisition (s)": inventory["recording_end_s"],
                    "Expected fish-time (s)": inventory["expected_fish_seconds"],
                    "Reference eligible fish-time (s)": inventory[
                        "reference_eligible_fish_seconds"
                    ],
                }
            ]
        ).round(6)
    )
    display(
        Markdown(
            "Elapsed camera clocks differ by at most 3.51 ms at the same native index. "
            "The existing camera index alignment is retained; absolute hardware synchronization "
            "has not been independently re-established. Durations use left-held intervals ending "
            "at the next acquisition, with no extrapolation beyond the last timestamp."
        )
    )
    """)
    md("## Results across all conditions")
    code(r"""
    display(
        overview[
            [
                "agreement_percent",
                "scored_time_percent",
                "correct_time_percent",
                "joint_correct_percent",
            ]
        ]
        .rename(
            columns={
                "agreement_percent": "Agreement on scored time (%)",
                "scored_time_percent": "Scored time / expected (%)",
                "correct_time_percent": "Correct ID time / expected (%)",
                "joint_correct_percent": "All three correct / recording time (%)",
            }
        )
        .round(2)
    )
    fig, ax = plt.subplots(figsize=(10, 4.8))
    correct = 100 * overview.correct_seconds / overview.expected_seconds
    wrong = 100 * overview.wrong_seconds / overview.expected_seconds
    unscored = 100 - correct - wrong
    labels = [f"{x:g}" for x in overview.index]
    ax.bar(labels, correct, color=colors["correct"], label="Agrees with reviewed ID")
    ax.bar(
        labels,
        wrong,
        bottom=correct,
        color=colors["wrong"],
        label="Disagrees with reviewed ID",
        hatch="//",
    )
    ax.bar(
        labels,
        unscored,
        bottom=correct + wrong,
        color=colors["unscored"],
        label="Unscored / unavailable",
    )
    ax.set(
        xlabel="Inherited lambda (mm); none selected",
        ylabel="Share of expected fish-time (%)",
        ylim=(0, 100),
        title="Identity agreement and unavailable time across all seven conditions",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, fontsize=9)
    save_figure(fig, "all_lambda_identity_coverage.png")
    """)
    md("""
    ## Individual identities at the historical display condition

    Coverage uses the complete post-preparation duration for each fish. Agreement uses
    only time with an assessable reference, an unambiguous spatial correspondence, and
    a supported fixed anchor mapping. Both denominators are necessary.
    """)
    code(r"""
    display(
        (
            shown_rows[
                [
                    "id_agreement_on_scored_time",
                    "scored_time_fraction",
                    "correct_id_time_fraction",
                ]
            ]
            * 100
        )
        .rename(
            columns={
                "id_agreement_on_scored_time": "Agreement (%)",
                "scored_time_fraction": "Scored time (%)",
                "correct_id_time_fraction": "Correct time / expected (%)",
            }
        )
        .round(2)
    )
    fig, ax = plt.subplots(figsize=(9, 3.6))
    left = np.zeros(3)
    parts = [
        ("correct", shown_rows.correct_id_seconds, "Agrees"),
        ("wrong", shown_rows.wrong_id_seconds, "Disagrees"),
        (
            "unscored",
            shown_rows.expected_seconds
            - shown_rows.correct_id_seconds
            - shown_rows.wrong_id_seconds,
            "Unscored",
        ),
    ]
    for key, seconds, title in parts:
        values = 100 * seconds.to_numpy() / shown_rows.expected_seconds.to_numpy()
        ax.barh(
            shown_rows.index,
            values,
            left=left,
            color=colors[key],
            label=title,
            hatch="//" if key == "wrong" else None,
        )
        for i, value in enumerate(values):
            if value > 5:
                ax.text(
                    left[i] + value / 2,
                    i,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    color="white",
                )
        left += values
    ax.set(
        xlim=(0, 100),
        xlabel="Share of each fish's expected time (%)",
        title=f"Reviewed identity agreement, lambda={shown:g}",
    )
    ax.invert_yaxis()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=3)
    save_figure(fig, "individual_identity_coverage.png")
    """)
    md("## When identity disagreement persists")
    code(r"""
    timeline = pd.read_csv(score / "per_second_agreement.csv")
    timeline = timeline[timeline.lambda_mm.eq(shown)]
    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True, sharey=True)
    for ax, label in zip(axes, ["F", "M1", "M2"], strict=True):
        rows = timeline[timeline.reference_label.eq(label)]
        x = rows.recording_second.to_numpy() / 60
        total = rows.observed_bin_seconds.to_numpy()
        wrong = (
            np.divide(
                rows.wrong_id_seconds, total, out=np.zeros_like(total), where=total > 0
            )
            * 100
        )
        unscored = (
            np.divide(
                total - rows.correct_id_seconds - rows.wrong_id_seconds,
                total,
                out=np.zeros_like(total),
                where=total > 0,
            )
            * 100
        )
        ax.fill_between(
            x, 0, wrong, color=colors["wrong"], step="post", label="Disagreement"
        )
        ax.plot(
            x, unscored, color="#62676E", linewidth=0.8, label="Unscored", linestyle="--"
        )
        ax.set(ylabel=f"{label}\n% of second", ylim=(0, 100))
    axes[0].set_title(f"Disagreement within each one-second bin, lambda={shown:g}")
    axes[0].legend(loc="upper right", ncol=2)
    axes[-1].set(
        xlabel="Recording time (minutes)", xlim=(5, inventory["recording_end_s"] / 60)
    )
    axes[-1].set_xticks([5, 10, 15, 20, 25, 30, 35])
    fig.tight_layout()
    save_figure(fig, "identity_disagreement_timeline.png")
    intervals = pd.read_csv(score / "wrong_identity_intervals.csv")
    display(
        intervals[intervals.lambda_mm.eq(shown)]
        .nlargest(10, "duration_s")[
            [
                "reference_label",
                "start_frame",
                "stop_frame_exclusive",
                "start_s",
                "stop_s",
                "duration_s",
            ]
        ]
        .round(3)
    )
    display(
        Markdown(
            "These intervals break at missing or ambiguous observations. "
            "They are observed disagreement runs, not a count of independently "
            "adjudicated crossings or a guarantee about the gap interiors."
        )
    )
    """)
    md("## Which identities are confused")
    code(r"""
    with np.load(score / f"scoring_arrays_lam{shown:g}.npz") as z:
        slots, weights, eligible, mapping = (
            z["matched_slot"],
            z["duration_s"],
            z["eligible"],
            z["anchor_reference_to_slot"],
        )
    confusion = np.array(
        [
            [weights[eligible & (slots[:, i] == mapping[j])].sum() for j in range(3)]
            for i in range(3)
        ]
    )
    percent = confusion / confusion.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(5.6, 4.5))
    image = ax.imshow(percent, vmin=0, vmax=100, cmap="Blues")
    ax.set(
        xticks=range(3),
        yticks=range(3),
        xticklabels=["F", "M1", "M2"],
        yticklabels=["F", "M1", "M2"],
        xlabel="Predicted identity from fixed anchor",
        ylabel="Reviewed identity",
        title=f"Identity confusion on scored time, lambda={shown:g}",
    )
    for i in range(3):
        for j in range(3):
            ax.text(
                j,
                i,
                f"{percent[i, j]:.1f}%",
                ha="center",
                va="center",
                color="white" if percent[i, j] > 60 else "#171717",
            )
    fig.colorbar(image, ax=ax, label="Share of scored time for reviewed identity (%)")
    save_figure(fig, "identity_confusion.png")
    """)
    md("## Matching sensitivity, fragmentation, and proximity")
    code(r"""
    sensitivity = summary[summary.lambda_mm.eq(shown)].copy()
    sensitivity["Agreement (%)"] = sensitivity.id_agreement_on_scored_time * 100
    sensitivity["Scored time (%)"] = sensitivity.scored_time_fraction * 100
    display(
        sensitivity[
            ["match_distance_mm", "reference_label", "Agreement (%)", "Scored time (%)"]
        ].round(3)
    )
    gated = pd.read_csv(score / "gated_local_track_purity.csv")
    gated_view = gated[gated.lambda_mm.eq(shown)]
    purity = gated_view.majority_seconds.sum() / gated_view.matched_seconds.sum()
    display(
        Markdown(
            f"The gated output contains **{len(gated_view)}** local tracks "
            "with assessable post-preparation matches. "
            f"Their retrospective, time-weighted majority-label purity is **{purity:.2%}**. "
            "Many local tracks can map to the same fish in this diagnostic. "
            "This does not establish "
            "consistent identity between fragments and is not trial-wide identity coverage."
        )
    )
    proximity = pd.read_csv(score / "proximity_context.csv")
    display(proximity[proximity.lambda_mm.eq(shown)].round(4))
    display(
        Markdown(
            "The 20 mm condition describes geometric proximity, not courtship. "
            "An identity exchange can start during contact and persist after "
            "the fish separate, so time spent with wrong identity is not "
            "the same measure as the risk of an exchange starting in each context."
        )
    )
    """)
    md("## Validation and interpretation limits")
    code(r"""
    independent = sum(
        a["independent_exact_coordinate_checks"] for a in validation["arms"].values()
    )
    different = sum(
        v["different_elements"]
        for a in validation["previous_run_array_comparison"].values()
        for v in a.values()
    )
    display(
        Markdown(
            "Scoring totals reconcile to all expected frames and durations. "
            "A separate scalar audit checked "
            f"**{independent:,}** exact-coordinate correspondences with no scoring discrepancies. "
            f"The raw-slot array comparison found **{different:,}** differing elements "
            "across xyz, source rows, "
            "slot labels, residuals, and view-count arrays (all seven conditions). "
            "This numerical audit does not independently certify "
            "the supplied human identity labels."
        )
    )
    display(
        pd.DataFrame(
            [
                {
                    "lambda_mm": float(lam),
                    **{k: v["different_elements"] for k, v in fields.items()},
                }
                for lam, fields in validation["previous_run_array_comparison"].items()
            ]
        )
    )
    display(
        Markdown(
            "Numeric slot names are arbitrary. The following comparison aligns each run "
            "once using its original [298, 300) s anchor, in F/M1/M2 order, "
            "and compares the post-preparation window. "
            "No mapping is refitted after a crossing. This separates slot renaming "
            "from changes in reviewed identity agreement."
        )
    )
    aligned_rows = []
    for lam, result in validation["anchor_aligned_comparison"].items():
        aligned_rows.append({
            "Lambda": float(lam),
            "Old slots (F/M1/M2)": str(result["previous_reference_to_slot"]),
            "New slots (F/M1/M2)": str(result["rerun_reference_to_slot"]),
            "Changed matched ID (fish-s)": result[
                "changed_identity_fish_seconds_both_matched"
            ],
            "Changed scoring status (rows)": result["changed_score_status_observations"],
            "Max aligned XYZ difference (mm)": result["post_preparation_fields"]["xyz"][
                "max_absolute_difference"
            ],
        })
    display(pd.DataFrame(aligned_rows).round(6))
    display(Markdown(
        "The preserved run used revision 50d56d0 and Python 3.13.17; this run uses "
        "revision c601b7b and Python 3.13.11. Tracked pipeline code is unchanged "
        "between these revisions. Any remaining run differences have not been "
        "causally isolated and are not evidence of a tested method improvement."
    ))
    display(
        Markdown(
            "Use the mismatch intervals and original video to review "
            "consequential male-male joins. If positions remain usable while "
            "identity correspondence fails, compare appearance-based crossing links "
            "on held-out events with geometry fixed. Keep unsupported links unresolved. "
            "No appearance model was trained or evaluated in this run."
        )
    )
    """)
    md("""
    ## Implementation concern to isolate before changing the method

    A separate observation-only audit ran the unchanged C24 tracker on the first
    2,000 frames per camera. In the top view at frame 20, the library returned
    input rows in order `[0, 1, 2, 4, 3, 5]`. The repository adapter writes labels
    by returned-list offset, so two labels were written to the wrong detection
    rows. No return-order mismatch was observed in the two side-view prefixes.

    This is an adapter row-correspondence defect, not evidence that appearance
    features are necessary. The short prefix is outside the scored window and
    does not quantify its contribution to later ID disagreement. This rerun
    deliberately preserves the existing implementation to measure the current
    baseline. A separately versioned repair and comparison should precede any
    conclusion that the remaining error is an inherent geometry limitation.
    Audit code, exact frames, and evidence are in `tracker_order_audit_2000frames`.

    An extended top-camera audit ran continuously from frame zero through 51,999.
    It found only the same frame-20 case and no return-order mismatch in its
    2,206 post-preparation frames. This is not a full-trial audit and does not
    establish this defect as the cause of the measured ID disagreements.
    Extended evidence is in `tracker_order_audit_52000frames_top_cam`.

    ## Files and safe use

    `automatic_pectoral_3d_lam*.csv.gz` contains the uncorrected automatic positions,
    original slot labels, local gated track IDs, and fixed-anchor predicted names.
    **These predicted names are not certified trial-wide identities.** The
    `identity_support` column explicitly marks them `automatic_unvalidated`.
    Raw stage arrays retain source-camera row mappings. Scoring arrays and tables
    record agreement, disagreement, and unassessed time separately.

    The source review, prior outputs, and existing repository edits were preserved.
    Input hashes, annotation archive location, timing audit, code snapshots,
    stage logs, and provenance are stored beside the generated results.
    """)
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {
        "display_name": "Medaka project Python",
        "language": "python",
        "name": "medaka-p11",
    }
    target = HERE / "analysis.ipynb"
    nbformat.write(notebook, target)
    subprocess.run([sys.executable, "-m", "ruff", "format", str(target)], check=True)
    subprocess.run([sys.executable, "-m", "ruff", "check", str(target)], check=True)
    notebook = nbformat.read(target, as_version=4)
    kernel_root = REPO / "work" / "p11_notebook_kernel"
    kernel_dir = kernel_root / "kernels" / "medaka-p11"
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
    with provenance("p11_notebook_execution", output) as rec:
        rec.input("notebook_source", target, sha256=sha256(target))
        rec.input(
            "score_summary",
            score / "identity_summary.csv",
            sha256=sha256(score / "identity_summary.csv"),
        )
        client = NotebookClient(
            notebook,
            timeout=600,
            kernel_name="medaka-p11",
            resources={"metadata": {"path": str(HERE)}},
        )
        client.execute()
        nbformat.write(notebook, target)
        shutil.copy2(target, output / "analysis.ipynb")
        html, _ = HTMLExporter(
            template_name="lab", exclude_input=True, exclude_input_prompt=True
        ).from_notebook_node(notebook)
        (output / "analysis.html").write_text(html, encoding="utf-8")
        rec.output(
            "executed_notebook", output / "analysis.ipynb", sha256=sha256(output / "analysis.ipynb")
        )
        rec.output("html", output / "analysis.html", sha256=sha256(output / "analysis.html"))
        for figure in sorted((output / "figures").glob("*.png")):
            rec.output(figure.stem, figure, sha256=sha256(figure))
    print(output / "analysis.html")


if __name__ == "__main__":
    main()
