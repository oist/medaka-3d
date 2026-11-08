"""Execute the risk comparison notebook and export inspectable review queues."""

import json
import os
import shutil
import sys
import tomllib
from textwrap import dedent

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from run_pipeline import HERE, settings, sha256

from medaka3d.provenance import provenance


def main():
    policy = tomllib.loads((HERE / "config_geometry_risk_v3.toml").read_text())
    os.environ["MEDAKA3D_P11_CONFIG"] = str(HERE / policy["source_config"])
    _, _, _, source = settings()
    run = source.parent / policy["run_id"]
    validation = run / "audit_v1/validation.json"
    if not validation.exists():
        raise ValueError("Audit the completed comparison before building the report")
    output = run / "report_v1"
    output.mkdir(exist_ok=False)
    target = HERE / "geometry_motion_review.ipynb"
    if target.exists():
        raise FileExistsError("Preserve the executed notebook; use a new report version")
    cells = []

    def md(value):
        cells.append(nbformat.v4.new_markdown_cell(dedent(value).strip()))

    def code(value):
        cells.append(nbformat.v4.new_code_cell(dedent(value).strip()))

    md("""
    # P-1-1: geometry, motion, and localized identity warnings

    Existing manual labels evaluate all seven fixed geometry arms. No appearance
    classifier or new pose inference was used; no new annotation was created. Coverage starts
    at 300 s on the native acquisition clock. This is a retrospective comparison
    within one trio; warning thresholds are not independently validated elsewhere.

    The original row-repaired extraction stays unchanged. `r1` tested position
    and bidirectional bridges. `r2` stopped at an evaluation-mask invariant;
    `r3` corrects that comparison adapter and is the valid follow-up.
    """)
    code("""
    import json
    import sys
    import tomllib
    from pathlib import Path

    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from IPython.display import display

    import medaka3d
    from medaka3d import load_paths

    matplotlib.use("module://matplotlib_inline.backend_inline")
    repo = Path(medaka3d.__file__).resolve().parents[2]
    assert Path(sys.prefix).resolve() == (repo / ".venv").resolve()
    analysis = repo / "analyses/20261031_p11_identity_evaluation"
    policy = tomllib.loads((analysis / "config_geometry_risk_v3.toml").read_text())
    source_cfg = tomllib.loads((analysis / policy["source_config"]).read_text())
    root = load_paths().raw_data_dir.parent / source_cfg["output"]["relative_to_data_parent"]
    run = root / policy["run_id"]
    first = root / policy["first_experiment"]
    assert json.loads((run / "status.json").read_text())["status"] == "complete"
    validation = json.loads((run / "audit_v1/validation.json").read_text())
    warnings = pd.read_csv(run / "warning_summary.csv")
    scores = pd.read_csv(run / "correction_summary.csv")
    first_scores = pd.read_csv(first / "correction_summary.csv")
    core = warnings[warnings.rule.eq("identity_core")].copy()
    core["contained_percent"] = 100 * core.groups_contained / core.group_count
    counts = pd.read_csv(run / "audit_v1/c30_connection_summary.csv")
    figures = run / "report_v1/figures"
    figures.mkdir(exist_ok=True)
    shown = 0.767
    print("Input:", source_cfg["run_id"], "| Comparison:", run.name)
    print("Checksummed comparison artifacts:", validation["manifest_artifacts_verified"])
    """)
    md("""
    ## Does motion improve the automatic connections?

    The permissive C30 variant removes only the 2.5 mm median prerequisite for
    velocity decisions; the other v4 rules and C34 remain fixed. A lower total
    error in one arm does not compensate for new incorrect connections elsewhere.
    Fish-seconds count each affected fish separately. Adjacent and gap endpoint
    changes are reported separately and are not counts of physical swap events.
    """)
    code("""
    full = scores[scores.block.eq("all") & ~scores.method.eq("reference_endpoint_oracle")]
    comparison = full.pivot(index="lambda_mm", columns="method", values="wrong_seconds")
    display(comparison.round(3))
    display(counts.groupby("lambda_mm")[["corrected_edges", "newly_wrong_edges",
                                       "unchanged_wrong_edges"]].sum())
    display(first_scores[first_scores.block.eq("all") & first_scores.lambda_mm.eq(shown)]
            [["method", "wrong_seconds", "corrected_seconds", "newly_wrong_seconds"]].round(3))
    """)
    md("""
    ## How much review can geometry concentrate?

    The core warning is: fish separation <10 mm, an adjacent step >10 mm,
    a non-identity nearest assignment, or a missing position. Pad by 0.05 s.
    Ray-count churn and residual-only alerts remain observation-quality evidence,
    outside this identity review queue. No reference labels enter detection.

    A sustained reference transition has >=0.1 matched seconds and >=10 matches
    on each side; overlapping uncertainty brackets form review groups. Full
    containment is stricter than touching a warning. All observed changes,
    including short ones, remain in separate tables. Do not equate groups with
    independent crossings or guaranteed physical swaps.
    """)
    code("""
    display(core[["lambda_mm", "interval_count", "warning_seconds", "warning_fraction",
                  "groups_contained", "group_count", "raw_contained", "raw_changes",
                  "duration_p50_s", "duration_p95_s", "duration_p100_s"]].round(3))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    labels = [f"{v:g}" for v in core.lambda_mm]
    axes[0].bar(labels, 100 * core.warning_fraction, color="#d88526")
    axes[0].axhline(10, color="#555555", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Lambda", ylabel="Warning time / full time (%)",
                title="Identity warnings occupy about 5% of time")
    comparison.plot.bar(ax=axes[1], color=["#58768c", "#c25747"], rot=0)
    axes[1].set(xlabel="Lambda", ylabel="Wrong-ID fish-seconds",
                title="Relaxing C30 is not consistently beneficial")
    axes[1].legend(["Original C30", "Permissive velocity"], fontsize=8)
    fig.savefig(figures / "comparison.png", dpi=160)
    plt.show()
    """)
    md("""
    ## Where are the warning and lower-risk intervals?

    Times below are recording seconds; native frame columns are retained in the
    downloadable tables. Warning intervals are never cut merely to make clips
    shorter. The two prolonged close encounters retain long ambiguity windows.

    Unflagged intervals are lower-risk local continuity candidates. They can
    inherit an earlier physical-ID error: a quiet track is not a verified name.
    The lower-risk table includes reference coverage, observed label counts and
    reference purity per slot so this distinction can be inspected directly.
    """)
    code("""
    queue = pd.read_csv(run / "audit_v1/review_queue_lam0.767.csv")
    groups = pd.read_csv(run / "lam0.767/group_capture_identity_core.csv")
    clean = pd.read_csv(run / "lam0.767/lower_risk_intervals.csv")
    raw = pd.read_csv(run / "lam0.767/raw_capture_identity_core.csv")
    fig, axes = plt.subplots(2, 1, figsize=(11, 4.7), layout="constrained")
    for ax, limits in zip(axes, [(300, 2101), (500, 660)], strict=True):
        ax.broken_barh(list(zip(clean.start_s, clean.duration_s, strict=True)),
                       (0, 0.65), facecolors="#648b9a")
        ax.broken_barh(list(zip(queue.start_s, queue.duration_s, strict=True)),
                       (1, 0.65), facecolors="#d88526")
        ax.scatter(groups.start_s, np.full(len(groups), 2.3), marker="|",
                   color="#202020", s=100)
        ax.set(xlim=limits, ylim=(-0.15, 2.7), xlabel="Recording time (s)")
        ax.set_yticks([0.325, 1.325, 2.3], ["Lower-risk candidate", "Warning", "Reference change"])
        ax.grid(axis="x", alpha=0.2)
    axes[0].set_title("Lambda 0.767: warning locations and reference evidence")
    axes[1].set_title("Two long close encounters remain explicit")
    fig.savefig(figures / "timeline.png", dpi=160)
    plt.show()
    display(queue.nlargest(10, "duration_s")[["interval_id", "start_s", "end_s",
                "duration_s", "reference_group_count", "reasons"]].round(3))
    display(clean.nlargest(10, "duration_s")[["segment_id", "start_s", "end_s", "duration_s",
                "slot0_reference_purity", "slot1_reference_purity",
                "slot2_reference_purity"]].round(3))
    display(raw[~raw.warning_contains_bracket][["reference_index", "start_s", "end_s",
                                               "warning_overlap"]])
    print("Post-preparation audit:", validation["arms"]["0.767"])
    """)
    md("""
    ## Decision and limits

    Retain the original automatic connections. The tested motion/position changes
    do not establish a safe improvement across the seven arms. Carry the smaller
    warning queue forward for boundary checking, with the remaining misses and
    long close encounters visible. External recording validation is still needed
    before treating unflagged segments as generally reliable.

    The reference endpoint oracle is a diagnostic that uses manual labels; it
    is neither an automatic method nor a guaranteed upper bound when endpoints
    are missing. Additional source-video review should address uncertain boundary
    cases, not block use of the annotations already available. No appearance
    classifier was tried. No thresholds were selected by random frame splitting.
    """)
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {
        "display_name": "Medaka project Python",
        "language": "python",
        "name": "medaka-p11-risk",
    }
    nbformat.write(notebook, target)
    kernel_root = HERE / "_outputs/geometry_risk_kernel"
    kernel = kernel_root / "kernels/medaka-p11-risk"
    kernel.mkdir(parents=True, exist_ok=True)
    (kernel / "kernel.json").write_text(
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
    with provenance("p11_geometry_risk_notebook", output) as rec:
        rec.input("validation", validation, sha256=sha256(validation))
        rec.input("builder", __file__, sha256=sha256(HERE / "build_geometry_risk_report.py"))
        rec.input("notebook_source", target, sha256=sha256(target))
        NotebookClient(
            notebook,
            timeout=600,
            kernel_name="medaka-p11-risk",
            resources={"metadata": {"path": str(HERE)}},
        ).execute()
        nbformat.write(notebook, target)
        shutil.copy2(target, output / target.name)
        html, _ = HTMLExporter(template_name="lab", exclude_input=True).from_notebook_node(notebook)
        (output / "geometry_motion_review.html").write_text(html, encoding="utf-8")
        # Standalone searchable queue, retaining exact CSVs beside the report.
        import pandas as pd

        queue = pd.read_csv(run / "audit_v1/review_queue_lam0.767.csv")
        clean = pd.read_csv(run / "lam0.767/lower_risk_intervals.csv")
        for filename, table in (
            ("warning_intervals.csv", queue),
            ("lower_risk_intervals.csv", clean),
        ):
            table.to_csv(output / filename, index=False)
        warning_cols = [
            "interval_id",
            "start_s",
            "end_s",
            "duration_s",
            "start_frame",
            "stop_frame_exclusive",
            "reasons",
            "reference_group_count",
        ]
        clean_cols = [
            "segment_id",
            "start_s",
            "end_s",
            "duration_s",
            "start_frame",
            "stop_frame_exclusive",
            "slot0_reference_purity",
            "slot1_reference_purity",
            "slot2_reference_purity",
        ]
        page = """<!doctype html><html lang="en"><meta charset="utf-8">
        <title>P-1-1 identity warning queue</title><style>
        body{font:16px system-ui;margin:2rem;color:#21303c}h1{font-size:1.7rem}
        table{border-collapse:collapse;font-size:13px;width:100%}
        td,th{padding:7px;border-bottom:1px solid #ddd;text-align:left}
        th{position:sticky;top:0;background:#e9eef0}input{padding:10px;width:28rem;max-width:90%}
        .scroll{max-height:65vh;overflow:auto;border:1px solid #ccc}
        p{max-width:75rem;line-height:1.5}
        </style><h1>P-1-1: identity warning queue</h1>
        <p>Lambda 0.767, original geometry and automatic connections. No appearance classifier.
        Times are recording seconds; the first 300 seconds are excluded. Warnings use geometry only.
        Reference group counts are retrospective teaching labels,
        not detector inputs or physical swap counts.</p>
        <p><a href="geometry_motion_review.html">Executed analysis</a> ·
        <a href="warning_intervals.csv">Warning CSV</a> ·
        <a href="lower_risk_intervals.csv">Lower-risk CSV</a></p>
        <input id="filter" placeholder="Filter by time, frame, or reason"
        aria-label="Filter tables">
        <h2>Warning intervals</h2><p>The 0.05-second padding is included.
        Add the CSV's review context
        on both sides when checking endpoint identity. Long ambiguity is not forcibly shortened.</p>
        <div class="scroll">WARNING_TABLE</div>
        <h2>Lower-risk local intervals</h2><p>These may inherit an earlier wrong physical identity.
        Reference purity measures local consistency; it does not certify trial-wide naming.</p>
        <div class="scroll">CLEAN_TABLE</div>
        <script>document.getElementById('filter').addEventListener('input',e=>{
        const q=e.target.value.toLowerCase();document.querySelectorAll('tbody tr').forEach(r=>{
        r.style.display=r.textContent.toLowerCase().includes(q)?'':'none';});});</script></html>"""
        page = page.replace(
            "WARNING_TABLE", queue[warning_cols].round(3).to_html(index=False, border=0)
        )
        page = page.replace(
            "CLEAN_TABLE", clean[clean_cols].round(3).to_html(index=False, border=0)
        )
        (output / "review_queue.html").write_text(page, encoding="utf-8")
        local = HERE / "_outputs/geometry_risk_figures"
        local.mkdir(parents=True, exist_ok=True)
        for path in (output / "figures").glob("*.png"):
            shutil.copy2(path, local / path.name)
        shutil.copy2(
            HERE / "build_geometry_risk_report.py", output / "build_geometry_risk_report.py"
        )
        rec.output("notebook", output / target.name, sha256=sha256(output / target.name))
        rec.output(
            "review_queue",
            output / "review_queue.html",
            sha256=sha256(output / "review_queue.html"),
        )
    print(output / "review_queue.html", flush=True)


if __name__ == "__main__":
    main()
