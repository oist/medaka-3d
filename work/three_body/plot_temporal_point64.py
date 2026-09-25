"""Plot a completed fixed64 comparison; flags and image consistency are not accuracy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WORK = Path(__file__).resolve().parent
VIEWS = ("top_cam", "side1_cam", "side2_cam")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run(source, expected_sha, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_relative_to(WORK) or not output.is_relative_to(WORK) or output.exists():
        raise ValueError("existing workspace source and new workspace output required")
    seal = source / "comparison.provenance.json"
    if sha(seal) != expected_sha:
        raise ValueError("comparison seal SHA changed")
    record = json.loads(seal.read_bytes())
    if record["status"] != "ok":
        raise ValueError("all three views must have completed successfully")
    bound = {Path(row["path"]).resolve(): row for row in record["outputs"].values()}
    evidence = {str(seal): expected_sha, str(Path(__file__).resolve()): sha(__file__)}

    def load(name):
        path = (source / name).resolve()
        item = bound[path]
        if not path.is_relative_to(source) or sha(path) != item["sha256"]:
            raise ValueError("bound plot input changed")
        if path.stat().st_size != item["size_bytes"]:
            raise ValueError("bound plot input size changed")
        evidence[str(path)] = item["sha256"]
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k].copy() for k in ("reported", "initial_mad")}

    output.mkdir()
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
    colors = {
        "lk": "#167d9a",
        "ct_first": "#d87620",
        "ct_final": "#8061a8",
        "stationary": "#777777",
    }
    labels = {
        "lk": "LK accepted",
        "ct_first": "CT first reported visible",
        "ct_final": "CT final reported visible",
    }
    rows = []
    x = np.arange(1, 64) / 166.0
    for index, view in enumerate(VIEWS):
        data = {name: load(f"{view}/{name}/metrics.npz") for name in colors}
        n = data["lk"]["reported"].shape[1]
        if any(
            d["reported"].shape != (64, n) or d["initial_mad"].shape != (64, n)
            for d in data.values()
        ):
            raise ValueError("fixed64 point domain differs")
        left, right = axes[index]
        fractions = {}
        for name in labels:
            flags = data[name]["reported"][1:]
            fraction = flags.sum(axis=1) / n if n else np.full(63, np.nan)
            fractions[name] = fraction.tolist()
            left.plot(
                x,
                fraction,
                color=colors[name],
                label=labels[name],
                linestyle="--" if name == "ct_final" else "-",
            )
        common = data["lk"]["reported"] & data["ct_first"]["reported"]
        for name in ("lk", "ct_first", "stationary"):
            common &= np.isfinite(data[name]["initial_mad"])
        medians = {}
        for name in ("lk", "ct_first", "stationary"):
            values = np.asarray(
                [
                    np.median(data[name]["initial_mad"][t, common[t]])
                    if common[t].any()
                    else np.nan
                    for t in range(1, 64)
                ]
            )
            medians[name] = values.tolist()
            right.plot(
                x,
                values,
                color=colors[name],
                label=name.replace("_", " "),
                linestyle=":" if name == "stationary" else "-",
            )
        count = common[1:].sum(axis=1)
        left.set(
            title=f"{view} | {n} initial queries",
            ylim=(-0.03, 1.03),
            ylabel="Fraction with method-specific flag",
        )
        right.set(
            title=f"Same query/frame subset | n={count.min()}..{count.max()}",
            ylabel="Median initial patch MAD (gray / 255)",
        )
        for ax in (left, right):
            ax.grid(alpha=0.2)
            ax.set_xlabel("Seconds after initial frame")
        if index == 0:
            left.legend(fontsize=8, loc="lower left")
            right.legend(fontsize=8)
        rows.append(
            dict(
                view=view,
                query_count=n,
                relative_frames=list(range(1, 64)),
                time_seconds=x.tolist(),
                flag_fractions=fractions,
                exact_common_query_counts=count.tolist(),
                common_patch_medians=medians,
            )
        )
    fig.suptitle(
        "Fixed64 point diagnostics: availability and image consistency\n"
        "Different visibility rules; no physical-point or animal-ID ground truth",
        fontsize=13,
    )
    fig.savefig(output / "diagnostics.png", dpi=160)
    plt.close(fig)
    result = dict(
        source=str(source),
        comparison_seal_sha256=expected_sha,
        initial_frame_excluded=True,
        rows=rows,
        image_reference="Same stored initial query patch for every method",
        conditional_subset="LK accepted and CT first reported-visible; all three MAD values finite",
        identity_accuracy=False,
        point_accuracy=False,
    )
    (output / "plot_data.json").write_text(
        json.dumps(clean(result), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    for path, digest in evidence.items():
        if sha(path) != digest:
            raise ValueError("plot input changed during rendering")
    outputs = {
        p.name: dict(path=str(p), sha256=sha(p), size_bytes=p.stat().st_size)
        for p in sorted(output.iterdir())
        if p.is_file()
    }
    (output / "plot.provenance.json").write_text(
        json.dumps(
            dict(
                stage="temporal_point64_plot_v1",
                status="ok",
                inputs=evidence,
                outputs=outputs,
                matplotlib_version=matplotlib.__version__,
                numpy_version=np.__version__,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return dict(status="ok", plot=str(output / "diagnostics.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--seal-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.seal_sha256, args.output)))
