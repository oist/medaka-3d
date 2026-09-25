"""Development-only descriptive histogram diagnosis; never reads held-out runs."""

from __future__ import annotations

import itertools
import json
import os
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "labs/20260908_generative_tracking"))
from benchmark import BASELINE_DIR, EXPERIMENT, sha256

from medaka3d.provenance import provenance

RUN = REPO / "work/generative_tracking/recovery_v1_nominal_dev_transient"
OUT = REPO / "work/generative_tracking/appearance_diagnostic/completed"
BASELINE = REPO / "work/generative_tracking_inputs" / BASELINE_DIR / (
    f"C34-label__{EXPERIMENT}__lam0.767.npz")
VIEWS = ("top_cam", "side1_cam", "side2_cam")
COLORS = ("#e6a72b", "#11a9c2", "#d65da7")
EVENT = 333547  # Parent's previously observed development event; not fitted.


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    with provenance("development_appearance_diagnostic", OUT, name="diagnostic") as record:
        candidate = RUN / "candidate.npz"
        record.input("development_candidate", candidate, sha256=sha256(candidate))
        record.input("baseline_diagnostic_only", BASELINE, sha256=sha256(BASELINE))
        record.input("script", Path(__file__), sha256=sha256(Path(__file__)))
        shutil.copyfile(__file__, OUT / "diagnose_appearance.py")
        record.param("information_barrier", "Only fixed v1 development transient and its baseline; "
                     "no validation/minute outputs or GT labels read. No tracker changes.")
        record.param("template_rule", "Fixed first 20 frames, view-specific quality-weighted mean; "
                     "quality > 0 only, no fitted threshold, no baseline labels in templates.")
        with np.load(candidate, allow_pickle=False) as arrays:
            frames = arrays["frame_idx"]
            hist = arrays["appearance"].astype(float)
            quality = arrays["appearance_quality"].astype(float)
            source = arrays["node_source"][:, :, :, 0]
        np.testing.assert_array_equal(frames, np.arange(333519, 333719))
        valid = (quality > 0) & (hist.sum(-1) > 0)
        quality = np.where(valid, quality, 0.)
        sums = quality[:20].sum(0)
        if (sums <= 0).any():
            raise ValueError("each prefix slot/view needs positive-quality evidence")
        templates = (hist[:20] * quality[:20, ..., None]).sum(0) / sums[..., None]
        templates /= templates.sum(-1, keepdims=True)
        # D[t, raw_slot, view, prefix_template_slot]. Frozen appearance cost is D squared.
        distances = np.sqrt(0.5 * ((np.sqrt(hist[:, :, :, None, :])
                                   - np.sqrt(templates.transpose(1, 0, 2)[None, None])) ** 2).sum(-1))
        distances = np.where(valid[..., None], distances, np.nan)
        template_distance = np.sqrt(0.5 * ((np.sqrt(templates[:, None])
                                           - np.sqrt(templates[None])) ** 2).sum(-1))
        winner = np.argmin(np.nan_to_num(distances, nan=np.inf), axis=-1)
        winner[~valid] = -1
        permutations = np.asarray(list(itertools.permutations(range(3))))
        weighted_cost = np.nansum(distances**2 * quality[..., None], axis=2)
        total_quality = quality.sum(2)
        all_views_valid = valid.all(axis=2)
        view_agreement = all_views_valid & (winner.min(2) == winner.max(2))

        # Baseline is opened only after all templates/distances are fixed.
        with np.load(BASELINE, allow_pickle=False) as arrays:
            baseline_source = arrays["source"][frames]
            baseline_accepted = arrays["slot_track"][frames] >= 0
        matches = ((source[:, :, None, :] == baseline_source[:, None, :, :])
                   & (source[:, :, None, :] >= 0)
                   & (baseline_source[:, None, :, :] >= 0)
                   & baseline_accepted[:, None, :, None]).sum(-1)
        correspondence_scores = np.stack([
            matches[:, np.arange(3), order].sum(1) for order in permutations], axis=1)
        order_indexes = correspondence_scores.argmax(1)
        mapping = permutations[order_indexes]
        sorted_scores = np.sort(correspondence_scores, axis=1)
        unique_mapping = sorted_scores[:, -1] > sorted_scores[:, -2]
        mapping[~unique_mapping] = -1
        reference_distance = np.take_along_axis(
            distances, np.maximum(mapping, 0)[:, :, None, None].repeat(3, 2), axis=-1)[..., 0]
        reference_distance[mapping[:, :, None].repeat(3, 2) < 0] = np.nan
        alternative = distances.copy()
        for slot in range(3):
            for t in range(len(frames)):
                if mapping[t, slot] >= 0:
                    alternative[t, slot, :, mapping[t, slot]] = np.nan
        alternative_distance = np.min(np.nan_to_num(alternative, nan=np.inf), axis=-1)
        reference_margin = alternative_distance - reference_distance
        reference_margin[~np.isfinite(reference_margin)] = np.nan
        correct_reference = (winner == mapping[:, :, None]) & valid & unique_mapping[:, None, None]
        reference_comparable = valid & unique_mapping[:, None, None]
        period = frames >= EVENT
        permutation_costs = np.stack([
            weighted_cost[period][:, np.arange(3), order].sum(1) for order in permutations], axis=1)
        cumulative_costs = np.cumsum(permutation_costs, axis=0)
        final_order = permutations[cumulative_costs[-1].argmin()]

        plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                             "axes.spines.right": False, "savefig.dpi": 160})
        figures = []

        def save(fig, name):
            fig.savefig(OUT / name, bbox_inches="tight")
            plt.close(fig)
            figures.append(name)
            record.output(name, OUT / name)

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
        for view, ax in enumerate(axes):
            ax.imshow(template_distance[:, :, view], vmin=0, vmax=.6, cmap="viridis")
            for i, j in itertools.product(range(3), repeat=2):
                ax.text(j, i, f"{template_distance[i,j,view]:.3f}", ha="center", va="center",
                        color="white" if template_distance[i,j,view] < .3 else "black")
            ax.set(xticks=range(3), yticks=range(3), xticklabels=["S0", "S1", "S2"],
                   yticklabels=["S0", "S1", "S2"], title=VIEWS[view],
                   xlabel="Prefix template slot", ylabel="Prefix template slot")
        fig.suptitle("Hellinger distance between view-specific prefix templates (first 20 frames)")
        save(fig, "01_template_separation.png")

        fig, axes = plt.subplots(3, 3, figsize=(15, 9), sharex=True, sharey=True,
                                 constrained_layout=True)
        for slot, view in itertools.product(range(3), repeat=2):
            ax = axes[slot, view]
            for template in range(3):
                ax.plot(frames, distances[:, slot, view, template], color=COLORS[template],
                        lw=1, label=f"prefix S{template}")
            ax.axvspan(frames[0], frames[19], color="gray", alpha=.15)
            ax.axvline(EVENT, color="black", ls=":", lw=1)
            ax.set(title=f"raw S{slot} / {VIEWS[view]}", ylim=(0, 1), ylabel="Hellinger distance")
            ax.ticklabel_format(axis="x", useOffset=False, style="plain")
        axes[0, 0].legend(fontsize=8)
        for ax in axes[-1]:
            ax.set_xlabel("Source frame; dotted line = previously observed event")
        save(fig, "02_distance_over_time.png")

        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True, constrained_layout=True)
        for slot in range(3):
            for view in range(3):
                axes[slot].plot(frames, quality[:, slot, view], label=VIEWS[view], lw=1)
            axes[slot].axvline(EVENT, color="black", ls=":")
            axes[slot].set(ylabel=f"raw S{slot} quality", ylim=(-.02, 1.03))
            axes[slot].legend(ncol=3, fontsize=8)
        agreement_rate = np.divide(view_agreement.sum(1), all_views_valid.sum(1),
                                   out=np.full(len(frames), np.nan),
                                   where=all_views_valid.sum(1) > 0)
        axes[3].plot(frames, agreement_rate, color="black", lw=1)
        axes[3].set(ylabel="3-view winner agreement\n(valid slots only)", ylim=(-.02, 1.03),
                    xlabel="Source frame")
        for ax in axes:
            ax.ticklabel_format(axis="x", useOffset=False, style="plain")
        fig.suptitle("Image support quality does not measure identity uniqueness")
        save(fig, "03_quality_and_view_agreement.png")

        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True, constrained_layout=True,
                                 gridspec_kw={"height_ratios": [3, 1]})
        cmap = ListedColormap(["#dddddd", *COLORS])
        rows = winner.transpose(1, 2, 0).reshape(9, len(frames))
        axes[0].imshow(rows, aspect="auto", interpolation="nearest", cmap=cmap, vmin=-1, vmax=2,
                       extent=[frames[0]-.5, frames[-1]+.5, 8.5, -.5])
        axes[0].set(yticks=range(9), yticklabels=[f"raw S{s}/{v}" for s in range(3) for v in VIEWS],
                    title="Nearest prefix template by view: orange S0, cyan S1, pink S2; gray unavailable")
        axes[1].imshow(mapping.T, aspect="auto", interpolation="nearest", cmap=cmap, vmin=-1, vmax=2,
                       extent=[frames[0]-.5, frames[-1]+.5, 2.5, -.5])
        axes[1].set(yticks=range(3), yticklabels=["raw S0", "raw S1", "raw S2"],
                    title="Baseline source correspondence for diagnosis only (not independent identity GT)",
                    xlabel="Source frame; gray = tied correspondence")
        for ax in axes:
            ax.axvline(EVENT, color="black", ls=":", lw=1)
            ax.ticklabel_format(axis="x", useOffset=False, style="plain")
        save(fig, "04_view_votes_and_reference.png")

        fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), constrained_layout=True)
        for order, values in zip(permutations, cumulative_costs.T, strict=True):
            axes[0].plot(frames[period], values, lw=1.3, label=str(order.tolist()))
        axes[0].set(title="Cumulative quality-weighted Hellinger-squared cost after fixed event",
                    ylabel="Cost sum (not probability / independent sample evidence)", xlabel="Source frame")
        axes[0].ticklabel_format(useOffset=False, style="plain")
        axes[0].legend(fontsize=8, title="Prefix IDs assigned to raw [S0,S1,S2]")
        for view in range(3):
            values = reference_margin[period, :, view]
            values = values[np.isfinite(values)]
            axes[1].hist(values, bins=np.linspace(-.6, .6, 45), histtype="step", lw=1.5,
                         label=VIEWS[view])
        axes[1].axvline(0, color="black", ls=":")
        axes[1].set(title="Reference-template advantage after event (model reference only)",
                    xlabel="Nearest-other distance minus reference-template distance",
                    ylabel="Correlated animal-view frames")
        axes[1].legend()
        save(fig, "05_cumulative_cost_and_drift.png")

        def ratio(numerator, denominator):
            return None if denominator == 0 else float(numerator / denominator)

        summary = {
            "frames": [int(frames[0]), int(frames[-1]) + 1], "prefix_frames": 20,
            "template_quality_sums": sums.tolist(),
            "template_hellinger_by_view": template_distance.transpose(2, 0, 1).tolist(),
            "positive_quality_counts": valid.sum(0).tolist(),
            "three_view_agreement_fraction_per_raw_slot": [
                ratio(view_agreement[:, s].sum(), all_views_valid[:, s].sum()) for s in range(3)],
            "post_event_reference_winner_fraction_by_raw_slot_view": [
                [ratio(correct_reference[period, s, v].sum(),
                       reference_comparable[period, s, v].sum()) for v in range(3)] for s in range(3)],
            "post_event_reference_margin_median_by_raw_slot_view": [
                [float(np.nanmedian(reference_margin[period, s, v])) for v in range(3)] for s in range(3)],
            "baseline_correspondence_unique_frames": int(unique_mapping.sum()),
            "post_event_cumulative_best_permutation": final_order.tolist(),
            "post_event_permutation_costs": {str(p.tolist()): float(c)
                                             for p, c in zip(permutations, cumulative_costs[-1], strict=True)},
            "limitations": ["Baseline correspondence is a model-derived diagnostic, not male/female GT.",
                            "Same raw slot can contain a different physical animal after a swap.",
                            "Prefix fitting uses no baseline labels; no thresholds or core code were tuned.",
                            "Adjacent 166-fps frames are correlated; cumulative margins are not calibrated confidence.",
                            "No validation or minute result was read."]}
        json_path = OUT / "summary.json"
        json_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        record.output("summary", json_path, sha256=sha256(json_path))
        arrays_path = OUT / "diagnostic_arrays.npz"
        np.savez_compressed(arrays_path, frame_idx=frames, templates=templates, quality=quality,
                            distances=distances, winner=winner, baseline_mapping=mapping,
                            reference_margin=reference_margin, cumulative_costs=cumulative_costs)
        record.output("arrays", arrays_path, sha256=sha256(arrays_path))
        images = "".join(f'<h2>{name}</h2><img src="{name}" width="1200">' for name in figures)
        previews = "".join(f'<p><a href="{os.path.relpath(p, OUT)}">{p.name}</a></p>'
                           for p in sorted(RUN.glob("preview_*.jpg")))
        html = '<!doctype html><meta charset="utf-8"><title>Development appearance diagnostic</title>'
        html += '<style>body{font:16px sans-serif;max-width:1250px;margin:30px auto}img{max-width:100%}</style>'
        html += '<h1>24-bin appearance: development-window diagnosis</h1>'
        html += '<p>Fixed first-20-frame templates; camera-specific, quality-weighted. '
        html += 'Baseline is a diagnostic reference, not identity ground truth. '
        html += 'No validation/minute results or ground-truth labels were read; no tracker was changed.</p>'
        html += '<p>Quality describes crop support and coherence, not calibrated identity discrimination. '
        html += 'Raw slot continuity does not imply animal continuity. Accumulated costs can confidently '
        html += 'prefer a wrong template when view or posture changes persist over correlated frames.</p>'
        html += images + '<h2>Existing development image previews (unchanged)</h2>' + previews
        (OUT / "index.html").write_text(html, encoding="utf-8")
        record.output("html", OUT / "index.html")
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
