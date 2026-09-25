"""Display all saved candidates reachable on the exact nonnegative-weight envelope.

Read saved reranking outputs, registered photographs and saved observation JSON.
No original candidate bank, video, mask, SL cache, calibration or solver is read.
Four candidates per page share the same camera crops across every case page.
"""

from __future__ import annotations

import argparse
import io
import math
import sys
import textwrap
from fractions import Fraction
from pathlib import Path

import numpy as np
import render_partial_body_bank_comparison as drawing

proof, require, prior = drawing.proof, drawing.require, drawing.prior
WORK, REPO, VIEWS = drawing.WORK, drawing.REPO, drawing.VIEWS
STAGE = "partial_body_weight_envelope_visual_review_v1"
ENVELOPE_SCHEMA = "partial_body_weight_envelope_v1"
COMPARISON_ROOT = WORK / "partial_body_saved_bank_comparison_v1"
COMPARISON_SHA = "10d29aad72ca98674abbcbaf386ce48244087cb4abae552b51e592b5dbb9910f"
DRAWING_SHA = "7658faac53fa0cb19ccfb6b1323144e26ef4c4ef8d4030fb4182d988af3d684e"
MODES = ("row_only", "row_plus_mask")
PER_PAGE = 4
MAX_CANDIDATES = 18


def fraction_value(value):
    require(
        isinstance(value, dict) and set(value) == {"numerator", "denominator", "decimal_approx"},
        "exact fraction descriptor required",
    )
    numerator, denominator = value["numerator"], value["denominator"]
    require(
        isinstance(numerator, str) and isinstance(denominator, str), "fraction strings required"
    )
    result = Fraction(int(numerator), int(denominator))
    require(result >= 0 and int(denominator) > 0, "nonnegative weight required")
    require(
        str(result.numerator) == numerator and str(result.denominator) == denominator,
        "fraction must be canonical",
    )
    require(float(result) == value["decimal_approx"], "fraction approximation differs")
    return result


def indices(value, count):
    require(
        isinstance(value, list)
        and value
        and all(type(i) is int and 0 <= i < count for i in value)
        and value == sorted(set(value)),
        "nonempty sorted candidate set required",
    )
    return value


def select_candidates(case):
    """Union all exact ties, including zero-only, saved/exact w=1 and factor minima."""
    count = case["candidate_count"]
    require(type(count) is int and 0 < count <= MAX_CANDIDATES, "candidate bound differs")
    require(set(case["modes"]) == set(MODES), "two envelope modes required")
    reasons = {i: [] for i in range(count)}

    def add(values, label):
        for i in indices(values, count):
            reasons[i].append(label)

    add(case["baseline_path_minimum_indices"], "baseline path minimum")
    for mode in MODES:
        row = case["modes"][mode]
        name = "row" if mode == "row_only" else "row+mask"
        add(row["factor_only_minimum_indices"], name + " factor minimum")
        add(row["weight_one_exact_minimum_indices"], name + " exact w=1")
        add(row["weight_one_saved_path_minimum_indices"], name + " saved w=1")
        envelope = row["envelope"]
        zero = indices(envelope["zero_weight_minimum_indices"], count)
        add(zero, name + " w=0")
        reachable = set(zero)
        boundary_values = []
        for boundary in envelope["positive_breakpoints"]:
            weight = fraction_value(boundary["weight"])
            require(weight > 0, "positive breakpoint required")
            boundary_values.append(weight)
            add(boundary["minimum_indices"], f"{name} w={float(weight):.5g} (tie)")
            reachable.update(boundary["minimum_indices"])
        require(boundary_values == sorted(set(boundary_values)), "breakpoint order differs")
        segments = envelope["intervals"]
        require(len(segments) == len(boundary_values) + 1, "complete envelope intervals required")
        for offset, segment in enumerate(segments):
            lower = fraction_value(segment["lower"])
            upper = None if segment["upper"] is None else fraction_value(segment["upper"])
            require(lower == [Fraction(0), *boundary_values][offset], "interval lower differs")
            require(
                upper == (boundary_values[offset] if offset < len(boundary_values) else None),
                "interval upper differs",
            )
            high = "inf" if upper is None else f"{float(upper):.5g}"
            add(segment["minimum_indices"], f"{name} w in ({float(lower):.5g},{high})")
            reachable.update(segment["minimum_indices"])
        require(
            sorted(reachable) == indices(envelope["ever_optimal_indices"], count),
            "envelope reachable set differs",
        )
    return {i: labels for i, labels in reasons.items() if labels}


def array_from(archive, descriptor, signatures):
    require(set(descriptor) == {"__array__", "signature"}, "array descriptor differs")
    key = descriptor["__array__"]
    value = archive[key]
    require(not value.dtype.hasobject, "object arrays forbidden")
    signature = proof.signature(value)
    require(signature == descriptor["signature"], "selected array signature differs")
    signatures[key] = signature
    result = value.copy()
    result.flags.writeable = False
    return result


def load_case(root, seal, envelope_case, evidence):
    arm, frame = envelope_case["arm"], envelope_case["frame_idx"]
    name = f"cases/{arm}_{frame}"
    companion = drawing.document(root, seal, name + ".json", evidence)
    path = drawing.previous.member(root, seal, name + ".npz", evidence)
    require(companion["archive"] == seal["outputs"][name + ".npz"], "case archive edge differs")
    tree = companion["tree"]
    require(
        tree["schema"] == drawing.COMPARISON_STAGE + "_case"
        and tree["arm"] == arm
        and tree["frame_idx"] == frame
        and tree["parent_history_fixed"] is True
        and tree["new_temporal_backtrace"] is False,
        "saved fixed-parent case differs",
    )
    bank, signatures = tree["baseline_bank"], {}
    with np.load(io.BytesIO(path.read_bytes()), allow_pickle=False) as archive:
        arrays = {
            key: array_from(archive, bank[key], signatures)
            for key in (
                "candidate_index",
                "candidate_pose_xyz",
                "candidate_parent",
                "candidate_plan",
                "base_local_cost",
                "parent_path_cost",
            )
        }
        terms = {
            mode: array_from(archive, tree["modes"][mode]["sixth_term"], signatures)
            for mode in ("baseline", *MODES)
        }
        projections = {
            view: array_from(archive, tree["projections"][view], signatures) for view in VIEWS
        }
        final = tree["final_selected_match"]
        final_matches = (
            array_from(
                archive, final["exact_pose_and_supported_source_candidate_indices"], signatures
            ).tolist()
            if final["available"]
            else None
        )
    count = envelope_case["candidate_count"]
    require(
        np.array_equal(arrays["candidate_index"], np.arange(count))
        and arrays["candidate_pose_xyz"].shape == (count, 3, 3, 3)
        and np.isfinite(arrays["candidate_pose_xyz"]).all()
        and arrays["candidate_plan"].shape == (count, 3, 3, 3)
        and arrays["candidate_plan"].dtype.kind in "iu"
        and np.all(arrays["candidate_plan"] >= -1)
        and arrays["candidate_parent"].shape == (count,)
        and arrays["candidate_parent"].dtype.kind in "iu"
        and np.all(arrays["candidate_parent"] >= 0),
        "saved candidate schema differs",
    )
    require(
        all(a.shape == (count, 3, 3, 2) for a in projections.values()), "projection shape differs"
    )
    require(
        len(bank["candidate_kind"]) == count
        and all(isinstance(k, str) for k in bank["candidate_kind"]),
        "candidate kinds differ",
    )
    for key in ("base_local_cost", "parent_path_cost"):
        require(
            arrays[key].shape == (count,) and np.isfinite(arrays[key]).all(), "base costs invalid"
        )
    for mode, value in terms.items():
        require(
            value.shape == (count,) and not np.isnan(value).any() and not np.isneginf(value).any(),
            "factor costs invalid",
        )
        if mode == "baseline":
            require(np.isfinite(value).all(), "baseline costs must be finite")
    selected = select_candidates(envelope_case)
    result = dict(
        arm=arm,
        frame=frame,
        count=count,
        arrays=arrays,
        terms=terms,
        projections=projections,
        kinds=bank["candidate_kind"],
        selected=selected,
        envelope=envelope_case,
        final_matches=final_matches,
        input_archive=companion["archive"],
        selected_array_signatures=signatures,
    )
    check_cost_metadata(result)
    return result


def check_cost_metadata(case):
    """Bind the envelope's cost/index table to the captured reranker output fields."""
    table = case["envelope"]["candidates"]
    require(len(table) == case["count"], "envelope cost table size differs")
    for i, row in enumerate(table):
        saved = candidate_record(case, i, allow_unselected=True)
        expected = dict(
            candidate_index=i,
            candidate_kind=saved["candidate_kind"],
            candidate_parent=saved["parent_index"],
            base_local_cost=saved["base_local_cost"],
            parent_path_cost=saved["parent_path_cost"],
            original_sixth_term=saved["original_sixth_term"],
            original_adjusted_local_cost=saved["baseline_local_cost"],
            original_adjusted_path_cost=saved["baseline_path_cost"],
            **{
                m + "_term": (
                    float(case["terms"][m][i]) if np.isfinite(case["terms"][m][i]) else None
                )
                for m in MODES
            },
        )
        require(row == expected, "envelope candidate cost metadata differs")
    base, parent = case["arrays"]["base_local_cost"], case["arrays"]["parent_path_cost"]
    for mode in ("baseline", *MODES):
        costs = parent + (base + case["terms"][mode])
        actual = np.flatnonzero(costs == costs.min()).tolist()
        expected = (
            case["envelope"]["baseline_path_minimum_indices"]
            if mode == "baseline"
            else case["envelope"]["modes"][mode]["weight_one_saved_path_minimum_indices"]
        )
        require(actual == expected, "saved w=1 or baseline minimum differs")
        if mode != "baseline":
            term = case["terms"][mode]
            actual_factor = np.flatnonzero(term == term.min()).tolist()
            require(
                actual_factor == case["envelope"]["modes"][mode]["factor_only_minimum_indices"],
                "factor minimum differs",
            )


def explicit_number(value):
    value = float(value)
    return value if math.isfinite(value) else "+inf" if value > 0 else "-inf"


def crop_for_case(case, images, points):
    crops = {}
    for view in VIEWS:
        projections = [case["projections"][view][i] for i in case["selected"]]
        lo, hi = drawing.visual.display_crop(images[view], None, points[view], projections)
        crops[view] = dict(lower_xy=lo.tolist(), upper_xy=hi.tolist())
    return crops


def candidate_record(case, index, allow_unselected=False):
    arrays = case["arrays"]
    base, parent = float(arrays["base_local_cost"][index]), float(arrays["parent_path_cost"][index])
    old = float(case["terms"]["baseline"][index])
    return dict(
        candidate_index=index,
        parent_index=int(arrays["candidate_parent"][index]),
        candidate_kind=case["kinds"][index],
        base_local_cost=base,
        parent_path_cost=parent,
        original_sixth_term=old,
        baseline_local_cost=base + old,
        baseline_path_cost=parent + (base + old),
        new_factor_costs={m: explicit_number(case["terms"][m][index]) for m in MODES},
        selection_reasons=(
            case["selected"].get(index, []) if allow_unselected else case["selected"][index]
        ),
        matches_saved_final_pose_and_sources=(
            None if case["final_matches"] is None else index in case["final_matches"]
        ),
        pose_xyz=arrays["candidate_pose_xyz"][index].tolist(),
        candidate_plan=arrays["candidate_plan"][index].tolist(),
        projections={v: prior.finite_json(case["projections"][v][index]) for v in VIEWS},
        plan_is_assigned_views_not_direct_support=True,
    )


def render_page(case, candidate_indices, images, points, crops, page_number, page_count):
    require(0 < len(candidate_indices) <= PER_PAGE, "page candidate count differs")
    fig, axes = prior.plt.subplots(
        len(candidate_indices),
        3,
        squeeze=False,
        figsize=(18, 1.8 + 5.2 * len(candidate_indices)),
        dpi=140,
    )
    try:
        frame = case["frame"]
        photo = "lossless native PNG" if frame in drawing.PRIMARY else "LOSSY JPEG CONTEXT"
        fig.suptitle(
            f"{case['arm']} native {frame} | page {page_number}/{page_count} | {photo}\n"
            "Saved fixed-parent fitted candidates; envelope over w>=0, both factors /15. "
            "Baseline, w=1, all envelope ties and factor minima included.\n"
            "S0 cyan / S1 yellow / S2 magenta = raw slots. "
            "Open M/C/T and dashed bodies are proposals. "
            "No recomputed support, trajectory, sex or physical-ID truth.\n"
            "Displayed weight bounds are rounded; "
            "exact fractions are preserved in the input envelope JSON.",
            fontsize=10,
        )
        for ri, index in enumerate(candidate_indices):
            row = candidate_record(case, index)
            reasons = "; ".join(row["selection_reasons"])
            # Wrap only the caption; do not truncate any candidate's interval or tie membership.
            reason_text = "\n".join(textwrap.wrap(reasons, width=85))
            for vi, view in enumerate(VIEWS):
                ax = axes[ri, vi]
                drawing.draw_panel(ax, images[view], points[view], case["projections"][view][index])
                ax.set_title(
                    f"{view} | candidate {index}; "
                    f"parent {row['parent_index']} {row['candidate_kind']}\n"
                    f"base={row['base_local_cost']:.6g}; parent={row['parent_path_cost']:.6g}; "
                    f"old6={row['original_sixth_term']:.6g}\n"
                    f"baseline local/path={row['baseline_local_cost']:.6g}/"
                    f"{row['baseline_path_cost']:.6g}; "
                    f"row={case['terms']['row_only'][index]:.6g}; "
                    f"row+mask={case['terms']['row_plus_mask'][index]:.6g}\n" + reason_text,
                    fontsize=7.5,
                )
                lo, hi = crops[view]["lower_xy"], crops[view]["upper_xy"]
                ax.set_xlim(lo[0], hi[0])
                ax.set_ylim(hi[1], lo[1])
                ax.tick_params(labelsize=7)
        fig.tight_layout(rect=(0, 0, 1, 0.92 if len(candidate_indices) > 1 else 0.80))
        stream = io.BytesIO()
        fig.savefig(stream, format="png")
        return stream.getvalue()
    finally:
        prior.plt.close(fig)


def load_context(envelope_path, digest, evidence):
    envelope_path = Path(envelope_path).resolve()
    require(
        envelope_path.is_relative_to(WORK) and proof.valid_sha(digest), "explicit envelope required"
    )
    evidence.add(envelope_path, digest)
    envelope = proof.read_json(envelope_path)
    require(envelope["schema"] == ENVELOPE_SCHEMA and envelope["status"] == "ok", "envelope failed")
    require(
        envelope["common_saved_normalizer"] == 15.0
        and envelope["original_parent_history_fixed"] is True
        and envelope["old_sixth_term_replaced_not_added"] is True
        and envelope["infinite_slope_excluded_even_at_zero"] is True
        and envelope["weight_selected_or_adopted"] is False
        and envelope["new_fit_or_trajectory_or_ID_created"] is False
        and envelope["physical_ID_truth"] is False,
        "envelope scientific scope differs",
    )
    ref = envelope["producer_seal"]
    require(
        Path(ref["path"]).resolve() == COMPARISON_ROOT / "comparison.provenance.json"
        and ref["sha256"] == COMPARISON_SHA,
        "envelope producer differs",
    )
    source = envelope["source"]
    require(
        Path(source["path"]).resolve() == WORK / "diagnose_partial_body_weight_envelope.py",
        "envelope source path differs",
    )
    evidence.add(Path(source["path"]), source["sha256"])
    evidence.add(WORK / "render_partial_body_bank_comparison.py", DRAWING_SHA)
    context = drawing.load_context(COMPARISON_ROOT, COMPARISON_SHA, evidence)
    seal = proof.read_json(COMPARISON_ROOT / "comparison.provenance.json")
    require(
        (COMPARISON_ROOT / "comparison.provenance.json").stat().st_size == ref["size_bytes"],
        "producer size differs",
    )
    expected = {(c["arm"], c["frame"]) for c in context["cases"]}
    require(
        envelope["case_count"] == len(envelope["cases"]) == len(expected) == 11
        and {(c["arm"], c["frame_idx"]) for c in envelope["cases"]} == expected,
        "all eleven envelope cases required",
    )
    for case in envelope["cases"]:
        select_candidates(case)
    context.update(
        envelope=envelope,
        envelope_path=str(envelope_path),
        envelope_sha256=digest,
        comparison_seal=seal,
    )
    context["source_code"][str(Path(__file__).resolve())] = evidence.add(Path(__file__).resolve())
    context["source_code"][str(WORK / "render_partial_body_bank_comparison.py")] = DRAWING_SHA
    context["source_code"][source["path"]] = source["sha256"]
    return context


def run(envelope_path, digest, output):
    output = Path(output).resolve()
    require(output.is_relative_to(WORK) and not output.exists(), "fresh workspace output required")
    require(
        "torch" not in sys.modules or not sys.modules["torch"].cuda.is_initialized(), "CPU only"
    )
    out, evidence, started = proof.Output(output), proof.Evidence(), proof.utc()
    try:
        context = load_context(envelope_path, digest, evidence)
        out.write("weight_envelope.json", Path(context["envelope_path"]).read_bytes())
        photos, points, photograph_refs = {}, {}, []
        frames = sorted({c["frame_idx"] for c in context["envelope"]["cases"]})
        for frame in frames:
            photos[frame], points[frame] = {}, {}
            for view in VIEWS:
                image, sl, reference = drawing.photograph_and_points(context, frame, view, evidence)
                photos[frame][view], points[frame][view] = image, sl
                photograph_refs.append(reference)
        counts = [len(select_candidates(c)) for c in context["envelope"]["cases"]]
        expected_pages = sum(math.ceil(count / PER_PAGE) for count in counts)
        require(expected_pages <= 55, "page bound exceeded")
        out.write(
            "plan.json",
            dict(
                schema=STAGE,
                envelope_path=context["envelope_path"],
                envelope_sha256=digest,
                comparison_root=str(COMPARISON_ROOT),
                comparison_sha256=COMPARISON_SHA,
                source_code=context["source_code"],
                photographs=photograph_refs,
                rows_per_page=PER_PAGE,
                candidate_counts_displayed=counts,
                expected_pages=expected_pages,
                maximum_pages=55,
                original_video_reads=0,
                original_candidate_bank_reads=0,
                original_SL_cache_reads=0,
                original_mask_reads=0,
                rig_reads=0,
                new_reranking_archives_to_read=11,
                new_trajectory_created=False,
                physical_ID_truth=False,
                support_recomputed=False,
            ),
        )
        records = []
        for envelope_case in context["envelope"]["cases"]:
            case = load_case(COMPARISON_ROOT, context["comparison_seal"], envelope_case, evidence)
            selected = list(case["selected"])
            crops = crop_for_case(case, photos[case["frame"]], points[case["frame"]])
            count = math.ceil(len(selected) / PER_PAGE)
            pages = []
            for page, start in enumerate(range(0, len(selected), PER_PAGE), 1):
                subset = selected[start : start + PER_PAGE]
                png = render_page(
                    case, subset, photos[case["frame"]], points[case["frame"]], crops, page, count
                )
                name = f"{case['arm']}_f{case['frame']}_p{page:02d}.png"
                pages.append(dict(page=page, candidate_indices=subset, image=out.write(name, png)))
            records.append(
                dict(
                    arm=case["arm"],
                    frame_idx=case["frame"],
                    crops_shared_by_every_page=crops,
                    selected_candidate_indices=selected,
                    pages=pages,
                    candidates=[candidate_record(case, i) for i in selected],
                    input_archive=case["input_archive"],
                    selected_array_signatures=case["selected_array_signatures"],
                    final_candidate_match_is_reference_not_selection=True,
                )
            )
            print(
                f"Rendered {case['arm']} native {case['frame']}: "
                f"{len(selected)} candidates / {count} pages",
                flush=True,
            )
        require(sum(len(c["pages"]) for c in records) == expected_pages, "page coverage differs")
        out.write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                cases=records,
                contact_sheets=expected_pages,
                photographs_decoded=len(photograph_refs),
                new_reranking_archives_decoded=len(records),
                all_requested_candidates_displayed=True,
                all_markers_are_open_proposals=True,
                raw_slots_preserved=True,
                support_recomputed=False,
                physical_ID_truth=False,
                new_trajectory_created=False,
                negative_photographs_are_lossy=True,
                exact_weight_intervals_in="weight_envelope.json",
            ),
        )
        evidence.final_guard()
        out.final_guard()
        require(
            "torch" not in sys.modules or not sys.modules["torch"].cuda.is_initialized(),
            "CUDA initialized",
        )
        seal = dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=out.records,
        )
    except BaseException as error:
        seal = dict(
            stage=STAGE,
            status="failed",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=out.records,
            error=repr(error),
        )
        (output / "render.provenance.json").write_bytes(proof.json_bytes(seal))
        raise
    (output / "render.provenance.json").write_bytes(proof.json_bytes(seal))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envelope-json", required=True, type=Path)
    parser.add_argument("--envelope-sha", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.envelope_json, args.envelope_sha, args.output)


if __name__ == "__main__":
    main()
