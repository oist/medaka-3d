"""Disabled same-human-review comparison of assigned SLEAP mouths for 13 streams.

Seven classical C34 labels and three saved 3D methods (raw and beam) use the
same registered SLEAP pixels and frozen anonymous encounter matcher. This is
an additional observation-association layer, not a new tracker or reconstructed
C34 mouth. The successful upstream 3D projection evaluation remains separate.
"""

from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path

import compare_fast_tail_transition_review as upstream
import numpy as np
import score_c34_female_five_minute_anchors as classical

proof, require = upstream.proof, upstream.require
WORK, REPO, VIEWS = upstream.WORK, upstream.REPO, upstream.VIEWS
STAGE = "transition_review_assigned_SL_mouth_comparison_v1"
EXECUTION_ENABLED = False
UPSTREAM_FALSE_SHA = "13c07c31b4ffbf3c8a54092e70bbd6a0a340c020f8c933eb7b9047916295648d"
C34_SOURCE_SHA = "9a46a7dabec305d082db4cdd766e193815577815b0d6027cfa773c8a8f1078b2"
LAYER = "assigned_SLEAP_mouth_human_spatial"
SL_FIELDS = {"experiment", "view", "nodes", "n_frames", "frame_idx", "points", "score"}


def enabled_source_sha(data):
    """Bind the frozen upstream with only its real execution assignment enabled."""
    lines = data.splitlines(keepends=True)
    old, new = b"EXECUTION_ENABLED = False", b"EXECUTION_ENABLED = True"
    selected = [i for i, line in enumerate(lines) if line.rstrip(b"\r\n") in (old, new)]
    require(len(selected) == 1, "one upstream execution assignment required")
    i = selected[0]
    lines[i] = lines[i].replace(new, old)
    require(
        hashlib.sha256(b"".join(lines)).hexdigest() == UPSTREAM_FALSE_SHA,
        "upstream changed beyond its execution assignment",
    )
    lines[i] = lines[i].replace(old, new)
    return hashlib.sha256(b"".join(lines)).hexdigest()


def source_code():
    # Upstream checks 21 explicit used files; no dynamic sys.modules inventory.
    enabled_source_sha(Path(upstream.__file__).read_bytes())
    code = upstream.source_code()
    for path, expected in (
        (Path(classical.__file__), C34_SOURCE_SHA),
        (Path(classical.baseline.__file__), classical.BASELINE_SOURCE_SHA),
        (
            REPO / "tools/c01_s16_predictions.py",
            "ebc383270e3dd845bf0cd2355320bc546fe607c9987dfb415f7736a6b53d9106",
        ),
    ):
        require(proof.sha(path) == expected, "frozen source differs: " + path.name)
        code[str(path.resolve())] = expected
    code[str(Path(__file__).resolve())] = proof.sha(__file__)
    return code


def bounded_anchor_archive(root, seal, name, keys, evidence):
    require(0 < seal["outputs"][name]["size_bytes"] <= 65536, "small anchor archive required")
    data = upstream.member(root, seal, name, evidence, binary=True)
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        require(
            len(archive.files) == len(set(archive.files)) == len(keys)
            and set(archive.files) == set(keys),
            "selected archive membership differs",
        )
        arrays = {key: archive[key] for key in keys}
    for value in arrays.values():
        value.flags.writeable = False
    return arrays


def validate_selected_mapping(mapping, frames, saved):
    """Only saved three-frame fields; upstream verified the original beam14/P012."""
    require(np.array_equal(mapping["frame_idx"], frames), "selected mapping native order")
    state, permutation = mapping["path_state"], mapping["permutation_by_id"]
    require(
        state.shape == (3,)
        and state.dtype.kind in "iu"
        and np.all((state >= 0) & (state < 6))
        and permutation.shape == (3, 3)
        and permutation.dtype.kind in "iu"
        and np.array_equal(permutation, upstream.prior.PERMS[state])
        and permutation.tolist() == saved["permutation_by_id"],
        "selected permutation differs",
    )
    gap = mapping["conditional_identity_alternative_gap"]
    status = mapping["conditional_identity_alternative_status"]
    require(
        gap.shape == status.shape == (3, 3)
        and gap.dtype.kind == "f"
        and not np.isnan(gap).any()
        and np.all(gap >= 0)
        and np.isin(status, ["finite_gap", "tie_within_tolerance", "not_retained_unknown"]).all()
        and np.array_equal(np.isposinf(gap), status == "not_retained_unknown")
        and status.tolist() == saved["conditional_ID_alternative_status"],
        "saved conditional ambiguity differs",
    )
    encoded = [["+Infinity" if np.isposinf(x) else float(x) for x in row] for row in gap]
    require(encoded == saved["conditional_ID_alternative_gap"], "saved conditional gaps differ")


def validate_upstream_summary(summary, parsed):
    upstream.comparison.common.exact_fields(
        summary,
        dict(
            schema=upstream.STAGE,
            status="ok",
            native_interval=[upstream.START, upstream.STOP],
            terminal_horizon=upstream.STOP,
            review_clip_range=upstream.CLIP,
            anchor_frames=parsed.anchor_frames.tolist(),
            requested_endpoints_per_arm=9,
            requested_links_per_arm=6,
            candidate_fields_decoded=5,
            candidate_member_count=21,
            mapping_fields_decoded=5,
            mapping_member_count=14,
            terminal_mapping_arrays_decoded=False,
            trajectory_modified=False,
            reviews_cleared=False,
            human_labels_created=False,
            biological_identity_assigned=False,
            all_five_minute_identity_truth=False,
            dense_swap_zero_certified=False,
        ),
        "successful upstream comparison scope differs",
    )
    require(
        set(summary["results"]) == set(upstream.ARMS)
        and summary["original_review_sha256"] == hashlib.sha256(parsed.review_bytes).hexdigest()
        and summary["review_status_preserved"] == parsed.review["status"]
        and summary["biological_naming_preserved_not_applied"]
        == parsed.review.get("biological_naming"),
        "upstream human labels differ",
    )


def load_upstream(root, digest, evidence):
    root = upstream.workspace(root)
    seal = upstream.seal_at(root, "comparison.provenance.json", digest, upstream.STAGE, evidence)
    source = Path(upstream.__file__).resolve()
    actual_sha = enabled_source_sha(source.read_bytes())
    upstream.depth.edge(seal, source, actual_sha)
    evidence.add(source, actual_sha)
    raw = upstream.member(root, seal, "original_review.json", evidence, binary=True)
    manifest = upstream.member(root, seal, "source_manifest.json", evidence, binary=True)
    parsed = upstream.human.parse_review(raw, manifest, clip_range=upstream.CLIP)
    summary = upstream.member(root, seal, "summary.json", evidence)
    validate_upstream_summary(summary, parsed)
    require(
        summary["source_sha256"].get(str(source)) == actual_sha,
        "upstream summary executed source differs",
    )
    rows, mappings = {}, {}
    for arm in upstream.ARMS:
        rows[arm] = bounded_anchor_archive(
            root, seal, f"anchors/{arm}.npz", upstream.FIELDS, evidence
        )
        upstream.validate_rows(rows[arm], parsed.anchor_frames)
        saved = summary["results"][arm]
        require(
            rows[arm]["sources"].tolist() == saved["raw_slots"]["saved_source_rows"]
            and rows[arm]["observed"].tolist() == saved["raw_slots"]["saved_node_observed"]
            and rows[arm]["identity_review"].tolist()
            == saved["raw_slots"]["saved_identity_review"],
            "selected raw source/support/review differs from upstream report",
        )
        mappings[arm] = bounded_anchor_archive(
            root, seal, f"identity_anchors/{arm}.npz", upstream.MAPPING_FIELDS, evidence
        )
        validate_selected_mapping(mappings[arm], parsed.anchor_frames, saved)
        ordered = upstream.prior.reorder_rows(rows[arm], mappings[arm]["permutation_by_id"])
        automatic = saved["automatic_initial_slot_IDs"]
        require(
            ordered["sources"].tolist() == automatic["saved_source_rows"]
            and ordered["observed"].tolist() == automatic["saved_node_observed"]
            and ordered["identity_review"].tolist() == automatic["saved_identity_review"],
            "upstream automatic whole-body source/support/review differs",
        )
    return dict(
        parsed=parsed,
        summary=summary,
        rows=rows,
        mappings=mappings,
        seal_sha256=digest,
        root=str(root),
        summary_descriptor=seal["outputs"]["summary.json"],
    )


def registered_references(shared, evidence):
    """The top cache is explicitly registered too; female57's two sides do not suffice."""
    catalog = classical.read_json(
        classical.INPUT_ROOT / "staging_manifest.json", classical.CATALOG_SHA, evidence
    )
    entries = classical.catalog_entries(catalog)
    baseline = classical.baseline
    c34 = {
        lam: classical.registered_file(
            entries, f"{baseline.BASELINE}/C34-label__{baseline.EXPERIMENT}__lam{lam}.npz"
        )
        for lam in classical.LAMBDAS
    }
    expected = {
        f"{baseline.BASELINE}/C34-label__{baseline.EXPERIMENT}__lam{x}.npz"
        for x in classical.LAMBDAS
    }
    require(
        {key for key in entries if key.startswith(baseline.BASELINE + "/C34-label__")} == expected,
        "registered classical lambda set differs",
    )
    sl = {}
    for view in VIEWS:
        name = f"work/c01_s16_predictions/bottomup/{baseline.EXPERIMENT}__{view}.npz"
        row = classical.registered_file(entries, name)
        require(
            shared.get(row["path"]) == row["sha256"], "registered/new-producer SL differs: " + view
        )
        sl[view] = row
    require(
        {p for p in shared if Path(p).suffix == ".npz"} == {r["path"] for r in sl.values()},
        "exact common SL3 binding required",
    )
    return c34, sl


def select_mouths(index, points, frames):
    """Keep only requested native rows, preserving each view's original local row order."""
    require(
        index.ndim == 1
        and index.dtype.kind in "iu"
        and np.all(index >= 0)
        and np.all(index < classical.FRAME_COUNT)
        and np.all(index[1:] >= index[:-1]),
        "SL native index order/range differs",
    )
    require(points.shape == (len(index), 3, 2) and points.dtype.kind == "f", "SL point schema")
    upstream.selected_indices(frames)
    require(np.all(frames[1:] > frames[:-1]), "selected native frame order differs")
    selected = []
    ranges = []
    for frame in frames:
        lo, hi = np.searchsorted(index, [frame, frame + 1])
        selected.append(points[lo:hi, 0].copy())
        ranges.append([int(lo), int(hi)])
    values = np.concatenate(selected, axis=0)
    offsets = np.r_[0, np.cumsum([len(row) for row in selected])].astype(np.int64)
    return dict(
        frame_idx=frames.copy(),
        offsets=offsets,
        mouth_uv=values,
        original_flat_row_ranges=np.asarray(ranges, np.int64),
    )


def load_mouths(reference, view, frames):
    path = classical.local(reference["path"])
    before = classical.file_stat(path)
    payload = path.read_bytes()
    require(
        len(payload) == reference["size_bytes"]
        and hashlib.sha256(payload).hexdigest() == reference["sha256"]
        and classical.file_stat(path) == before,
        "SL captured bytes/stat differ",
    )
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        require(
            len(archive.files) == len(set(archive.files)) == 7 and set(archive.files) == SL_FIELDS,
            "registered SL7 schema",
        )
        count = archive["n_frames"]
        require(
            count.shape == () and count.dtype.kind in "iu" and int(count) == classical.FRAME_COUNT,
            "SL declared native extent differs",
        )
        require(
            str(archive["experiment"]) == classical.baseline.EXPERIMENT
            and str(archive["view"]) == view
            and tuple(archive["nodes"]) == ("mouth", "pectoral", "tail"),
            "SL cache identity differs",
        )
        index, points = archive["frame_idx"], archive["points"]
    selected = select_mouths(index, points, frames)
    for value in selected.values():
        value.flags.writeable = False
    return selected, dict(
        **reference,
        file_stat=before,
        archive_reads=1,
        numeric_fields_decoded=["frame_idx", "points"],
        score_decoded=False,
        selected_signatures={k: proof.signature(v) for k, v in selected.items()},
    )


def assigned_mouths(source, frames, caches):
    """No matching/inference here: literal local row lookup independently in each view."""
    require(
        source.shape == (3, 3, 3) and source.dtype.kind == "i" and np.all(source >= -1),
        "integer assigned source rows >=-1 required",
    )
    require(set(caches) == set(VIEWS), "all three independently indexed views required")
    points = np.full((3, 3, 3, 2), np.nan)
    reasons = np.full((3, 3, 3), "unassigned_source", dtype="U32")
    for v, view in enumerate(VIEWS):
        cache = caches[view]
        require(np.array_equal(cache["frame_idx"], frames), "SL selected frame order differs")
        offsets, mouths = cache["offsets"], cache["mouth_uv"]
        require(
            offsets.shape == (4,)
            and offsets.dtype.kind in "iu"
            and offsets[0] == 0
            and np.all(offsets[1:] >= offsets[:-1])
            and offsets[-1] == len(mouths)
            and mouths.shape == (len(mouths), 2)
            and mouths.dtype.kind == "f",
            "SL selected schema",
        )
        for t in range(3):
            local = mouths[offsets[t] : offsets[t + 1]]
            for slot in range(3):
                row = int(source[t, slot, v])
                if row == -1:
                    continue
                require(row < len(local), "assigned source row outside native SL frame")
                if np.isfinite(local[row]).all():
                    points[t, slot, v] = local[row]
                    reasons[t, slot, v] = "assigned_finite_mouth"
                else:
                    reasons[t, slot, v] = "nonfinite_assigned_mouth"
    return points, reasons


def score_stream(parsed, source, caches, availability, *, review=None, eligibility_name):
    parsed = upstream.checked_review(parsed)
    require(availability.shape == (3, 3) and availability.dtype.kind == "b", "eligibility mask")
    if review is not None:
        require(review.shape == (3, 3) and review.dtype.kind == "b", "saved review mask")
    before = (
        proof.signature(source),
        proof.signature(availability),
        None if review is None else proof.signature(review),
    )
    pixels, reasons = assigned_mouths(source, parsed.anchor_frames, caches)
    args = (parsed.anchor_frames, pixels)
    tail = (parsed.anchor_frames, parsed.human_points, parsed.continuity)
    common = upstream.encounter_review.score_encounter_review(
        *args, np.ones((3, 3), bool), np.zeros((3, 3), bool), *tail
    )["raw"]
    filtered = upstream.encounter_review.score_encounter_review(
        *args, availability, np.zeros((3, 3), bool) if review is None else review, *tail
    )
    scores = {LAYER: common, eligibility_name: filtered["raw"]}
    if review is not None:
        scores[eligibility_name + "_and_original_review"] = filtered["accepted"]
    for value in scores.values():
        require(
            value["requested_anchors"] == 9
            and value["requested_animal_intervals"] == 6
            and sum(value["interval_counts"].values()) == 6,
            "all9/all6 denominator changed",
        )
    require(
        before
        == (
            proof.signature(source),
            proof.signature(availability),
            None if review is None else proof.signature(review),
        ),
        "saved fields mutated",
    )
    return dict(
        scores=scores,
        local_human_endpoint_links=upstream.local_links(common, parsed.continuity),
        assigned_SL_mouth_pixels=proof.safe_json(pixels.tolist()),
        source_rows=source.tolist(),
        source_lookup_status=reasons.tolist(),
        original_eligibility=availability.tolist(),
        original_review=None if review is None else review.tolist(),
        method_specific_eligibility=eligibility_name,
        common_match_radius_px=5.0,
        common_marked_view_minimum=1,
        all_marked_views_must_agree=True,
        eligibility_filters_applied_after_all_competitor_matching=True,
        equivalent_review_flag_available=review is not None,
        core_reason_observed_means=eligibility_name,
        frozen_core_reason_translation=(
            {"mouth lacks declared measured support": "C34 saved label unavailable"}
            if eligibility_name == "C34_saved_label_available"
            else {}
        ),
        physical_swap_event_count_certified=False,
        interior_identity_certified=False,
    )


def evaluate_three_dimensional(parsed, rows, mapping, caches):
    upstream.validate_rows(rows, parsed.anchor_frames)
    before = {k: proof.signature(v) for k, v in rows.items()}
    ordered = upstream.prior.reorder_rows(rows, mapping["permutation_by_id"])
    results = {}
    for space, data in (("raw_slots", rows), ("automatic_initial_slot_IDs", ordered)):
        results[space] = score_stream(
            parsed,
            data["sources"][:, :, :, 0],
            caches,
            data["observed"][:, :, 0],
            review=data["identity_review"],
            eligibility_name="saved_observed_mouth",
        )
    require(before == {k: proof.signature(v) for k, v in rows.items()}, "raw whole body changed")
    return results


def run(args):
    require(EXECUTION_ENABLED, "source-only: common source-mouth evaluation is disabled")
    output = Path(args.output).resolve()
    require(
        output.is_relative_to(WORK.resolve()) and not output.exists(),
        "fresh workspace output required",
    )
    upstream.old.cpu_only()
    out, evidence, captures, started = proof.Output(output), proof.Evidence(), {}, proof.utc()
    try:
        code = source_code()
        for path, sha in code.items():
            evidence.add(path, sha)
        context = load_upstream(args.human_comparison_root, args.human_comparison_sha, evidence)
        parsed, saved = context["parsed"], context["summary"]
        refs, sl_refs = registered_references(
            saved["lineage"]["shared_SL_calibration_sha256"], evidence
        )
        caches = {}
        for view in VIEWS:
            caches[view], captures["SL/" + view] = load_mouths(
                sl_refs[view], view, parsed.anchor_frames
            )
            out.write(f"SL_anchors/{view}.npz", caches[view])
        results = {}
        for lam in classical.LAMBDAS:
            rows, captures["C34/" + lam] = classical.load_anchor_rows(
                refs[lam], parsed.anchor_frames
            )
            key = "C34_lambda_" + lam
            results[key] = score_stream(
                parsed,
                rows["source"],
                caches,
                rows["slot_track"] >= 0,
                eligibility_name="C34_saved_label_available",
            )
            out.write(f"C34_anchors/lambda_{lam}.npz", rows)
        for arm in upstream.ARMS:
            scored = evaluate_three_dimensional(
                parsed, context["rows"][arm], context["mappings"][arm], caches
            )
            for space, value in scored.items():
                results[arm + "/" + space] = value
        require(len(results) == 13, "all7 classical plus3 raw/beam methods required")
        out.write("original_review.json", parsed.review_bytes)
        out.write("source_manifest.json", parsed.manifest_bytes)
        out.write("captured_new_inputs.json", captures)
        out.write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                layer=LAYER,
                requested_endpoints_per_stream=9,
                requested_links_per_stream=6,
                streams=13,
                anchor_frames=parsed.anchor_frames.tolist(),
                results=results,
                upstream_comparison_root=context["root"],
                upstream_comparison_seal_sha256=context["seal_sha256"],
                unchanged_3D_primary_reference=context["summary_descriptor"],
                inherited_lineage=saved["lineage"],
                inherited_identity_bindings=saved["identity_bindings"],
                inherited_conditional_ID_ambiguity={
                    arm: {
                        k: saved["results"][arm][k]
                        for k in (
                            "permutation_by_id",
                            "conditional_ID_alternative_gap",
                            "conditional_ID_alternative_status",
                        )
                    }
                    for arm in upstream.ARMS
                },
                full_beam14_and_initial_P012_verification_inherited=True,
                selected_identity_archive_members_directly_verified=5,
                full_beam14_reopened=False,
                common_observation_is_assigned_SL_pixel_not_3D_projection=True,
                latent_mask_supported_geometry_evaluation_remains_only_in_upstream_3D_primary=True,
                original_review_sha256=hashlib.sha256(parsed.review_bytes).hexdigest(),
                biological_naming_preserved_not_applied=parsed.review.get("biological_naming"),
                classical_references=refs,
                SL3_references=sl_refs,
                all7_lambdas_scored=True,
                lambda_selected=False,
                old_C34_producer_provenance_requalified=False,
                old_C34_full_recording_offline_future_information=True,
                same_seed_or_information_conditions=False,
                source_sha256=code,
                SL_archive_reads=3,
                C34_archive_reads=7,
                full_3D_candidate_or_mapping_reads=0,
                selected_3D_anchor_archive_reads=6,
                original_video_mask_calibration_checkpoint_reads=0,
                inference_or_solver_calls=0,
                GPU_used=False,
                geometry_created=False,
                track_or_review_modified=False,
                biological_identity_assigned=False,
                all_three_physical_ID_truth=False,
                dense_swap_zero_certified=False,
                endpoint_only_diagnostics_are_not_interior_certificates=True,
                capture_final_guard=(
                    "each new archive captured once with SHA; "
                    "original file stat unchanged at finish"
                ),
            ),
        )
        require(source_code() == code, "used source changed")
        upstream.old.cpu_only()
        evidence.final_guard()
        for record in captures.values():
            require(
                classical.file_stat(Path(record["path"])) == record["file_stat"],
                "captured input changed",
            )
        out.final_guard()
        status = dict(status="ok")
    except BaseException as error:
        status = dict(status="failed", error_type=type(error).__name__, error=str(error))
        out.write("failure.json", status)
        raise
    finally:
        inputs = {
            **evidence.paths,
            **{record["path"]: record["sha256"] for record in captures.values()},
        }
        with (output / "comparison.provenance.json").open("xb") as stream:
            stream.write(
                proof.json_bytes(
                    dict(
                        stage=STAGE,
                        started_at=started,
                        finished_at=proof.utc(),
                        inputs=inputs,
                        outputs=out.records,
                        **status,
                    )
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-comparison-root", type=Path, required=True)
    parser.add_argument("--human-comparison-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
