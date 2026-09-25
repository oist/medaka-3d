"""Bind an actual anonymous human review to compared five-minute trajectories.

Compare the same human mouth marks against Original, FastTail raw slots, and
optionally the independently computed FastTail ID permutation. This maps local
endpoints only: neither interval interiors nor all three physical IDs become
certified, and no trajectory, review flag or annotation is changed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path

import compare_fast_tail_continuation as comparison
import fast_tail_review_identity_binding as identity_binding
import fast_tail_review_source_binding as source_binding
import import_long_review as review
import map_long_review_to_final as legacy
import numpy as np

proof, require = comparison.proof, comparison.require
WORK, REPO = comparison.WORK, comparison.REPO
FIRST, START, STOP = 49800, 66400, 116200
STAGE = "human_review_fast_tail_five_minute_mapping_v1"
COMPARISON_SHA = "2e4a749d5ea13116f05a622afd47478af311525fbd856b32a116bd9ac765acba"
PINS = {
    "import_long_review.py": "4ee9a2af648231658714c017ff3d48232700f09a701092f7a70a75869421bcc5",
    "map_long_review_to_final.py": (
        "b0ad0a2225efe86a9cd4ef51b05818f3cabe1de780623949dd513d7b1528ee78"
    ),
    "fast_tail_review_source_binding.py": (
        "6938a0a578b3518c75d8e07373e1df36e96db691ba5732ddf2f572d3a0b6c39f"
    ),
    "fast_tail_review_identity_binding.py": (
        "1245e0f2bf5c632c1e337ef40657f68f27c4d0e69eb3fe2155a6ed216f9f836a"
    ),
}
EXECUTION_READY = True


def source_code():
    code = comparison.source_code()
    require(
        code.get(str(Path(comparison.__file__).resolve())) == COMPARISON_SHA,
        "reviewed five-minute comparison source changed",
    )
    for name, digest in PINS.items():
        require(
            proof.valid_sha(digest) and code.get(str(WORK / name)) == digest,
            "reviewed human mapping source required: " + name,
        )
    return code


def validate_comparison(plan, summary):
    expected = dict(
        schema=comparison.STAGE,
        status="ok",
        horizon_stop=STOP,
        native_interval=[START, STOP],
        native_frames=STOP - START,
        whole_five_minute_process_complete=True,
        final_tracking_seal_verified=True,
        same_parent_and_horizon=True,
        candidate_all21_and_source_arrays_unchanged=True,
        review_modified=False,
        all_three_ID_truth=False,
        external_pose_resets=0,
    )
    require(
        all(type(summary.get(k)) is type(v) and summary[k] == v for k, v in expected.items()),
        "successful complete five-minute body comparison required",
    )
    require(
        plan["schema"] == comparison.STAGE
        and plan["horizon_stop"] == STOP
        and plan["candidate_references"] == summary["candidate_references"]
        and set(plan["candidate_references"]) == {"original", "fast_tail"}
        and proof.valid_sha(plan["milestone_sha256"])
        and proof.valid_sha(plan["final_tracking_sha256"]),
        "same candidates and successful final tracking seal required",
    )


def load_comparison(root, digest, evidence):
    root = root.resolve(strict=True)
    require(root.is_relative_to(WORK) and proof.valid_sha(digest), "bound workspace comparison")
    seal = comparison.json_input(root / "comparison.provenance.json", evidence, digest)
    require(seal["stage"] == comparison.STAGE and seal["status"] == "ok", "comparison failed")
    require(
        seal["inputs"].get(str(Path(comparison.__file__).resolve())) == COMPARISON_SHA,
        "comparison source edge differs",
    )
    plan = proof.read_json(comparison.custom_member(seal, root, "plan.json", evidence))
    summary = proof.read_json(comparison.custom_member(seal, root, "summary.json", evidence))
    validate_comparison(plan, summary)
    new = comparison.milestone_metadata(
        STOP, plan["milestone_sha256"], evidence, plan["final_tracking_sha256"]
    )
    original = comparison.original_metadata(new, STOP, evidence)
    for path, sha in (
        (comparison.BRANCH / "tracking.provenance.json", plan["final_tracking_sha256"]),
        (
            comparison.BRANCH / f"milestones/through_{STOP:06d}/summary.json",
            plan["milestone_sha256"],
        ),
        (
            comparison.ORIGINALS[STOP]["root"] / "continuation.provenance.json",
            comparison.ORIGINALS[STOP]["seal"],
        ),
    ):
        require(seal["inputs"].get(str(path)) == sha, "comparison consumed-run edge differs")
    return dict(root=root, seal=seal, plan=plan, summary=summary, new=new, original=original)


def selected_candidates(context, frames, evidence):
    """Read the current full-horizon candidates; retain only the three marked rows."""
    selected, references = {}, {}
    for arm in ("original", "fast_tail"):
        if arm == "original":
            candidate, ref = comparison.previous.load_candidate(
                context["original"]["bundle"], "candidate.npz", evidence, STOP
            )
        else:
            candidate, ref = comparison.load_milestone_candidate(context["new"], STOP, evidence)
        require(
            ref == context["plan"]["candidate_references"][arm]
            and context["seal"]["inputs"].get(ref["path"]) == ref["sha256"],
            "candidate differs from the completed same-horizon comparison",
        )
        before = comparison.previous.signature(candidate)
        selected[arm] = legacy.anchor_arrays(candidate, frames)
        require(comparison.previous.signature(candidate) == before, "candidate was modified")
        references[arm] = ref
        del candidate
    return selected, references


def scope_record(anchor_frames, clip_range):
    frames = np.asarray(anchor_frames)
    require(
        frames.shape == (3,)
        and frames.dtype.kind in "iu"
        and np.all(np.diff(frames.astype(np.int64)) > 0)
        and frames[0] >= FIRST
        and frames[-1] < STOP,
        "three ordered anchors inside saved full history required",
    )
    clip = review.range_pair(clip_range, "review clip")
    require(clip[0] <= frames[0] and frames[-1] < clip[1], "anchors outside displayed clip")
    inside = (frames >= START) & (frames < STOP)
    intersection = [max(START, clip[0]), min(STOP, clip[1])]
    return dict(
        scientific_interval=[START, STOP],
        saved_history_interval=[FIRST, STOP],
        raw_review_clip=clip,
        actual_anchor_frames=frames.tolist(),
        anchors_inside_scientific_interval=inside.tolist(),
        scientific_anchor_times=int(inside.sum()),
        requested_scientific_fish_endpoints=3 * int(inside.sum()),
        displayed_clip_intersection=intersection if intersection[0] < intersection[1] else None,
        interval_interior_certified=False,
        independently_verified_dense_frames=0,
        pre_start_anchors_counted_as_five_minute_truth=False,
    )


def register_document(document_evidence, review_path, evidence):
    explicit = Path(review_path).expanduser().resolve(strict=True)
    external = {}
    for path, digest in document_evidence.hashes.items():
        path = Path(path).resolve(strict=True)
        if path.is_relative_to(REPO):
            evidence.add(path, digest)
        else:
            require(path == explicit, "only the explicitly provided human JSON may be external")
            external[str(path)] = digest
    return external


def score_rows(parsed, rows, rig, points, candidate_sha, calibration, projection_sha):
    before = {k: review.array_sha256(v) for k, v in rows.items()}
    mapped = review.match_anchor_rows(
        parsed,
        rows,
        rig,
        points,
        candidate_sha256=candidate_sha,
        calibration_sha256=calibration,
        projection_origin_sha256=projection_sha,
    )
    require(before == {k: review.array_sha256(v) for k, v in rows.items()}, "anchor rows changed")
    require(
        mapped["original_review_flags_preserved"] is True
        and mapped["dense_trajectory_modified"] is False
        and mapped["global_identity_assigned"] is False
        and all(r["interior_identity_certified"] is False for r in mapped["links"]),
        "local endpoint mapping must not promote interval or identity truth",
    )
    return mapped


def serialize_identity_result(result, out):
    """Keep exact mapping/gap arrays, including +inf, in a separate NPZ."""
    require(type(result.get("available")) is bool, "explicit identity availability required")
    document = copy.deepcopy(result)
    if result["available"]:
        arrays = {"permutation_by_id": result["permutation_by_id"]}
        arrays.update({"ambiguity__" + key: value for key, value in result["ambiguity"].items()})
        require(
            all(isinstance(v, np.ndarray) and v.dtype.kind != "O" for v in arrays.values()),
            "numeric/string identity arrays required",
        )
        signatures = {key: proof.signature(value) for key, value in arrays.items()}
        record = out.write("identity_anchor_arrays.npz", arrays)
        with np.load(record["path"], allow_pickle=False) as saved:
            require(
                set(saved.files) == set(arrays)
                and all(proof.signature(saved[k]) == signatures[k] for k in arrays),
                "saved identity anchor arrays differ",
            )
        require(
            all(proof.signature(arrays[k]) == signatures[k] for k in arrays),
            "identity serialization mutated input arrays",
        )
        document["anchor_arrays_archive"] = record
        document["permutation_by_id"] = dict(
            archive_member="permutation_by_id", **signatures["permutation_by_id"]
        )
        document["ambiguity"] = {
            key: dict(archive_member="ambiguity__" + key, **signatures["ambiguity__" + key])
            for key in result["ambiguity"]
        }
    else:
        require(
            result.get("permutation_by_id") is None and result.get("ambiguity") is None,
            "unavailable identity must not contain a partial permutation",
        )
        document["anchor_arrays_archive"] = None
    out.write("identity_binding.json", document)
    return document


def describe_slot_space(arm, candidate_reference, identity_document=None):
    """Describe every legacy matcher 'slot' integer in this arm explicitly."""
    automatic = arm == "fast_tail_automatic_ID"
    require(arm in ("original", "fast_tail", "fast_tail_automatic_ID"), "unknown mapping arm")
    result = dict(
        arm=arm,
        kind="automatic_initial_slot_ID" if automatic else "raw_candidate_slot",
        legacy_slot_fields_are_raw_candidate_slots=not automatic,
        applies_to="all slot fields including raw_match.slot and left_raw_slot/right_raw_slot",
        raw_candidate_reference=candidate_reference,
        biological_identity_assigned=False,
    )
    if automatic:
        require(
            identity_document is not None and identity_document["available"] is True,
            "automatic slot meaning requires the saved identity mapping",
        )
        result.update(
            identity_provenance_sha256=identity_document["binding"]["identity_provenance_sha256"],
            identity_anchor_arrays=identity_document["anchor_arrays_archive"],
            permutation_member="permutation_by_id",
            relation="raw_slot = permutation_by_id[anchor_index, automatic_ID]",
        )
    return result


def run(args):
    require(EXECUTION_READY and args.execute is True, "reviewed execution required")
    require((args.identity_root is None) == (args.identity_sha is None), "identity root/SHA pair")
    destination = args.output.resolve()
    require(destination.is_relative_to(WORK) and not destination.exists(), "fresh workspace output")
    code = source_code()
    out, evidence = proof.Output(destination), comparison.producer.original.phase.Digests()
    started = proof.utc()
    document_evidence, external = None, {}
    try:
        require(not comparison.producer.torch.cuda.is_initialized(), "CPU-only review mapping")
        for path, digest in code.items():
            evidence.add(path, digest)
            out.write(
                "code_snapshot/" + Path(path).relative_to(REPO).as_posix(), Path(path).read_bytes()
            )
        context = load_comparison(args.comparison_root, args.comparison_sha, evidence)
        parsed, pack_summary, document_evidence = review.load_document(
            args.review_pack, args.review
        )
        external = register_document(document_evidence, args.review, evidence)
        pack = review.local(args.review_pack)
        scope = scope_record(parsed.anchor_frames, parsed.review["review_clip_range"])
        source = source_binding.verify_source_binding(
            comparison.BRANCH,
            context["plan"]["final_tracking_sha256"],
            parsed,
            pack_summary,
            pack,
            evidence,
        )
        require(
            source["shared"] == context["plan"]["shared_SL_calibration_sha256"],
            "review and comparison SL/calibration inputs differ",
        )
        selected, references = selected_candidates(context, parsed.anchor_frames, evidence)
        require(
            references["fast_tail"]["sha256"] == source["proof"]["milestone_candidate_sha256"],
            "review raw source linked to a different five-minute milestone",
        )
        rig, points, calibration = legacy.load_anchor_geometry(
            source["shared"], context["new"]["metadata"]["context"], parsed.anchor_frames, evidence
        )
        native_images = legacy.decode_anchor_images(
            source["videos"], source["manifest"], parsed.anchor_frames
        )
        projection = Path(review.evaluate_review.__file__).resolve()
        projection_sha = evidence.add(projection)
        mappings = {}
        for arm, rows in selected.items():
            mappings[arm] = score_rows(
                parsed, rows, rig, points, references[arm]["sha256"], calibration, projection_sha
            )
            mappings[arm]["slot_index_space"] = describe_slot_space(arm, references[arm])
            out.write(f"anchor_rows/{arm}.npz", rows)
            out.write(f"mapping/{arm}.json", mappings[arm])
        identity = dict(available=False, reason="identity_result_not_supplied")
        if args.identity_root is not None:
            identity = identity_binding.load_identity_anchors(
                args.identity_root,
                args.identity_sha,
                comparison_root=args.comparison_root,
                comparison_sha=args.comparison_sha,
                milestone_sha=context["plan"]["milestone_sha256"],
                raw_candidate_reference=references["fast_tail"],
                anchor_frames=parsed.anchor_frames,
                evidence=evidence,
            )
        identity_document = serialize_identity_result(identity, out)
        if identity["available"]:
            rows = identity_binding.reorder_anchor_rows(
                selected["fast_tail"], identity["permutation_by_id"]
            )
            mappings["fast_tail_automatic_ID"] = score_rows(
                parsed,
                rows,
                rig,
                points,
                references["fast_tail"]["sha256"],
                calibration,
                projection_sha,
            )
            mappings["fast_tail_automatic_ID"]["slot_index_space"] = describe_slot_space(
                "fast_tail_automatic_ID", references["fast_tail"], identity_document
            )
            out.write("anchor_rows/fast_tail_automatic_ID.npz", rows)
            out.write("mapping/fast_tail_automatic_ID.json", mappings["fast_tail_automatic_ID"])
        out.write("original_review.json", parsed.review_bytes)
        out.write("original_source_manifest.json", parsed.manifest_bytes)
        out.write("native_anchor_images.json", native_images)
        out.write("source_binding.json", source["proof"])
        out.write("displayed_anchor_files.json", source["displayed"])
        out.write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                scope=scope,
                comparison_sha256=args.comparison_sha,
                final_tracking_sha256=context["plan"]["final_tracking_sha256"],
                milestone_sha256=context["plan"]["milestone_sha256"],
                candidate_references=references,
                review_pack=str(pack),
                original_review_path=str(Path(args.review).expanduser().resolve()),
                original_review_sha256=hashlib.sha256(parsed.review_bytes).hexdigest(),
                human_status_preserved=parsed.review["status"],
                biological_naming_preserved=copy.deepcopy(parsed.review.get("biological_naming")),
                arms=list(mappings),
                automatic_identity_mapping_included="fast_tail_automatic_ID" in mappings,
                identity_result_sha256=args.identity_sha,
                native_anchor_BGR_images_redecoded=len(native_images),
                requested_fish_endpoints_per_arm=9,
                requested_links_per_arm=6,
                staged_endpoint_links_by_arm={
                    arm: sum(r["status"] == "staged_endpoint_link" for r in m["links"])
                    for arm, m in mappings.items()
                },
                interval_interior_certified=False,
                independently_verified_dense_frames=0,
                all_three_ID_truth=False,
                biological_swap_zero_verified=False,
                human_labels_created=False,
                human_annotations_modified=False,
                original_review_flags_preserved=True,
                raw_trajectory_modified=False,
                production_correction_applied=False,
                GPU_used=False,
            ),
        )
        require(
            not comparison.producer.torch.cuda.is_initialized(), "review mapping initialized CUDA"
        )
        document_evidence.final_guard()
        require(source_code() == code, "mapping source closure changed")
        evidence.final_guard()
        out.final_guard()
        seal = dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=proof.utc(),
            inputs={**evidence.paths, **external},
            outputs=out.records,
        )
    except BaseException as error:
        seal = dict(
            stage=STAGE,
            status="failed",
            started_at=started,
            finished_at=proof.utc(),
            inputs={**evidence.paths, **external},
            outputs=out.records,
            error=repr(error),
        )
        with (destination / "mapping.provenance.json").open("xb") as stream:
            stream.write(proof.json_bytes(seal))
        raise
    with (destination / "mapping.provenance.json").open("xb") as stream:
        stream.write(proof.json_bytes(seal))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--comparison-sha", required=True)
    parser.add_argument("--review-pack", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--identity-root", type=Path)
    parser.add_argument("--identity-sha")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
