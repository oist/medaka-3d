"""Disabled import of one explicit human JSON for the raw-only 360-frame pack.

Checks the new size_bytes / SHA-string pack schema, all encoded output files,
and small original-source metadata. Never opens original videos, mask/SL/pose
arrays or any solver. Only the explicit review JSON may be outside the workspace.
Original answer bytes and uncertain human states are preserved without matching
to trajectories or certifying physical identities beyond the reviewed clip.
"""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import import_long_review as frozen
import prepare_fast_tail_transition_raw_review as creator

proof, require = creator.proof, creator.require
WORK, REPO, VIEWS = creator.WORK, creator.REPO, creator.VIEWS
STAGE = "fast_tail_transition_review_document_import_v1"
PACK_ROOT = WORK / "fast_tail_transition_raw_review_pack_v1"
CREATOR_SHA = "db6cabad5df80d391f7068b5a0c37aa8f357e7bbd5281e9488c988133f17ea9c"
FROZEN_SHA = "4ee9a2af648231658714c017ff3d48232700f09a701092f7a70a75869421bcc5"
START, STOP, PROPOSALS = 90040, 90400, (90120, 90220, 90380)
EXECUTION_READY = True
ParsedReview = frozen.ParsedReview
parse_review = frozen.parse_review
read_explicit_review = frozen.read_explicit_review


def source_code():
    code = creator.source_code()
    for module, digest in ((creator, CREATOR_SHA), (frozen, FROZEN_SHA)):
        require(code.get(str(Path(module.__file__).resolve())) == digest, "frozen source differs")
    require(str(Path(__file__).resolve()) in code, "new importer absent from source closure")
    return code


def expected_outputs():
    return {
        *(f"frames/f{frame}_{view}.jpg" for frame in range(START, STOP) for view in VIEWS),
        *(f"f{frame}_{view}.png" for frame in PROPOSALS for view in VIEWS),
        "source_manifest.json",
        "index.html",
        "truth_template.json",
        "anchor_selection.json",
        "summary.json",
        "raw_three_views_slow.mp4",
        "raw_three_views_mpeg4_intermediate.mp4",
    }


def output_files(root, seal, evidence):
    """Hashes only the encoded review pack, never its external source-video paths."""
    require(set(seal["outputs"]) == expected_outputs(), "all1096 pack output roles required")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    unreferenced = sorted(actual - expected_outputs() - {"review.provenance.json"})
    for name, row in seal["outputs"].items():
        path = (root / name).resolve()
        require(
            path.is_relative_to(root)
            and not (root / name).is_symlink()
            and Path(row["path"]).resolve() == path
            and type(row["size_bytes"]) is int
            and path.stat().st_size == row["size_bytes"],
            "pack output path/size differs",
        )
        evidence.add(path, row["sha256"])
    # A human may save their answer beside the sealed review files. Such additions
    # are named here, never consumed as pack evidence or silently selected as answers.
    return unreferenced


def exact(value, expected, label):
    require(
        all(type(value.get(k)) is type(v) and value[k] == v for k, v in expected.items()), label
    )


def validate_documents(summary, manifest, template, anchor_selection, manifest_sha):
    frames = list(range(START, STOP))
    exact(
        summary,
        dict(
            schema=creator.STAGE,
            status="ok",
            technical_completion=True,
            review_clip_range=[START, STOP],
            source_frame_range=[89640, 91300],
            displayed_frames=frames,
            probe_frames=list(PROPOSALS),
            frame_count=360,
            source_fps=166,
            playback_fps=20,
            source_duration_seconds=360 / 166,
            playback_duration_seconds=18.0,
            source_absolute_seconds_range=[START / 166, STOP / 166],
            source_clip_time_rule="(native_frame - 90040) / 166; source time is native_frame / 166",
            absolute_seek_frame=START,
            first_manifest_sha_offset=400,
            original_source_video_frame_count=348600,
            native_pixel_hashes_verified=1080,
            original_raw_storage="original_source_video",
            source_manifest_sha256=manifest_sha,
            raw_video_payload_bytes_copied=0,
            whole_video_rehashes=0,
            model_outputs_shown=False,
            candidate_arrays_deserialized=False,
            mapping_arrays_deserialized=False,
            masks_or_sleap_arrays_deserialized=False,
            human_truth_created=False,
            source_selection_is_model_derived=True,
            gpu_used=False,
            tracking_or_ID_solver_calls=0,
            selection_render_sha256=creator.SELECTION_SHA,
            final_tracking_sha256=creator.ids.TRACKING_SEAL_SHA,
        ),
        "raw pack scope/timebase differs",
    )
    for name in ("review_video", "intermediate_video"):
        exact(
            summary[name],
            dict(
                decoded_frame_count=360,
                playback_fps=20.0,
                image_shape_hw=[636, 1800],
                all_selected_native_frames_included=True,
            ),
            "encoded video coverage/timebase differs",
        )
    chunk = summary["consumed_record"]
    require(
        (chunk["index"], chunk["start"], chunk["stop"]) == (24, 89640, 91300)
        and chunk["manifest_sha256"] == manifest_sha
        and manifest["frames"] == list(range(89640, 91300)),
        "chunk24 source extent differs",
    )
    require(set(summary["native_bgr_sha256"]) == set(VIEWS), "native view membership")
    for view in VIEWS:
        require(
            summary["native_bgr_sha256"][view]
            == manifest["raw"][view]["frame_bgr_sha256"][400:760],
            "1080 source/native pixel SHA sequence differs",
        )
    expected, selected, offset = creator.ui.bind_template(
        manifest["frames"], manifest_sha, START, STOP, list(PROPOSALS)
    )
    require(
        template == expected and selected == frames and offset == 400,
        "original blank template differs; never substitute an answer",
    )
    exact(
        anchor_selection,
        dict(
            schema=creator.STAGE + "_anchor_selection",
            proposed_anchor_frames=list(PROPOSALS),
            review_clip_range=[START, STOP],
            proposed_not_human_confirmed=True,
            actual_clip_view_suitability_checked=False,
            anchor_adjustments=[],
            anonymous_names=["A", "B", "C"],
            physical_identity_labels_created=False,
        ),
        "original proposed-anchor history differs",
    )


def read_small(path, evidence, expected):
    path = Path(path).resolve()
    require(path.is_relative_to(WORK) and path.suffix == ".json", "small workspace JSON required")
    evidence.add(path, expected)
    return frozen.document(path.read_bytes(), str(path))


def metadata_video_path(value, root=None):
    """Lexical path normalization only; never resolve/stat an original video."""
    path = Path(value)
    require(".." not in path.parts, "video metadata path traversal")
    if root is None:
        require(path.is_absolute(), "canonical video metadata must be absolute")
    else:
        require(not path.is_absolute() and not path.drive, "staged video metadata must be relative")
        path = Path(root) / path
    return Path(os.path.abspath(path))


def validate_source_chain(pack_seal, summary, manifest, evidence):
    """Original-source metadata only; no source video stat/hash/decode or model data."""
    chunk = summary["consumed_record"]
    path = Path(summary["source_manifest"]).resolve()
    manifest_sha = summary["source_manifest_sha256"]
    creator.ids.edge(pack_seal, path, manifest_sha)
    source_manifest = read_small(path, evidence, manifest_sha)
    require(source_manifest == manifest, "source manifest copy differs")
    staging_path = path.parent / "staging.provenance.json"
    creator.ids.edge(pack_seal, staging_path, chunk["provenance_sha256"])
    staging = read_small(staging_path, evidence, chunk["provenance_sha256"])
    exact(staging, dict(stage="prepare_independent_mask_chunk", status="ok"), "staging failed")
    require(
        {k: dict(path=v["path"], sha256=v["sha256"]) for k, v in staging["outputs"].items()}
        == chunk["output_hashes"],
        "recorded staging output roles differ",
    )
    require(
        chunk["output_hashes"]["manifest"] == dict(path=str(path), sha256=manifest_sha),
        "manifest role differs",
    )
    creator.raw.validate_source(manifest, chunk, staging["params"]["binding"])
    tracking_path = creator.ids.TRACKING_ROOT / "tracking.provenance.json"
    creator.ids.edge(pack_seal, tracking_path, creator.ids.TRACKING_SEAL_SHA)
    tracking = read_small(tracking_path, evidence, creator.ids.TRACKING_SEAL_SHA)
    exact(tracking, dict(stage=creator.ids.comparison.runner.STAGE, status="ok"), "tracking failed")
    creator.ids.edge(tracking, path, manifest_sha)
    creator.ids.edge(tracking, staging_path, chunk["provenance_sha256"])
    creator.ids.edge(
        pack_seal, creator.SELECTION_ROOT / "render.provenance.json", creator.SELECTION_SHA
    )
    video_bindings = {}
    for view in VIEWS:
        row, link = manifest["raw"][view], chunk["raw_hardlinks"][view]
        staged = metadata_video_path(row["path"], path.parent)
        canonical = metadata_video_path(link["source_path"])
        require(
            metadata_video_path(link["path"]) == staged
            and row["sha256"] == link["sha256"]
            and chunk["output_hashes"][view + "_raw"]
            == dict(path=str(staged), sha256=row["sha256"]),
            "staged hardlink role differs",
        )
        source_row = staging["inputs"]["video:" + view]
        require(
            metadata_video_path(source_row["path"]) == canonical
            and source_row["sha256"] == row["sha256"],
            "canonical detector video differs",
        )
        require(
            proof.valid_sha(row["sha256"])
            and tracking["inputs"].get(str(canonical)) == row["sha256"],
            "missing consumed canonical input edge",
        )
        video_bindings[view] = dict(
            canonical_path=str(canonical), staged_path=str(staged), sha256=row["sha256"]
        )
    return video_bindings


def load_pack(pack_sha, evidence):
    require(proof.valid_sha(pack_sha), "explicit successful pack seal SHA required")
    root = PACK_ROOT.resolve()
    require(root.is_relative_to(WORK), "pack must be inside workspace")
    seal = read_small(root / "review.provenance.json", evidence, pack_sha)
    exact(seal, dict(stage=creator.STAGE, status="ok"), "successful new raw pack required")
    creator.ids.edge(seal, Path(creator.__file__), CREATOR_SHA)
    unreferenced = output_files(root, seal, evidence)
    docs = {
        name: frozen.document((root / name).read_bytes(), name)
        for name in (
            "summary.json",
            "source_manifest.json",
            "truth_template.json",
            "anchor_selection.json",
        )
    }
    summary, manifest = docs["summary.json"], docs["source_manifest.json"]
    manifest_sha = seal["outputs"]["source_manifest.json"]["sha256"]
    require(
        summary["code_sha256"].get(str(Path(creator.__file__).resolve())) == CREATOR_SHA,
        "raw creator source differs",
    )
    validate_documents(
        summary, manifest, docs["truth_template.json"], docs["anchor_selection.json"], manifest_sha
    )
    bindings = validate_source_chain(seal, summary, manifest, evidence)
    return dict(
        seal=seal,
        summary=summary,
        manifest_bytes=(root / "source_manifest.json").read_bytes(),
        template=docs["truth_template.json"],
        anchor_selection=docs["anchor_selection.json"],
        video_bindings=bindings,
        unreferenced_pack_files=unreferenced,
    )


def describe_review(parsed, proposal):
    require(type(parsed) is ParsedReview, "frozen ParsedReview required")
    endpoints, links, changes = [], [], []
    for index, anchor in enumerate(parsed.review["anchors"]):
        proposed = proposal["proposed_anchor_frames"][index]
        changes.append(
            dict(
                phase=anchor["phase"],
                proposed_frame=proposed,
                returned_frame=anchor["frame"],
                changed=anchor["frame"] != proposed,
                returned_frame_choice=anchor["frame_choice"],
            )
        )
        for name in frozen.ANIMALS:
            mark = anchor["animals"][name]
            endpoints.append(
                dict(
                    phase=anchor["phase"],
                    frame=anchor["frame"],
                    anonymous_animal=name,
                    visibility=mark["visibility"],
                    original_points=copy.deepcopy(mark["points"]),
                    model_slot=None,
                    physical_identity_certified=False,
                )
            )
    for name in frozen.ANIMALS:
        value = parsed.review["continuity"][name]
        for i, key in enumerate(frozen.EDGES):
            links.append(
                dict(
                    anonymous_animal=name,
                    edge=key,
                    start_frame=int(parsed.anchor_frames[i]),
                    stop_frame=int(parsed.anchor_frames[i + 1]),
                    human_status=value[key],
                    human_basis=value["basis"],
                    human_notes=value.get("notes", ""),
                    trajectory_constraint_created=False,
                )
            )
    require(len(endpoints) == 9 and len(links) == 6, "all nine endpoints/six links required")
    return dict(
        endpoints=endpoints,
        links=links,
        proposed_to_returned_anchor_changes=changes,
        original_anchor_proposal=copy.deepcopy(proposal),
        intermediate_anchor_edit_history_available=False,
        intermediate_anchor_edit_history_inferred=False,
        biological_naming_preserved_not_applied=copy.deepcopy(
            parsed.review.get("biological_naming")
        ),
    )


def run(pack_sha, review_path, output):
    require(EXECUTION_READY, "source-only: actual human import not enabled")
    creator.ids.cpu_only()
    output = Path(output).resolve()
    require(
        output.is_relative_to(WORK) and not output.exists(), "fresh workspace artifact required"
    )
    out, evidence, external, started = (
        proof.Output(output),
        proof.Evidence(),
        frozen.Evidence(),
        proof.utc(),
    )
    try:
        code = source_code()
        for path, value in code.items():
            evidence.add(path, value)
        pack = load_pack(pack_sha, evidence)
        require(
            Path(review_path).suffix.lower() == ".json", "one explicitly named review JSON required"
        )
        review_bytes = read_explicit_review(review_path, external)
        require(len(external.hashes) == 1, "only one explicit external JSON allowed")
        out.write("original_review.json", review_bytes)
        parsed = parse_review(review_bytes, pack["manifest_bytes"], clip_range=[START, STOP])
        details = describe_review(parsed, pack["anchor_selection"])
        out.write("source_manifest.json", pack["manifest_bytes"])
        out.write("original_anchor_selection.json", pack["anchor_selection"])
        summary = dict(
            schema=STAGE,
            status="ok",
            review_status_preserved=parsed.review["status"],
            review_clip_range=[START, STOP],
            source_frame_range=[89640, 91300],
            pack_seal_sha256=pack_sha,
            creator_source_sha256=CREATOR_SHA,
            parser_source_sha256=FROZEN_SHA,
            original_review_sha256=next(iter(external.hashes.values())),
            external_input_count=1,
            original_review_bytes_preserved=True,
            **details,
            anonymous_identity_scope="within this clip only; A/B/C are human entry choices",
            pack_outputs_verified=1096,
            unreferenced_pack_files_not_used_as_pack_evidence=pack["unreferenced_pack_files"],
            source_video_bindings=pack["video_bindings"],
            original_video_stat_or_payload_reads=0,
            SLEAP_mask_candidate_checkpoint_reads=0,
            image_decodes=0,
            GPU_used=False,
            identity_solver_calls=0,
            trajectory_modified=False,
            constraints_created=False,
            labels_filled=False,
            physical_identity_certified=False,
            all_five_minute_identity_truth=False,
            production_correction_ready=False,
            human_truth_created=False,
            review_notes_preserved=parsed.review.get("notes", ""),
            source_sha256=code,
        )
        out.write("summary.json", summary)
        require(
            (output / "original_review.json").read_bytes() == parsed.review_bytes,
            "original review bytes changed",
        )
        evidence.final_guard()
        external.final_guard()
        out.final_guard()
        require(source_code() == code, "importer source changed")
        creator.ids.cpu_only()
        status = dict(status="ok")
    except BaseException as error:
        status = dict(status="failed", error=repr(error), actual_review_import_complete=False)
        out.write("failure.json", status)
        raise
    finally:
        record = dict(
            stage=STAGE,
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            explicit_review_inputs={str(p): sha for p, sha in external.hashes.items()},
            outputs=out.records,
            **status,
        )
        with (output / "import.provenance.json").open("xb") as stream:
            stream.write(proof.json_bytes(record))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-seal-sha256", required=True)
    parser.add_argument("--review-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.pack_seal_sha256, args.review_json, args.output)
