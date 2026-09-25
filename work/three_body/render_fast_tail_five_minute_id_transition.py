"""Six post-hoc OldFast5 slot transitions, without changing any geometry.

CPU rendering only, after source review: 18 requested native BGR frames and 18
mask stacks; three packed archives captured once. Native pixel SHA, consumed
chunk24, SL/rig, final116200 candidate and beam mapping have separate bindings.
Rows are observations, raw slots, and initial-slot motion IDs. Sex/physical IDs
are unknown; static images do not validate continuity or count physical swaps.
"""

from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path

import numpy as np
import render_expanded_depth_10s_review as visual
import run_fast_tail_five_minute_identity_beam as ids

proof, require, prior = visual.proof, visual.require, visual.prior
previous, observations = visual.previous, visual.observations
WORK, REPO, VIEWS, KEYS = visual.WORK, visual.REPO, visual.VIEWS, visual.KEYS
STAGE = "fast_tail_five_minute_id_transition_raw_review_v1"
FRAMES = (90141, 90147, 90258, 90259, 90269, 90290)
FIRST, START, STOP, CHUNK = 49800, 66400, 116200, 1660
ID_ROOT = WORK / ids.STAGE
ID_SEAL_SHA = "4f2c676bbaddace1aa5d4a0696767e5ecdd042787dbbb7208b20d3d70403d2db"
MAPPING_SHA = "cd7885ed4be7f869cbc5738ac08e03b40fea7169811ba1443eac4bdc087573bc"
ID_SOURCE_SHA = "9bf62d84cf2eff9cb2dc19ed2dd6beab901633b8221ca6491ab5f0b2bc3a55e1"
VISUAL_SOURCE_SHA = "36eea92fa9126a65495b055a0f9046d88de31382d36535ba45478651b5933e25"
MAPPING_KEYS = (
    "frame_idx",
    "permutation_by_id",
    "conditional_state_alternative_status",
    "conditional_identity_alternative_status",
    "conditional_state_alternative_gap",
    "conditional_identity_alternative_gap",
)
EXECUTION_READY = True


def source_code():
    code = ids.source_code()
    for module, digest in ((ids, ID_SOURCE_SHA), (visual, VISUAL_SOURCE_SHA)):
        require(code.get(str(Path(module.__file__).resolve())) == digest, "frozen helper changed")
    for name, digest in visual.HELPER_PINS.items():
        require(code.get(str(WORK / name)) == digest, "frozen observation helper changed")
    require(str(Path(__file__).resolve()) in code, "renderer absent from source closure")
    return code


def select_chunk(index):
    require(set(index) == {"original_binding", "prefix_chunks"}, "actual index schema differs")
    chunks = index["prefix_chunks"]
    require(len(chunks) == (STOP - FIRST) // CHUNK == 40, "complete forty chunks required")
    for i, row in enumerate(chunks):
        require(
            all(type(row[k]) is int for k in ("index", "start", "stop"))
            and (row["index"], row["start"], row["stop"])
            == (i, FIRST + i * CHUNK, FIRST + (i + 1) * CHUNK),
            "consumed chunk native order differs",
        )
    selected = chunks[(FRAMES[0] - FIRST) // CHUNK]
    require(
        selected["index"] == 24
        and selected["start"] == 89640
        and selected["stop"] == 91300
        and all(selected["start"] <= f < selected["stop"] for f in FRAMES),
        "all six targets must belong to chunk24",
    )
    return selected


def validate_identity(seal, plan, summary, binding):
    require(seal["stage"] == ids.STAGE and seal["status"] == "ok", "successful ID seal required")
    ids.edge(seal, Path(ids.__file__), ID_SOURCE_SHA)
    ids.edge(seal, ids.COMPARISON_ROOT / "comparison.provenance.json", ids.COMPARISON_SEAL_SHA)
    for doc in (plan, summary):
        require(
            doc["schema"] == ids.STAGE
            and doc["native_interval"] == [START, STOP]
            and doc["native_frames"] == STOP - START
            and doc["beam_width"] == 256
            and doc["physical_ID_truth"] is False
            and doc["automatic_adoption"] is False
            and doc["gaps_conditional_on_retained_graph"] is True
            and doc["gaps_are_probabilities"] is False
            and doc["global_optimality_certified"] is False,
            "ID scope or truth contract differs",
        )
        require(all(doc[k] == v for k, v in binding.items()), "ID and candidate binding differ")
        require(
            doc["raw21_signatures"] == binding["comparison_candidate_reference"]["arrays"],
            "ID consumed another candidate",
        )
    require(
        plan["initial_P"] == [0, 1, 2]
        and plan["node_indices"] == list(ids.NODES)
        and plan["dp_config"] == ids.CONFIG
        and plan["source_sha256"].get(str(Path(ids.__file__).resolve())) == ID_SOURCE_SHA
        and summary["status"] == "ok"
        and summary["terminal_horizon"] == STOP
        and summary["solve_calls"] == 1
        and summary["original_raw21_unchanged"] is True
        and summary["raw_sources_review_unchanged"] is True
        and summary["review_flags_cleared"] is False
        and summary["intermediate_ID_resets"] == 0
        and summary["automatic_retry"] is False
        and summary["cap_increase"] is False,
        "ID initial convention or immutable lifecycle differs",
    )


def selected_rows(arrays, expected_frames):
    require(np.array_equal(arrays["frame_idx"], expected_frames), "native horizon/order differs")
    positions = np.searchsorted(arrays["frame_idx"], FRAMES)
    require(np.array_equal(arrays["frame_idx"][positions], FRAMES), "six targets absent")
    rows = {k: a[positions].copy() for k, a in arrays.items()}
    for value in rows.values():
        value.flags.writeable = False
    return rows


def candidate_rows(arrays, reference):
    require(set(arrays) == set(KEYS), "five candidate fields required")
    require(
        all(proof.signature(a) == reference["arrays"][k] for k, a in arrays.items()),
        "candidate numeric signatures differ",
    )
    n = STOP - FIRST
    require(
        arrays["pose_xyz"].shape == arrays["sources"].shape == (n, 3, 3, 3)
        and arrays["observed"].shape == (n, 3, 3)
        and arrays["identity_review"].shape == (n, 3),
        "candidate body shape differs",
    )
    return selected_rows(arrays, np.arange(FIRST, STOP))


def mapping_rows(arrays):
    require(set(arrays) == set(MAPPING_KEYS), "six mapping fields required")
    n, p = STOP - START, arrays["permutation_by_id"]
    require(
        p.shape == (n, 3)
        and p.dtype.kind in "iu"
        and np.array_equal(np.sort(p, axis=1), np.broadcast_to([0, 1, 2], (n, 3)))
        and np.array_equal(p[0], [0, 1, 2]),
        "ID-to-slot permutation or initial convention differs",
    )
    for prefix in ("conditional_state_alternative", "conditional_identity_alternative"):
        shape = (n,) if "state" in prefix else (n, 3)
        require(
            all(arrays[prefix + suffix].shape == shape for suffix in ("_status", "_gap")),
            "conditional mapping annotation shape differs",
        )
    return selected_rows(arrays, np.arange(START, STOP))


def load_poses_and_mapping(binding, evidence):
    seal = previous.read(ID_ROOT / "identity.provenance.json", evidence, ID_SEAL_SHA)
    plan, summary = [
        ids.document(ID_ROOT, seal, name, evidence) for name in ("plan.json", "summary.json")
    ]
    validate_identity(seal, plan, summary, binding)
    name = "identity_mapping_and_ambiguity.npz"
    require(seal["outputs"][name]["sha256"] == MAPPING_SHA, "fixed mapping archive differs")
    path = previous.member(ID_ROOT, seal, name, evidence)
    with np.load(path, allow_pickle=False) as archive:
        require(
            len(archive.files) == len(set(archive.files)) == len(ids.frozen.MAPPING_FIELDS)
            and set(archive.files) == set(ids.frozen.MAPPING_FIELDS),
            "mapping membership differs",
        )
        mapping = mapping_rows({k: archive[k] for k in MAPPING_KEYS})
    # This completed comparison references its milestone directly, without a copy.
    for row in binding["candidate_archive_references"].values():
        path = Path(row["path"])
        require(path.stat().st_size == row["size_bytes"], "candidate resized")
        ids.edge(seal, path, row["sha256"])
        evidence.add(path, row["sha256"])
    path = binding["candidate_archive_references"]["milestone"]["path"]
    reference = binding["comparison_candidate_reference"]
    with np.load(path, allow_pickle=False) as archive:
        require(
            len(archive.files) == len(set(archive.files)) == 21
            and set(archive.files) == set(reference["arrays"]),
            "candidate all21 membership differs",
        )
        poses = candidate_rows({k: archive[k] for k in KEYS}, reference)
    return poses, mapping, seal["outputs"][name]


def capture_masks(path, digest, frames):
    before = observations.file_stat(path)
    payload = path.read_bytes()  # Exactly one disk payload read; final guard is stat + memory SHA.
    require(
        hashlib.sha256(payload).hexdigest() == digest and observations.file_stat(path) == before,
        "captured mask payload changed",
    )
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        require(
            len(archive.files) == len(set(archive.files)) == 13
            and set(archive.files) == set(observations.chunk_masks.SCHEMA_DTYPES),
            "packed mask membership differs",
        )
        data = {k: archive[k] for k in archive.files}
    require(
        all(data[k].dtype == dt for k, dt in observations.chunk_masks.SCHEMA_DTYPES.items()),
        "packed mask dtype differs",
    )
    require(
        np.array_equal(data["frame_idx"], frames)
        and np.array_equal(data["image_shape_hw"], [1200, 1200])
        and data["view_available"].shape == data["frame_mask_sha256"].shape == frames.shape,
        "packed native extent differs",
    )
    for value in data.values():
        value.flags.writeable = False
    return data, dict(
        path=str(path),
        sha256=digest,
        file_stat=before,
        arrays={k: proof.signature(v) for k, v in data.items()},
        payload_reads=1,
        payload_decodes=1,
    )


def canonical_video_edge(tracking, staging, binding, link, view):
    """Tracking consumes the canonical MP4; the detector produces its staged hardlink."""
    canonical = prior.raw.local(link["source_path"])
    expected = dict(path=str(canonical), sha256=link["sha256"])
    row = staging["inputs"]["video:" + view]
    require(
        dict(path=row["path"], sha256=row["sha256"])
        == binding["detector_inputs"]["video:" + view]
        == expected,
        "staging canonical detector input differs",
    )
    ids.edge(tracking, canonical, link["sha256"])


def load_observations(evidence):
    tracking = previous.read(
        ids.TRACKING_ROOT / "tracking.provenance.json", evidence, ids.TRACKING_SEAL_SHA
    )
    index = ids.document(ids.TRACKING_ROOT, tracking, "input_index.json", evidence)
    plan = ids.document(ids.TRACKING_ROOT, tracking, "plan.json", evidence)
    require(
        index["original_binding"] == plan["original_binding"],
        "producer prefix configuration differs",
    )
    chunk, binding = select_chunk(index), index["original_binding"]["inputs"]
    root = prior.native.checked_relative(
        prior.raw.local(binding["input_stage_root"]), chunk["attempt_path"]
    )
    manifest_path = prior.raw.local(chunk["output_hashes"]["manifest"]["path"])
    require(manifest_path == root / "manifest.json", "consumed manifest root differs")
    ids.edge(tracking, manifest_path, chunk["manifest_sha256"])
    manifest = previous.read(manifest_path, evidence, chunk["manifest_sha256"])
    chunk_path = root / "staging.provenance.json"
    ids.edge(tracking, chunk_path, chunk["provenance_sha256"])
    seal = previous.read(chunk_path, evidence, chunk["provenance_sha256"])
    require(
        seal["status"] == "ok" and seal["stage"] == "prepare_independent_mask_chunk",
        "successful consumed mask chunk required",
    )
    require(
        {k: dict(path=v["path"], sha256=v["sha256"]) for k, v in seal["outputs"].items()}
        == chunk["output_hashes"],
        "consumed mask output roles differ",
    )
    prior.raw.validate_source(manifest, chunk, binding["detector_binding"])
    require(
        manifest["binding"] == seal["params"]["binding"]
        and manifest["source_interval_half_open"] == [chunk["start"], chunk["stop"]]
        and set(manifest["masks"]) == set(VIEWS),
        "detector/native mask binding differs",
    )
    frames = np.arange(chunk["start"], chunk["stop"], dtype=np.int64)
    videos, packed, records = {}, {}, {}
    for view in VIEWS:
        descriptor, link = manifest["raw"][view], chunk["raw_hardlinks"][view]
        path = prior.native.checked_relative(root, descriptor["path"])
        require(
            chunk["output_hashes"][view + "_raw"]
            == dict(path=str(path), sha256=descriptor["sha256"])
            and descriptor["sha256"] == link["sha256"]
            and binding["detector_inputs"]["video:" + view]
            == dict(path=str(prior.raw.local(link["source_path"])), sha256=descriptor["sha256"]),
            "canonical raw source binding differs",
        )
        canonical_video_edge(tracking, seal, binding, link, view)
        videos[view] = path  # Never evidence.add: no full-video hashing in this renderer.
        descriptor = manifest["masks"][view]
        path = prior.native.checked_relative(root, descriptor["path"])
        require(
            chunk["output_hashes"][view + "_masks"]
            == dict(path=str(path), sha256=descriptor["sha256"]),
            "consumed mask path/SHA differs",
        )
        ids.edge(tracking, path, descriptor["sha256"])
        packed[view], records[view] = capture_masks(path, descriptor["sha256"], frames)
    previous.video_stat_guard(videos, chunk)
    seed, shared = binding["seed"], binding["seed"]["SL_and_calibration_sha256"]
    comp = previous.read(
        ids.COMPARISON_ROOT / "comparison.provenance.json", evidence, ids.COMPARISON_SEAL_SHA
    )
    comp_plan = ids.document(ids.COMPARISON_ROOT, comp, "plan.json", evidence)
    require(
        len(shared) == 7 and shared == comp_plan["shared_SL_calibration_sha256"],
        "same compared SL3/calibration4 required",
    )
    for path, digest in shared.items():
        ids.edge(tracking, Path(path), digest)
        evidence.add(Path(path), digest)
    nodes = observations.observations.CachedNodes(
        seed["input_root"], manifest["experiment"], frames
    )
    require(
        np.array_equal(nodes.frames, frames)
        and set(nodes.paths) == set(VIEWS)
        and {str(Path(p).resolve()) for p in nodes.paths.values()}
        == {p for p in shared if Path(p).suffix == ".npz"},
        "consumed SL membership differs",
    )
    paths = [Path(p) for p in shared if Path(p).name == "calibration_manifest.json"]
    require(len(paths) == 1, "one calibration manifest required")
    config = proof.read_json(paths[0])
    models = {
        str(prior.native.checked_relative(paths[0].parent, r["path"]))
        for r in config["model_files"]
    }
    require(
        len(models) == 3
        and models | {str(paths[0])} == {p for p in shared if Path(p).suffix == ".json"},
        "calibration model membership differs",
    )
    rig = observations.calib.load_rig(paths[0].parent)
    require(tuple(rig.views) == tuple(VIEWS) and rig.world_units == "mm", "rig view/unit differs")
    return dict(
        nodes=nodes,
        rig=rig,
        packed=packed,
        mask_records=records,
        chunk=chunk,
        manifest=manifest,
        videos=videos,
        shared_SL_calibration_sha256=shared,
    )


def remap_display(projected, observed, permutation):
    p = np.asarray(permutation)
    require(
        p.shape == (3,) and p.dtype.kind in "iu" and np.array_equal(np.sort(p), [0, 1, 2]),
        "ID-to-raw-slot permutation must be bijective",
    )
    require(projected.shape == (3, 3, 2) and observed.shape == (3, 3), "display body shape differs")
    xy, obs = projected[p].copy(), observed[p].copy()
    require(
        np.array_equal(xy[np.argsort(p)], projected, equal_nan=True)
        and np.array_equal(obs[np.argsort(p)], observed),
        "display changed geometry or support",
    )
    xy.flags.writeable = obs.flags.writeable = False
    return xy, obs


def draw_identity_panel(ax, image, masks, points, projected, observed, permutation):
    xy, obs = remap_display(projected, observed, permutation)
    visual.draw_panel(ax, image, masks, points, xy, obs, "raw")
    labels = {f"S{i}{n}": f"ID{i}{n}(S{int(permutation[i])})" for i in range(3) for n in "MCT"}
    for label in ax.texts:
        if label.get_text() in labels:
            label.set_text(labels[label.get_text()])


def render(frame, images, masks, points, projected, observed, permutation, statuses):
    fig, axes = prior.plt.subplots(3, 3, figsize=(18, 17), dpi=160)
    fig.suptitle(
        f"Native {frame} | final horizon 116200 | post-hoc motion-ID transition diagnostic\n"
        "Rows: observations / raw slots / initial-slot motion IDs. "
        "Same geometry and crop; label/color remap only.\n"
        f"ID0/1/2 -> raw slots {list(map(int, permutation))}; "
        f"conditional status {list(map(str, statuses))}.\n"
        "Sex/physical IDs unknown. "
        "Static images do not validate continuity or count physical swaps.\n"
        "M circle / C square / T triangle: filled direct, open latent. "
        "D2# and SL rows are untracked.",
        fontsize=10,
    )
    crops = {}
    try:
        for vi, view in enumerate(VIEWS):
            lo, hi = visual.display_crop(images[view], masks[view], points[view], [projected[view]])
            crops[view] = dict(lower_xy=lo.tolist(), upper_xy=hi.tolist())
            for ri, title in enumerate(
                (
                    "Independent observations",
                    "Raw slots S0/S1/S2",
                    "Motion IDs, initial-slot convention",
                )
            ):
                ax = axes[ri, vi]
                if ri == 2:
                    draw_identity_panel(
                        ax,
                        images[view],
                        masks[view],
                        points[view],
                        projected[view],
                        observed,
                        permutation,
                    )
                else:
                    visual.draw_panel(
                        ax,
                        images[view],
                        masks[view],
                        points[view],
                        projected[view] if ri else None,
                        observed if ri else None,
                        "raw",
                    )
                ax.set_xlim(float(lo[0]), float(hi[0]))
                ax.set_ylim(float(hi[1]), float(lo[1]))
                ax.set_title(f"{view} | {title}", fontsize=9)
                ax.tick_params(labelsize=7)
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        stream = io.BytesIO()
        fig.savefig(stream, format="png")
        return stream.getvalue(), crops
    finally:
        prior.plt.close(fig)


def native_images(context, frame):
    iterator = prior.raw.iter_native(context["videos"], context["manifest"], frame, frame + 1)
    try:
        native, images = next(iterator)
        require(
            native == frame and next(iterator, None) is None,
            "exactly one requested native required",
        )
        require(set(images) == set(VIEWS), "three raw views required")
        return images
    finally:
        iterator.close()


def run(output):
    require(EXECUTION_READY, "source review and ROOT render Go required")
    ids.cpu_only()
    output = output.resolve()
    require(output.is_relative_to(WORK) and not output.exists(), "fresh workspace output required")
    out, evidence, started = proof.Output(output), proof.Evidence(), proof.utc()
    try:
        code = source_code()
        for path, digest in code.items():
            evidence.add(Path(path), digest)
        binding = ids.comparison_headers(ids.COMPARISON_SEAL_SHA, evidence)
        rows, mapping, mapping_record = load_poses_and_mapping(binding, evidence)
        context = load_observations(evidence)
        before = {"poses": ids.signature(rows), "mapping": ids.signature(mapping)}
        nodes = context["nodes"]
        node_before = {v: [proof.signature(a) for a in pair] for v, pair in nodes.data.items()}
        scope = dict(
            schema=STAGE,
            frames=list(FRAMES),
            terminal_horizon=STOP,
            source_chunk_index=24,
            comparison_sha256=ids.COMPARISON_SEAL_SHA,
            identity_sha256=ID_SEAL_SHA,
            source_code=code,
            candidate_binding=binding,
            mapping_record=mapping_record,
            candidate_decoded_fields=list(KEYS),
            mapping_decoded_fields=list(MAPPING_KEYS),
            candidate_all21_membership_checked=True,
            candidate_all21_decoded=False,
            requested_BGR_frames=18,
            target_mask_decodes=18,
            packed_mask_payload_reads=3,
            images=6,
            same_geometry_labels_and_colors_only=True,
            raw_slot_labels_are_physical_IDs=False,
            sex_confirmed=False,
            physical_ID_truth=False,
            automatic_adoption=False,
            selected_frames_are_holdout=False,
            whole_video_rehashes=0,
            checkpoint_or_DAG_reads=0,
            tracking_or_solver_calls=0,
            GPU_used=False,
            review_modified=False,
            limitation=(
                "Six post-hoc static samples; no physical swap count or continuity validation."
            ),
        )
        out.write("plan.json", scope)
        records = []
        for t, frame in enumerate(FRAMES):
            images = native_images(context, frame)
            masks, points, observed_rows = {}, {}, {}
            for view in VIEWS:
                pos, packed = frame - context["chunk"]["start"], context["packed"][view]
                require(int(packed["frame_idx"][pos]) == frame, "native mask row differs")
                decoded = prior.prepare_sequence.decode_frame_masks(packed, pos)
                digest = prior.prepare_sequence.array_sha256(decoded)
                require(
                    digest == bytes(packed["frame_mask_sha256"][pos]).decode(),
                    "target mask pixel SHA differs",
                )
                available = bool(packed["view_available"][pos])
                masks[view] = decoded if available else None
                points[view] = nodes.at(view, frame)
                observed_rows[view] = dict(
                    bgr_sha256=prior.prepare_sequence.array_sha256(images[view]),
                    mask_sha256=digest,
                    mask_count=len(decoded),
                    mask_available=available,
                    SL_points_signature=proof.signature(points[view]),
                    SLEAP_points=prior.finite_json(points[view]),
                )
            projected = {v: context["rig"].camera(v).project(rows["pose_xyz"][t]) for v in VIEWS}
            p = mapping["permutation_by_id"][t]
            png, crops = render(
                frame,
                images,
                masks,
                points,
                projected,
                rows["observed"][t],
                p,
                mapping["conditional_identity_alternative_status"][t],
            )
            out.write(f"frame_{frame}.png", png)
            records.append(
                dict(
                    frame=frame,
                    observations=observed_rows,
                    display_crops=crops,
                    raw_candidate={k: prior.finite_json(v[t]) for k, v in rows.items()},
                    mapping={k: ids.explicit_json(v[t]) for k, v in mapping.items()},
                    raw_projections={v: prior.finite_json(x) for v, x in projected.items()},
                    ID_projections={
                        v: prior.finite_json(remap_display(x, rows["observed"][t], p)[0])
                        for v, x in projected.items()
                    },
                    geometry_changed=False,
                )
            )
            print(
                f"Rendered native {frame}; requested BGR/mask frames {(t + 1) * 3}/18", flush=True
            )
        require(
            before == {"poses": ids.signature(rows), "mapping": ids.signature(mapping)},
            "candidate/mapping rows changed",
        )
        require(
            node_before
            == {v: [proof.signature(a) for a in pair] for v, pair in nodes.data.items()},
            "SL changed",
        )
        for view, record in context["mask_records"].items():
            require(
                observations.file_stat(Path(record["path"])) == record["file_stat"]
                and {k: proof.signature(v) for k, v in context["packed"][view].items()}
                == record["arrays"],
                "captured mask file/stat/sidecars changed",
            )
        previous.video_stat_guard(context["videos"], context["chunk"])
        out.write(
            "source_and_frame_evidence.json",
            dict(
                **scope,
                status="ok",
                selected_arrays=before,
                source_chunk=context["chunk"],
                captured_mask_records=context["mask_records"],
                shared_SL_calibration_sha256=context["shared_SL_calibration_sha256"],
                records=records,
                native_pixel_SHA_checked=True,
                candidate_modified=False,
                mapping_modified=False,
            ),
        )
        ids.cpu_only()
        require(source_code() == code, "source closure changed")
        evidence.final_guard()
        out.final_guard()
        seal = dict(stage=STAGE, status="ok")
    except BaseException as error:
        seal = dict(stage=STAGE, status="failed", error=repr(error))
        seal.update(
            started_at=started, finished_at=proof.utc(), inputs=evidence.paths, outputs=out.records
        )
        (output / "render.provenance.json").write_bytes(proof.json_bytes(seal))
        raise
    seal.update(
        started_at=started, finished_at=proof.utc(), inputs=evidence.paths, outputs=out.records
    )
    (output / "render.provenance.json").write_bytes(proof.json_bytes(seal))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
