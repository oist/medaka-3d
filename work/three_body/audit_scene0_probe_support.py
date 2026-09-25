"""Read-only geometric readiness of proposed raw-review probes, without human labels."""

import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import measure_persistent_silhouette as m
from identity_output import build_identity_output
from observations import CachedNodes

from medaka3d.calib import load_rig
from medaka3d.provenance import provenance

PROBES = (79316, 79405, 79480)
CLIP = (79236, 79488)


def inspect_state(pose, sources, observed, review, points, rig):
    """Recheck saved assignments; finite projection is separate from direct support."""
    distances = m.verify_support(pose, sources, observed, points, rig)
    projections = np.stack([rig.camera(view).project(pose[:, 0]) for view in m.VIEWS], axis=1)
    error = np.full((3, 3), np.nan)
    source_xy = np.full((3, 3, 2), np.nan)
    for v, view in enumerate(m.VIEWS):
        for fish in range(3):
            row = sources[fish, v, 0]
            if row >= 0:
                source_xy[fish, v] = points[view][row, 0]
                error[fish, v] = np.linalg.norm(projections[fish, v] - source_xy[fish, v])
    gate = (sources[:, :, 0] >= 0) & np.isfinite(error) & (error <= 12) & (distances[:, :, 0] <= 2)
    m.require(np.array_equal(gate.sum(1) >= 2, observed[:, 0]), "mouth support gate differs")
    rows = []
    for fish in range(3):
        finite = np.isfinite(projections[fish]).all(-1)
        rows.append(
            dict(
                slot=fish,
                mouth_xyz_mm=pose[fish, 0],
                mouth_xyz_finite=bool(np.isfinite(pose[fish, 0]).all()),
                mouth_observed=bool(observed[fish, 0]),
                mouth_sources_by_view=sources[fish, :, 0],
                assigned_native_mouth_xy=source_xy[fish],
                projected_native_mouth_xy=projections[fish],
                projection_finite_by_view=finite,
                projection_inside_image_by_view=finite
                & (projections[fish] >= 0).all(-1)
                & (projections[fish] < 1200).all(-1),
                assigned_ray_distance_mm_by_view=distances[fish, :, 0],
                assigned_reprojection_error_pixels_by_view=error[fish],
                direct_view_gate=gate[fish],
                direct_view_count=int(gate[fish].sum()),
                original_identity_review=bool(review[fish]),
            )
        )
    return rows, projections, gate, error


def intervals(frames):
    frames = [int(frame) for frame in frames]
    if not frames:
        return []
    out, start, previous = [], frames[0], frames[0]
    for frame in frames[1:]:
        if frame != previous + 1:
            out.append([start, previous + 1])
            start = frame
        previous = frame
    return [*out, [start, previous + 1]]


def main():
    output = m.REPO / "work/three_body/scene0_probe_support_audit_v1"
    if output.exists():
        raise FileExistsError("audit output exists")
    codes = {
        str(path): m.sha256(path)
        for path in sorted(set(m.code_paths()) | {Path(__file__).resolve()})
    }
    source = m.identity_inputs._verify_run(
        m.REPO / "work/three_body/persistent_mouth_continuous_one_minute_v1", appearance=False
    )
    hashes = dict(source.hashes) | codes

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        m.require(path.is_relative_to(m.REPO), "input outside workspace")
        actual = hashes.get(str(path)) or m.sha256(path)
        m.require(expected is None or actual == expected, "input SHA mismatch")
        hashes[str(path)] = actual
        return path

    geometry = m.identity_inputs._npz(source.root / "candidate.npz")
    memory_root = m.REPO / "work/three_body/identity_memory_continuous_one_minute_v1"
    memory_record = json.loads(bind(memory_root / "experiment.provenance.json").read_text())
    m.require(
        memory_record["status"] == "ok"
        and memory_record["stage"] == "three_body_fixed_geometry_identity_memory_dp",
        "memory run incomplete or wrong stage",
    )
    m.require(
        Path(memory_record["params"]["geometry_run"]).resolve() == source.root,
        "memory uses different geometry",
    )
    for name in ("code_unchanged_during_run", "inputs_unchanged_during_run"):
        m.require(memory_record["params"][name] is True, "memory completion guard missing")
    m.require(
        memory_record["params"]["evaluation_data_opened_during_inference"] is False,
        "memory inference used evaluation data",
    )
    for item in memory_record["outputs"].values():
        bind(item["path"], item["sha256"])
    for key, item in memory_record["inputs"].items():
        m.require(
            Path(key).resolve() == Path(item["path"]).resolve(), "memory input role/path mismatch"
        )
        bind(item["path"], item["sha256"])
    m.require(
        str(source.root / "candidate.npz") in memory_record["inputs"],
        "geometry candidate unbound in memory",
    )
    identity_path = bind(memory_root / "identity.npz")
    m.require(
        any(Path(e["path"]).resolve() == identity_path for e in memory_record["outputs"].values()),
        "identity output unsealed",
    )
    identity = m.identity_inputs._npz(identity_path)
    expected = build_identity_output(geometry, identity["identity_to_raw_slot"])
    m.require(set(expected) == set(identity), "identity export schema differs")
    for key, value in expected.items():
        m.require(
            identity[key].dtype == value.dtype
            and identity[key].shape == value.shape
            and identity[key].tobytes() == value.tobytes(),
            f"memory {key} is not the frozen-geometry permutation",
        )
    page = m.REPO / "work/three_body/minute_reviews/scene0_79236-79488"
    page_record = json.loads(bind(page / "review_scene.provenance.json").read_text())
    m.require(page_record["status"] == "ok", "review page incomplete")
    for item in page_record["outputs"].values():
        bind(item["path"], item["sha256"])
    page_summary = json.loads(bind(page / "summary.json").read_text())
    m.require(page_summary["probe_frames"] == list(PROBES), "review probes changed")
    m.require(
        page_summary["review_clip_range"] == list(CLIP)
        and page_summary["displayed_frames"] == list(range(*CLIP)),
        "review clip changed",
    )
    m.require(
        page_summary["source_manifest_sha256"] == hashes[str(source.sequence / "manifest.json")],
        "review page uses another raw source",
    )
    rig = load_rig(source.root / "calibration_snapshot")
    m.require(tuple(rig.views) == tuple(m.VIEWS), "camera order differs")
    node_path = m.LAB / "node_geometry.py"
    m.require(
        m.sha256(node_path) == source.code["labs/20260909_three_body/node_geometry.py"],
        "ray support code changed",
    )
    frames = np.arange(*CLIP, dtype=np.int64)
    nodes = CachedNodes(source.input_root, source.manifest["experiment"], frames)
    for path in nodes.paths.values():
        m.require(str(path.resolve()) in hashes, "unbound source nodes")
    output.mkdir(parents=True)
    with provenance("unreviewed_probe_geometric_readiness", output, name="audit") as record:
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
        for path, digest in codes.items():
            target = output / "code_snapshot" / Path(path).relative_to(m.REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            m.require(m.sha256(target) == digest, "source snapshot differs")
            record.output(str(target.relative_to(output)), target, sha256=digest)
        gates, projection, errors = [], [], []
        probe_rows, csv_rows = [], []
        for frame in frames:
            t = int(frame - geometry["frame_idx"][0])
            points = {view: nodes.at(view, int(frame)) for view in m.VIEWS}
            rows, uv, gate, error = inspect_state(
                geometry["pose_xyz"][t],
                geometry["sources"][t],
                geometry["observed"][t],
                geometry["identity_review"][t],
                points,
                rig,
            )
            gates.append(gate)
            projection.append(uv)
            errors.append(error)
            if frame in PROBES:
                mapped = identity["identity_to_raw_slot"][t]
                other, other_uv, other_gate, _ = inspect_state(
                    identity["proposed_pose_xyz"][t],
                    identity["proposed_sources"][t],
                    identity["proposed_observed"][t],
                    identity["prior_review_for_proposed_identity"][t],
                    points,
                    rig,
                )
                m.require(
                    np.array_equal(uv[mapped], other_uv)
                    and np.array_equal(gate[mapped], other_gate),
                    "memory mouth reprojection differs after permutation",
                )
                probe_rows.append(
                    dict(
                        frame=int(frame),
                        memory_identity_to_raw_slot=mapped,
                        persistent_raw_slots=rows,
                        memory_proposed_slots=other,
                    )
                )
                for fish, row in enumerate(rows):
                    for v, view in enumerate(m.VIEWS):
                        csv_rows.append(
                            dict(
                                frame=int(frame),
                                raw_slot=fish,
                                view=view,
                                mouth_observed=row["mouth_observed"],
                                source_row=int(row["mouth_sources_by_view"][v]),
                                projection_finite=bool(row["projection_finite_by_view"][v]),
                                projection_inside_image=bool(
                                    row["projection_inside_image_by_view"][v]
                                ),
                                ray_distance_mm=float(row["assigned_ray_distance_mm_by_view"][v]),
                                reprojection_error_pixels=float(
                                    row["assigned_reprojection_error_pixels_by_view"][v]
                                ),
                                direct_gate=bool(row["direct_view_gate"][v]),
                                original_identity_review=row["original_identity_review"],
                            )
                        )
        gates = np.stack(gates)
        all_direct = (gates.sum(-1) >= 2).all(-1)
        eligible = frames[all_direct]
        path = output / "clip_mouth_support.npz"
        np.savez_compressed(
            path,
            frame_idx=frames,
            raw_mouth_view_gate=gates,
            projected_mouth_xy=np.stack(projection),
            assigned_pixel_error=np.stack(errors),
            all_three_mouth_direct=all_direct,
            memory_identity_to_raw_slot=identity["identity_to_raw_slot"][
                frames - geometry["frame_idx"][0]
            ],
        )
        record.output(path.name, path, sha256=m.sha256(path))
        path = output / "probe_view_support.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
        record.output(path.name, path, sha256=m.sha256(path))
        result = dict(
            status="ok",
            probe_frames=PROBES,
            clip_range=CLIP,
            source_manifest_sha256=hashes[str(source.sequence / "manifest.json")],
            persistent_candidate_sha256=hashes[str(source.root / "candidate.npz")],
            memory_identity_sha256=hashes[str(identity_path)],
            human_template_deserialized=False,
            human_correspondence_or_continuity_created=False,
            thresholds_unchanged=dict(ray_mm=2, pixel_error=12, minimum_views=2),
            direct_gate_uses_original_float32_assigned_rays=True,
            full_identity_export_bitwise_reproduced=True,
            probes=probe_rows,
            clip_frames=len(frames),
            all_three_mouth_direct_count=len(eligible),
            all_three_mouth_direct_frames=eligible,
            all_three_mouth_direct_ranges_half_open=intervals(eligible),
            clip_persistent_and_memory_mapping_same=bool(
                np.all(
                    identity["identity_to_raw_slot"][frames - geometry["frame_idx"][0]] == [0, 1, 2]
                )
            ),
            limitation=(
                "Finite latent mouth projection or two-view geometric support does not "
                "establish correct anatomical correspondence, physical identity, visibility "
                "or confirmed human continuity. Original review flags are preserved; "
                "no source image or template was changed."
            ),
        )
        path = output / "summary.json"
        path.write_text(json.dumps(m.safe(result), indent=2, allow_nan=False), encoding="utf-8")
        record.output("summary", path, sha256=m.sha256(path))
        m.require(
            all(m.sha256(path) == digest for path, digest in hashes.items()),
            "audit input/source changed",
        )
        m.require(
            all(m.sha256(item["path"]) == item["sha256"] for item in record.outputs.values()),
            "audit output changed",
        )
        record.param("all_input_source_output_sha_reverified", True)
        print(
            json.dumps(
                dict(
                    status="ok",
                    probes=list(PROBES),
                    all_three_mouth_direct_count=len(eligible),
                    ranges=intervals(eligible),
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
