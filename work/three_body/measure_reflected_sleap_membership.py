"""Classify existing SLEAP pixels by fixed finite optical paths and D2 membership.

No image decoding, fit, detection filtering, triangulation or identity changes.
This development diagnostic does not establish a pixel or animal truth label.
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))

import identity_inputs as verified  # noqa: E402
import reflection_paths as optics  # noqa: E402
from observations import CachedNodes  # noqa: E402
from prepare_sequence import VIEWS, array_sha256, decode_frame_masks, sha256  # noqa: E402
from reflection_probe import FRAMES  # noqa: E402

from medaka3d.calib import load_rig  # noqa: E402
from medaka3d.provenance import provenance  # noqa: E402


def distance(point, origin, direction, limit):
    if not np.isfinite(origin).all() or not np.isfinite(direction).all():
        return None
    depth = float(np.dot(point - origin, direction))
    clipped = float(np.clip(depth, 0, limit))
    return dict(peak_depth_mm=depth, segment_depth_mm=clipped,
                distance_mm=float(np.linalg.norm(point - origin - clipped * direction)))


def main():
    output = ROOT / "work/three_body/reflected_sleap_membership_13frames_v2"
    if output.exists():
        raise ValueError("new output required")
    run = verified._verify_run(
        ROOT / "work/three_body/persistent_mouth_continuous_one_minute_v1", appearance=False
    )
    source_paths = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == ".py" and (
                path.is_relative_to(LAB) or path.is_relative_to(ROOT / "src")
            ):
                source_paths.add(path)
    sources = {str(p): sha256(p) for p in sorted(source_paths)}
    for path in (LAB / "observations.py", LAB / "prepare_sequence.py",
                 ROOT / "src/medaka3d/calib.py", ROOT / "src/medaka3d/types.py"):
        if sha256(path) != run.code[path.relative_to(ROOT).as_posix()]:
            raise ValueError("measurement source differs from the fixed producer")
    if sha256(Path(optics.__file__)) != (
        "e8f64a776d546a35c7eb10c0af30edee896b9ab75b388d1df289dc4dc3cc86e3"
    ):
        raise ValueError("fixed optical source changed")
    inputs = {**run.hashes, **sources}
    output.mkdir()
    plan = dict(
        frames=list(FRAMES), semantic_nodes=["mouth", "pectoral", "tail"],
        all_SLEAP_rows=True, masks="sealed independent D2, all overlapping instances",
        lookup="nearest integer pixel via np.rint; reject rounded coordinates outside image",
        optics="fixed nominal box and y192/y196 branches, unchanged from prior probe",
        fit=False, GT=False, raw_image_decode=False,
        meanings="Optical eligibility is not proof of reflection or animal identity. "
                 "Selected old poses are diagnostic references only; no new coordinates.",
        input_and_source_hashes=dict(inputs),
    )
    plan_path = output / "plan.json"
    with provenance("register_reflected_sleap_membership", output, name="registration") as record:
        for path, digest in inputs.items():
            record.input(path, path, sha256=digest)
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        record.output("plan", plan_path, sha256=sha256(plan_path))
    # Measurements begin only after the plan has been sealed.
    inputs[str(plan_path)] = sha256(plan_path)
    inputs[str(output / "registration.provenance.json")] = sha256(
        output / "registration.provenance.json"
    )
    with provenance("reflected_sleap_membership", output, name="measurement") as record:
        for path, digest in inputs.items():
            record.input(path, path, sha256=digest)
        for path, digest in sources.items():
            target = output / "code_snapshot" / Path(path).relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            record.output(target.relative_to(output).as_posix(), target, sha256=digest)
        rig = load_rig(run.root / "calibration_snapshot")
        nodes = CachedNodes(run.input_root, run.manifest["experiment"], run.frames)
        saved = verified._npz(run.root / "candidate.npz")
        packed = {
            view: verified._npz(run.sequence / run.manifest["masks"][view]["path"])
            for view in VIEWS
        }
        records = []
        for frame in FRAMES:
            t = frame - int(run.frames[0])
            for v, view in enumerate(VIEWS):
                points = nodes.at(view, frame)
                masks = decode_frame_masks(packed[view], t)
                if array_sha256(masks) != packed[view]["frame_mask_sha256"][t].decode():
                    raise ValueError("decoded mask changed")
                camera = rig.camera(view)
                flat = points.reshape(-1, 2)
                finite = np.isfinite(flat).all(axis=-1)
                # Camera API receives only finite pixels, retaining original row/node indices.
                origins, directions = camera.rays(flat[finite])
                paths = optics.trace_y192_reflections(
                    origins, directions, index_water=camera.index_water,
                    index_glass=camera.index_glass, index_air=camera.index_air,
                )
                for k, flat_index in enumerate(np.flatnonzero(finite)):
                    row, node = divmod(int(flat_index), 3)
                    pixel = np.rint(flat[flat_index]).astype(np.int64)
                    inside = bool(np.all(pixel >= 0) and np.all(pixel < 1200))
                    membership = (
                        np.flatnonzero(masks[:, pixel[1], pixel[0]]).tolist() if inside else []
                    )
                    owners = np.flatnonzero(saved["sources"][t, :, v, node] == row)
                    selected = []
                    for slot in owners:
                        point = saved["pose_xyz"][t, slot, node]
                        selected.append(dict(
                            slot=int(slot), pose_xyz=point.tolist(),
                            unbounded=distance(point, origins[k], directions[k], np.inf),
                            finite_direct=(distance(point, origins[k], directions[k],
                                                    paths.first_distance_mm[k])
                                           if paths.direct_valid[k] else None),
                            inner=(distance(point, paths.inner.origins[k],
                                            paths.inner.directions[k],
                                            paths.inner.max_distance_mm[k])
                                   if paths.inner.valid[k] else None),
                            outer=(distance(point, paths.outer.origins[k],
                                            paths.outer.directions[k],
                                            paths.outer.max_distance_mm[k])
                                   if paths.outer.valid[k] else None),
                        ))
                    records.append(dict(
                        frame_idx=frame, view=view, sleap_row=row, node=node,
                        uv=flat[flat_index].tolist(), D2_instances=membership,
                        image_pixel_valid=inside,
                        direct_valid=bool(paths.direct_valid[k]),
                        first_wall=optics.Wall(int(paths.first_wall[k])).name,
                        first_reason=optics.Reason(int(paths.first_reason[k])).name,
                        inner_valid=bool(paths.inner.valid[k]),
                        outer_valid=bool(paths.outer.valid[k]),
                        inner_weight=float(paths.inner.weight[k]),
                        outer_weight=float(paths.outer.weight[k]), selected=selected,
                    ))
        primary = [r for r in records if r["frame_idx"] == 79381
                   and r["view"] == "top_cam" and 3 in r["D2_instances"]]
        result = dict(plan=plan, records=records, primary_reflection_mask_SLEAP=primary,
                      finite_SLEAP_nodes=len(records), no_geometry_or_ID_change=True)
        target = output / "summary.json"
        target.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        record.output("summary", target, sha256=sha256(target))
        for path, digest in inputs.items():
            if sha256(path) != digest:
                raise ValueError("measurement input changed")
        for entry in record.outputs.values():
            if sha256(entry["path"]) != entry["sha256"]:
                raise ValueError("measurement output changed")
    print(json.dumps(dict(output=str(output), nodes=len(records), primary=primary)))


if __name__ == "__main__":
    main()
