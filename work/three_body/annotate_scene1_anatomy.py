"""Check diagnostic ray intersections against body length and node exclusivity."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))
from diagnose_minute_scene1 import RUN, clean, sha256  # noqa: E402
from observations import CachedNodes  # noqa: E402

from medaka3d.calib import load_rig  # noqa: E402
from medaka3d.provenance import provenance  # noqa: E402
from medaka3d.types import VIEWS  # noqa: E402

SOURCE = ROOT / "work/three_body/minute_scene1_mouth_diagnosis_v2"
OUT = ROOT / "work/three_body/minute_scene1_anatomy_v2"


def main():
    OUT.mkdir(exist_ok=False)
    source_seal = json.loads((SOURCE / "diagnosis.provenance.json").read_text())
    assert source_seal["status"] == "ok"
    source_path = SOURCE / "native_geometry.json"
    assert sha256(source_path) == source_seal["outputs"]["native_geometry"]["sha256"]
    rows = json.loads(source_path.read_text())
    with np.load(RUN / "candidate.npz", allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files}
    initial = np.load(RUN / "initial_pose_xyz.npy", allow_pickle=False)
    lengths = np.linalg.norm(initial[:, 0] - initial[:, 1], axis=-1)
    rig = load_rig(RUN / "calibration_snapshot")
    nodes = CachedNodes(
        ROOT / "work/generative_tracking_inputs", "20251225_K_1_2_FCMC", result["frame_idx"]
    )
    paths = [
        Path(__file__),
        ROOT / "work/three_body/diagnose_minute_scene1.py",
        source_path,
        SOURCE / "diagnosis.provenance.json",
        RUN / "candidate.npz",
        RUN / "initial_pose_xyz.npy",
        *(RUN / "calibration_snapshot").glob("*.json"),
        *nodes.paths.values(),
    ]
    hashes = {str(path): sha256(path) for path in paths}
    with provenance("diagnostic_scene1_anatomical_gate", OUT, name="anatomy") as record:
        record.param("diagnostic_only_no_tracking_or_optimization", True)
        record.param("anatomy_reference", "registered initial mouth-to-pectoral length")
        record.param("engineering_ratio_test", [0.5, 1.5])
        record.param("same_node_per_view_exclusivity", True)
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
        annotated = []
        for row in rows:
            if row["mouth_direct"]:
                continue
            frame, fish = row["frame"], row["fish"]
            t = frame - int(result["frame_idx"][0])
            pectoral = result["pose_xyz"][t, fish, 1]
            candidates = []
            for candidate in row["nearest_unclaimed_ray_intersections"]:
                xyz = np.array(candidate["xyz"])
                length = float(np.linalg.norm(xyz - pectoral))
                ratio = length / lengths[fish]
                per_view = []
                for view, source in zip(candidate["views"], candidate["source_rows"], strict=True):
                    v = VIEWS.index(view)
                    mouth = nodes.at(view, frame)[source, 0]
                    origin, direction = rig.camera(view).rays(mouth)
                    origin, direction = origin.astype(np.float32), direction.astype(np.float32)
                    delta = xyz - origin
                    error = float(np.linalg.norm(delta - np.dot(delta, direction) * direction))
                    projected = rig.camera(view).project(xyz)
                    pixel_error = float(np.linalg.norm(projected - mouth))
                    exclusive = bool(source not in result["sources"][t, np.arange(3) != fish, v, 0])
                    per_view.append(
                        dict(
                            view=view,
                            source_row=source,
                            ray_error_mm=error,
                            reprojection_error_px=pixel_error,
                            same_node_source_unused_by_other_slots=exclusive,
                            passes_saved_support_threshold=bool(error <= 2 and pixel_error <= 12),
                        )
                    )
                anatomy = bool(0.5 <= ratio <= 1.5)
                exclusive = all(v["same_node_source_unused_by_other_slots"] for v in per_view)
                supports = sum(v["passes_saved_support_threshold"] for v in per_view)
                geometry = bool(
                    candidate["within_engineering_box"] and anatomy and exclusive and supports >= 2
                )
                candidates.append(
                    dict(
                        **candidate,
                        mouth_to_saved_pectoral_mm=length,
                        ratio_to_initial_mouth_pectoral=ratio,
                        anatomy_05_to_15=anatomy,
                        per_view=per_view,
                        supported_view_count=supports,
                        geometry_and_anatomy_eligible=geometry,
                        geometry_eligible_with_direct_pectoral=bool(
                            geometry and result["observed"][t, fish, 1]
                        ),
                    )
                )
            annotated.append(
                dict(
                    frame=frame,
                    fish=fish,
                    saved_mouth_xyz=row["mouth_xyz"],
                    saved_pectoral_xyz=pectoral,
                    pectoral_direct=bool(result["observed"][t, fish, 1]),
                    saved_mouth_to_pectoral_ratio=float(
                        np.linalg.norm(np.array(row["mouth_xyz"]) - pectoral) / lengths[fish]
                    ),
                    candidates=candidates,
                )
            )
        path = OUT / "anatomy.json"
        path.write_text(json.dumps(clean(annotated), indent=2, allow_nan=False))
        record.output("anatomy", path, sha256=sha256(path))
        summary = dict(initial_mouth_pectoral_mm=lengths, fish=[])
        for fish, start, stop in ((1, 79965, 80370), (2, 80340, 80463)):
            mask = (result["frame_idx"] >= start) & (result["frame_idx"] < stop)
            selected = [r for r in annotated if r["fish"] == fish and start <= r["frame"] < stop]
            eligible = [
                r["frame"]
                for r in selected
                if any(c["geometry_eligible_with_direct_pectoral"] for c in r["candidates"])
            ]
            summary["fish"].append(
                dict(
                    fish=fish,
                    missing_mouth_range=[start, stop],
                    frames=stop - start,
                    missing_seconds=(stop - start) / 166,
                    nearest_direct_mouth_before=start - 1,
                    nearest_direct_mouth_after=stop,
                    directly_supported_pectoral_frames=int(result["observed"][mask, fish, 1].sum()),
                    body_supported_frames=int(result["body_supported"][mask, fish].sum()),
                    mask_supported_frames=int(result["mask_supported"][mask, fish].sum()),
                    eligible_diagnostic_frames=eligible,
                    eligible_inside_original_scene=sum(80261 <= f < 80432 for f in eligible),
                )
            )
        summary["limitations"] = [
            "No ID truth was read or created; eligibility does not establish animal identity.",
            "The first audit retained only the three intersections nearest the saved mouth.",
            "A negative eligibility finding is not an exhaustive proof that no candidate exists.",
            "Only same-node source exclusivity is required, matching the frozen tracker schema.",
            "No saved pose, identity review, source assignment or review interval was changed.",
        ]
        path = OUT / "summary.json"
        path.write_text(json.dumps(clean(summary), indent=2, allow_nan=False))
        record.output("summary", path, sha256=sha256(path))
        assert {path: sha256(path) for path in hashes} == hashes
        (OUT / "audit.py").write_bytes(Path(__file__).read_bytes())
        record.output("audit.py", OUT / "audit.py", sha256=sha256(OUT / "audit.py"))
        print(json.dumps(clean(summary), indent=2))


if __name__ == "__main__":
    main()
