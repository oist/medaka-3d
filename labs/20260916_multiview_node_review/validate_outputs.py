"""Verify saved proposals remain tied to actual detector points and preserved baseline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from run import DetectionCache, write_json

from medaka3d.aquarium import sha256
from medaka3d.provenance import provenance


def main():
    root = Path("work/multiview_node_review_v1")
    summary = json.loads((root / "analysis/summary.json").read_text(encoding="utf-8"))
    result = dict(cases=[], accepted_identity_corrections=0)
    with provenance("multiview_output_validation", root) as rec:
        rec.input("validator", Path(__file__), sha256=sha256(Path(__file__)))
        cache = DetectionCache(Path("work/generative_tracking_inputs"), rec)
        for entry in summary["cases"]:
            name = entry["case"]["name"]
            path = root / "analysis" / name / "evaluation.npz"
            rec.input(name, path, sha256=sha256(path))
            with np.load(path, allow_pickle=False) as a:
                frames = a["frames"]
                untested = ~a["tested"]
                assert np.array_equal(
                    a["baseline_pose"][untested], a["candidate_pose"][untested], equal_nan=True
                )
                assert np.array_equal(
                    a["original_source"][untested], a["candidate_source"][untested]
                )
                for source_key, uv_key in (
                    ("original_source", "original_uv"),
                    ("candidate_source", "candidate_uv"),
                ):
                    owners = a[source_key]
                    for view, (points, offsets) in enumerate(cache.data):
                        for node in range(3):
                            rows = owners[:, :, view, node]
                            present = rows >= 0
                            assert np.all(
                                rows[present]
                                < np.broadcast_to(np.diff(offsets)[frames, None], rows.shape)[
                                    present
                                ]
                            )
                            selected = np.full((len(frames), 3, 2), np.nan)
                            indices = offsets[frames, None] + rows
                            selected[present] = points[indices[present], node]
                            assert np.array_equal(
                                selected, a[uv_key][:, :, view, node], equal_nan=True
                            )
                            for s1, s2 in ((0, 1), (0, 2), (1, 2)):
                                assert not np.any((rows[:, s1] >= 0) & (rows[:, s1] == rows[:, s2]))
                for count_key, mask in (
                    ("tested_frames", a["tested"]),
                    ("changed_frames", a["changed"]),
                    ("review_worthy_frames", a["supported"]),
                ):
                    assert int((mask & a["core"]).sum()) == entry[count_key]
                result["cases"].append(
                    dict(
                        name=name,
                        source_node_coordinates_verified=True,
                        bijective_ownership_verified=True,
                        untested_frames_preserved=True,
                        summary_counts_verified=True,
                    )
                )
        paths = {}
        for filename in ("manifest.json", "supplemental_manifest_v3.json"):
            path = root / "review" / filename
            rec.input(filename, path, sha256=sha256(path))
            for entry in json.loads(path.read_text(encoding="utf-8"))["entries"]:
                artifact = root / "review" / entry["path"]
                assert sha256(artifact) == entry["metadata"]["sha256"]
                paths[entry["path"]] = entry["kind"]
        result["verified_unique_manifest_artifacts"] = {
            kind: sum(x == kind for x in paths.values()) for kind in ("image", "plot", "video")
        }
        support = root / "review/tail_support_76988.png"
        support_record = json.loads(
            (root / "review/tail_support_76988.provenance.json").read_text(encoding="utf-8")
        )
        assert support_record["status"] == "ok"
        assert any(x.get("sha256") == sha256(support) for x in support_record["outputs"].values())
        result["additional_support_plot_verified"] = True
        result["existing_tracking_outputs_modified"] = False
        write_json(root / "validation.json", result)
        rec.output("validation", root / "validation.json", sha256=sha256(root / "validation.json"))
    print(json.dumps(result["verified_unique_manifest_artifacts"]))


if __name__ == "__main__":
    main()
