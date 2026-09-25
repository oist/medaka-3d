"""Count geometry-only coverage across views; never compute appearance or IDs."""

import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GEOMETRY = ROOT / "dino_minute_adjacent_geometry_preparation_v2"
SEAL_SHA = "32f601b81241b9fc3e8eb93d7820fd90efcbc080473a7c381d06c3a96926d15c"
VIEWS = ("top_cam", "side1_cam", "side2_cam")
FRAMES = tuple(66483 + 166 * k for k in range(60))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    seal_path = GEOMETRY / "geometry.provenance.json"
    assert sha(seal_path) == SEAL_SHA
    seal = json.loads(seal_path.read_text())
    assert seal["status"] == "ok"
    path = GEOMETRY / "geometry_references.json"
    expected = seal["outputs"][path.name]
    assert sha(path) == expected["sha256"]
    assert path.stat().st_size == expected["size_bytes"]
    groups = json.loads(path.read_text())
    index = {(g["method"], g["frame_idx"], g["view"]): g for g in groups}
    assert len(index) == len(groups) == 360
    assert set(index) == set(itertools.product(("original", "tail"), FRAMES, VIEWS))
    records, summaries = [], {}
    for method in ("original", "tail"):
        rows = []
        for left, right in itertools.pairwise(FRAMES):
            edges = {}
            full_views = []
            for view in VIEWS:
                a, b = index[method, left, view], index[method, right, view]
                for group in (a, b):
                    selected = [
                        s["assigned_mask_row"] for s in group["slots"] if s["endpoint_eligible"]
                    ]
                    assert len(selected) == len(set(selected))
                edges[view] = [
                    [
                        bool(
                            a["slots"][i]["endpoint_eligible"]
                            and b["slots"][j]["endpoint_eligible"]
                        )
                        for j in range(3)
                    ]
                    for i in range(3)
                ]
                if a["all3_endpoint_eligible"] and b["all3_endpoint_eligible"]:
                    full_views.append(view)
            counts = [[sum(edges[v][i][j] for v in VIEWS) for j in range(3)] for i in range(3)]
            possible = [
                list(p)
                for p in itertools.permutations(range(3))
                if all(counts[i][p[i]] for i in range(3))
            ]
            row = dict(
                method=method,
                left=left,
                right=right,
                view_edge_eligible=edges,
                edge_available_view_counts=counts,
                single_view_full=full_views,
                cross_view_full9=all(c for line in counts for c in line),
                same_slot_all3=all(counts[i][i] for i in range(3)),
                available_permutations=possible,
            )
            rows.append(row)
        summaries[method] = dict(
            adjacent_pairs=len(rows),
            any_single_view_full9=sum(bool(r["single_view_full"]) for r in rows),
            cross_view_union_full9=sum(r["cross_view_full9"] for r in rows),
            full9_only_after_view_union=sum(
                r["cross_view_full9"] and not r["single_view_full"] for r in rows
            ),
            same_slot_all3_after_view_union=sum(r["same_slot_all3"] for r in rows),
            any_permutation_after_view_union=sum(bool(r["available_permutations"]) for r in rows),
            available_permutation_count_histogram={
                str(n): sum(len(r["available_permutations"]) == n for r in rows) for n in range(7)
            },
        )
        records.extend(rows)
    result = dict(
        schema="dino_minute_geometry_observability_v1",
        source_sha256=sha(Path(__file__)),
        geometry_seal_sha256=SEAL_SHA,
        geometry_references_sha256=expected["sha256"],
        summaries=summaries,
        records=records,
        limitations=[
            "Geometry eligibility only; token validity and numerical distances not consulted.",
            "View-union coverage does not establish comparable camera scores or correct identity.",
            "Same slot and permutation refer only to each method's latent 3D slots.",
            "No tracker, image, model, or ID updates; all 59 fixed adjacent pairs retained.",
        ],
    )
    out = ROOT / "dino_minute_geometry_observability_v2.json"
    with out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    assert sha(path) == expected["sha256"] and sha(seal_path) == SEAL_SHA
    print(json.dumps(dict(output=str(out), sha256=sha(out), summaries=summaries)))


if __name__ == "__main__":
    main()
