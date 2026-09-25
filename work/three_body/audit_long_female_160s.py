"""Independent arithmetic audit of completed fixed160 sparse female endpoints.

Read-only producers; no raw video, fitting, GPU, correction or new annotation.
The matching and endpoint loops below do not call the evaluated scorer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from medaka3d.provenance import provenance  # noqa: E402

FIRST, STOP, CHUNK = 49800, 76360, 1660
VIEWS = ("top_cam", "side1_cam", "side2_cam")
METHODS = ("new", *("c34_lambda_" + x for x in ("0", "0.1", "0.25", "0.5", "0.767", "2", "8")))
EVAL_SEAL_SHA = "e5894da372b2f1b577e8c0ac2d6930140231a6674db90594501dda58d1fba5b6"
GT_SHA = "ab0a173ac737a63311aec834c5832fc19475d4f7d1892b8193f8465bba67519f"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Evidence:
    def __init__(self):
        self.hashes = {}

    def add(self, path, expected=None):
        path = Path(path).resolve(strict=True)
        require(path.is_relative_to(REPO), "input outside workspace")
        digest = self.hashes.get(str(path)) or sha(path)
        require(expected is None or digest == expected, "bound input SHA mismatch")
        self.hashes[str(path)] = digest
        return path

    def read(self, path, expected=None):
        return json.loads(self.add(path, expected).read_text(encoding="utf-8-sig"))

    def guard(self):
        require(all(sha(p) == h for p, h in self.hashes.items()), "opened input changed")


def verify_seal(root, name, stage, evidence, expected=None):
    record = evidence.read(root / name, expected)
    require(record["stage"] == stage and record["status"] == "ok",
            "completed expected seal required")
    for item in record["outputs"].values():
        path = evidence.add(item["path"], item["sha256"])
        require(path.is_relative_to(root) and path.stat().st_size == item["size"],
                "output scope/size mismatch")
    return record


def independent_anchors(csv_path, node_paths):
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        selected = [r for r in csv.DictReader(stream)
                    if r["gt_status"] == "confirmed" and FIRST <= int(r["frame"]) < STOP]
    selected.sort(key=lambda r: int(r["frame"]))
    frames = [int(r["frame"]) for r in selected]
    require(len(frames) == len(set(frames)), "duplicate confirmed anchor")
    require(all(r["sex_label"] == "F" for r in selected), "confirmed row not female")
    pools = {}
    for view, path in node_paths.items():
        with np.load(path, allow_pickle=False) as stored:
            index, points = stored["frame_idx"], stored["points"]
            require(str(stored["view"]) == view and tuple(stored["nodes"])
                    == ("mouth", "pectoral", "tail"), "SL view/node schema differs")
            require(index.ndim == 1 and index.dtype.kind in "iu" and np.all(np.diff(index) >= 0)
                    and points.shape == (len(index), 3, 2), "SL native order differs")
            pools[view] = {f: points[index == f, 0].astype(np.float64) for f in frames}
    anchors, diagnostics = [], []
    for row, frame in zip(selected, frames, strict=True):
        matches, pixels, errors = {}, {}, {}
        for view, prefix in (("side1_cam", "side1"), ("side2_cam", "side2")):
            if row[prefix + "_selection_status"] != "sleap_linked":
                diagnostics.append(dict(frame=frame, view=view, reason="human view not linked"))
                continue
            target = np.array([row[prefix + "_mouth_x"], row[prefix + "_mouth_y"]], float)
            require(np.isfinite(target).all(), "nonfinite actual human mouth")
            pixels[view] = target.tolist()
            values = pools[view][frame]
            squared = np.sum((values - target)**2, axis=1)
            distance = np.sqrt(np.where(np.isfinite(squared), squared, np.inf))
            ordered = sorted(range(len(distance)), key=lambda i: distance[i])
            closest = distance[ordered[0]] if ordered else np.inf
            second = distance[ordered[1]] if len(ordered) > 1 else np.inf
            tied = len(ordered) > 1 and bool(np.isclose(closest, second))
            accepted = bool(ordered and closest <= 5. and not tied)
            if accepted:
                matches[view], errors[view] = ordered[0], float(closest)
            diagnostics.append(dict(
                frame=frame, view=view, row=ordered[0] if accepted else None,
                nearest_error_px=float(closest) if np.isfinite(closest) else None,
                second_error_px=float(second) if np.isfinite(second) else None,
                tied=tied, reason="matched" if accepted else "missing, outside5px or tied"))
        anchors.append(dict(frame=frame, detection_rows=matches, pixels=pixels,
                            match_errors_px=errors))
    return anchors, diagnostics, pools


def equal_anchors(actual, expected):
    require(len(actual) == len(expected), "anchor denominator differs")
    for a, b in zip(actual, expected, strict=True):
        require(all(a[k] == b[k] for k in ("frame", "detection_rows", "pixels")),
                "actual CSV/SL anchor correspondence differs")
        require(set(a["match_errors_px"]) == set(b["match_errors_px"]), "view errors differ")
        for view in a["match_errors_px"]:
            require(abs(a["match_errors_px"][view] - b["match_errors_px"][view]) <= 1e-12,
                    "anchor match distance differs")


def independent_score(anchor_states, anchors, accepted):
    assigned = []
    for anchor in anchors:
        state = anchor_states[anchor["frame"]]
        candidates, duplicate = set(), False
        for view, row in anchor["detection_rows"].items():
            slots = [i for i in range(3) if state["sources"][i, VIEWS.index(view), 0] == row]
            candidates.update(slots)
            duplicate |= len(slots) > 1
        slot, reason = None, "matched"
        if duplicate or len(candidates) > 1:
            reason = "conflicting source identities"
        elif not candidates:
            reason = "mouth observation unmatched"
        else:
            slot = next(iter(candidates))
            if not state["observed"][slot, 0]:
                slot, reason = None, "mouth not supported by two views"
            elif accepted and not state["label"][slot]:
                slot, reason = None, "identity requires review"
        assigned.append(dict(frame=anchor["frame"], identity=slot, reason=reason))
    intervals, counts = [], dict(preserved=0, swap=0, unscored=0)
    for left, right in pairwise(assigned):
        outcome = "unscored" if left["identity"] is None or right["identity"] is None else (
            "preserved" if left["identity"] == right["identity"] else "swap")
        counts[outcome] += 1
        intervals.append(dict(start=left["frame"], stop=right["frame"], outcome=outcome))
    return dict(requested_anchors=len(anchors), available_anchors=sum(
        x["identity"] is not None for x in assigned), requested_intervals=len(intervals),
        interval_counts=counts, anchors=assigned, intervals=intervals)


def run(evaluation_root, output):
    root, output = Path(evaluation_root).resolve(), Path(output).resolve()
    require(output.is_relative_to(REPO) and not output.exists(), "new workspace output required")
    evidence = Evidence()
    evidence.add(__file__)
    evidence.add(REPO / "src/medaka3d/provenance.py")
    evaluation = verify_seal(root, "evaluation.provenance.json",
                             "long_female_endpoint_evaluation_160s", evidence, EVAL_SEAL_SHA)
    require(evaluation["params"]["all_input_source_output_sha_unchanged"] is True,
            "evaluation guard missing")
    for item in evaluation["inputs"].values():
        evidence.add(item["path"], item["sha256"])
    results = evidence.read(root / "summary.json")
    saved_anchors = evidence.read(root / "anchor_matches.json")
    compared = [Path(x["path"]) for x in evaluation["inputs"].values()
                if Path(x["path"]).name == "comparison.provenance.json"]
    require(len(compared) == 1, "one source geometry comparison required")
    comparison_root = compared[0].parent
    comparison = verify_seal(comparison_root, "comparison.provenance.json",
                             "long_160s_all7_geometry_support_comparison", evidence)
    require(comparison["params"]["all_used_input_source_snapshot_output_sha_unchanged"] is True,
            "comparison guard missing")
    summary = evidence.read(comparison_root / "summary.json")
    require(tuple(m["name"] for m in summary["methods"]) == METHODS
            == tuple(m["name"] for m in results["methods"]), "method order/count differs")
    gt = [Path(x["path"]) for x in evaluation["inputs"].values()
          if Path(x["path"]).name == "female_sex_gt.csv"]
    require(len(gt) == 1, "one original female CSV required")
    evidence.add(gt[0], GT_SHA)
    require(sha(root / "original_female_sex_gt.csv") == GT_SHA, "copied annotation differs")
    nodes = {v: Path(p) for p in summary["shared_SL_and_calibration_sha256"] for v in VIEWS
             if Path(p).name == f"20251225_K_1_2_FCMC__{v}.npz"}
    require(set(nodes) == set(VIEWS), "three shared SL caches required")
    for path in nodes.values():
        require(str(path) in evidence.hashes, "SL cache is not evaluation-bound")
    anchors, match_diagnostics, pools = independent_anchors(gt[0], nodes)
    equal_anchors(anchors, saved_anchors)
    require(len(anchors) == results["requested_anchors"], "summary anchor count differs")
    block_paths, rescored, state_report = set(), [], {}
    for method, saved in zip(summary["methods"], results["methods"], strict=True):
        require(len(method["blocks"]) == 16, "incomplete method blocks")
        states = {}
        for i, item in enumerate(method["blocks"]):
            start, stop = FIRST + i*CHUNK, FIRST + (i+1)*CHUNK
            require(item["frame_range"] == [start, stop], "block interval differs")
            path = (comparison_root / item["path"]).resolve()
            require(path.is_relative_to(comparison_root) and str(path) in evidence.hashes
                    and evidence.hashes[str(path)] == item["sha256"], "block not comparison-bound")
            block_paths.add(str(path))
            with np.load(path, allow_pickle=False) as stored:
                source, observed = stored["sources"], stored["observed"]
                label = stored["original_label_available"]
                review = stored["original_identity_review"]
                require(np.array_equal(stored["frame_idx"], np.arange(start, stop))
                        and stored["frame_idx"].dtype == np.int64, "native frame mismatch")
            require(source.shape == (CHUNK, 3, 3, 3) and source.dtype.kind == "i"
                    and np.all(source >= -1), "source axes/schema mismatch")
            require(observed.shape == (CHUNK, 3, 3) and observed.dtype == bool
                    and np.array_equal(observed, (source >= 0).sum(axis=2) >= 2),
                    "support mismatch")
            require(label.shape == review.shape == (CHUNK, 3)
                    and label.dtype == review.dtype == bool, "acceptance schema mismatch")
            require(np.array_equal(label, ~review) if method["name"] == "new" else not review.any(),
                    "original acceptance interpretation differs")
            for f, g in ((0, 1), (0, 2), (1, 2)):
                require(not np.any((source[:, f] >= 0) & (source[:, f] == source[:, g])),
                        "source shared between slots")
            for anchor in anchors:
                frame = anchor["frame"]
                if start <= frame < stop:
                    t = frame - start
                    for v, view in enumerate(VIEWS):
                        for row in source[t, :, v, 0]:
                            require(row < 0 or (row < len(pools[view][frame])
                                    and np.isfinite(pools[view][frame][row]).all()),
                                    "declared anchor mouth source absent in original SL")
                    states[frame] = dict(sources=source[t].copy(), observed=observed[t].copy(),
                                         label=label[t].copy(), review=review[t].copy())
        pair = {}
        for key, accepted in (("strict_raw", False), ("strict_with_original_acceptance", True)):
            score = independent_score(states, anchors, accepted)
            require(all(score[k] == saved[key][k] for k in score), "saved female score differs")
            pair[key] = score
        rescored.append(dict(name=method["name"], **pair))
        state_report[method["name"]] = {
            str(f): {k: a.tolist() for k, a in state.items()} for f, state in states.items()}
    require(len(block_paths) == 128, "not all 128 distinct blocks verified")
    report = dict(
        status="ok", schema="independent_female_160s_audit_v1", frame_range=[FIRST, STOP],
        requested_anchors=len(anchors), requested_adjacent_intervals=max(0, len(anchors)-1),
        anchor_native_frames=[a["frame"] for a in anchors],
        geometry_blocks=128, methods=rescored, source_axis="T, raw slot, view, node; mouth node=0",
        evaluation_seal_sha256=EVAL_SEAL_SHA, evaluation_summary_sha256=sha(root / "summary.json"),
        anchor_matches_exact=True, independent_scoring_used=True,
        geometric_ray_gates_rerun=False, raw_images_decoded=0, GPU_used=False,
        limitations=["Sparse female endpoints only; no male or dense identity truth.",
                     "P cannot rule out returning changes; S counts intervals, not events.",
                     "New160s trajectory is provisional; C34 uses whole-recording future context.",
                     "Strict geometry is inherited from the sealed comparison; this audit checks "
                     "its hashes/source flags, not a second full ray projection."])
    with provenance("independent_female_160s_audit", output, name="audit") as record:
        for path, digest in evidence.hashes.items():
            record.input(path, path, sha256=digest)
        target = output / Path(__file__).name
        shutil.copyfile(__file__, target)
        require(sha(target) == evidence.hashes[str(Path(__file__).resolve())], "audit copy changed")
        record.output("audit_source", target, sha256=sha(target))
        for name, value in (("summary", report), ("independent_anchor_matches", anchors),
                            ("anchor_view_diagnostics", match_diagnostics),
                            ("anchor_source_acceptance", state_report)):
            path = output / (name + ".json")
            path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
            record.output(name, path, sha256=sha(path))
        lines = ["# 固定160秒・メス端点の独立監査", "", "全24アンカーを元CSV/SLへ再対応し、"
                 "128保存blockから全8方式のraw/元受理を別計算。保存結果と一致。", "",
                 "|方式|P|S|U|", "|---|---:|---:|---:|"]
        for method in rescored:
            c = method["strict_raw"]["interval_counts"]
            lines.append(f"|{method['name']}|{c['preserved']}|{c['swap']}|{c['unscored']}|")
        require(all(x["strict_raw"] == x["strict_with_original_acceptance"] for x in rescored),
                "report cannot describe unequal raw/accepted results as equal")
        lines.extend(["", "今回はrawと元受理を適用した表が同一。24端点/23隣接区間が分母。",
                      "", *report["limitations"]])
        path = output / "RESULT.md"
        path.write_text("\n".join(lines)+"\n", encoding="utf-8")
        record.output("report", path, sha256=sha(path))
        evidence.guard()
        require(all(sha(x["path"]) == x["sha256"] for x in record.outputs.values()),
                "audit output changed")
        record.param("all_opened_input_source_output_sha_unchanged", True)
    print(json.dumps(dict(status="ok", anchors=len(anchors), blocks=128, output=str(output))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.evaluation_root, args.output)
