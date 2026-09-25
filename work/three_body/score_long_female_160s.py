"""Sparse female endpoint diagnostics from a sealed all-seven geometry comparison.

No fitting, identity correction, raw decoding, or new human labels. All adjacent
requested anchors remain in the denominator, including unmatched observations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from long_baseline_support import (  # noqa: E402
    CATALOG_SHA256,
    EXPERIMENT,
    LAMBDAS,
    RegisteredNodes,
    relative,
)

from medaka3d.eval.generative import match_mouth_anchors, score_female_identity  # noqa: E402
from medaka3d.provenance import provenance  # noqa: E402
from medaka3d.types import VIEWS  # noqa: E402

FIRST, STOP, CHUNK = 49800, 76360, 1660
METHODS = ("new", *("c34_lambda_" + v for v in LAMBDAS))
COMPARE_STAGE = "long_160s_all7_geometry_support_comparison"
GT_RELATIVE = f"assets/groundtruth/fish_sex_gt/{EXPERIMENT}/female_sex_gt.csv"
GT_SHA = "ab0a173ac737a63311aec834c5832fc19475d4f7d1892b8193f8465bba67519f"
NEW_CANDIDATE = (
    REPO / "work/three_body/long_remaining_supervisor_v1/phases/"
    "f066400-076360/tracking/candidate.npz"
)
NEW_CANDIDATE_SHA = "870836f9e086db6d1675dd283846e758038fb942fb6e8df6296f63aad98a5877"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Evidence:
    def __init__(self):
        self.hashes = {}

    def bind(self, path, expected=None):
        path = Path(path).resolve(strict=True)
        require(path.is_relative_to(REPO), "input outside workspace")
        digest = sha(path)
        require(expected is None or digest == expected, "input SHA mismatch: " + str(path))
        require(self.hashes.get(str(path), digest) == digest, "conflicting input SHA")
        self.hashes[str(path)] = digest
        return path

    def read(self, path, expected=None):
        return json.loads(self.bind(path, expected).read_text(encoding="utf-8-sig"))

    def guard(self):
        for path, digest in self.hashes.items():
            require(sha(path) == digest, "input changed during endpoint evaluation")


def validate_methods(summary):
    require(summary.get("schema") == COMPARE_STAGE + "_v1", "wrong comparison schema")
    methods = summary["methods"]
    require(
        tuple(row["name"] for row in methods) == METHODS, "all seven ordered baselines required"
    )
    for row in methods:
        require(len(row["blocks"]) == (STOP - FIRST) // CHUNK, "partial method comparison")
        for index, block in enumerate(row["blocks"]):
            require(
                block["frame_range"] == [FIRST + CHUNK * index, FIRST + CHUNK * (index + 1)],
                "wrong native block interval",
            )
    return methods


def validate_block(data, start, stop, *, method):
    frames, source, observed, label, review = (
        data[k]
        for k in (
            "frame_idx",
            "sources",
            "observed",
            "original_label_available",
            "original_identity_review",
        )
    )
    n = stop - start
    require(
        frames.dtype == np.int64 and np.array_equal(frames, np.arange(start, stop)),
        "wrong native block frames",
    )
    require(
        source.shape == (n, 3, 3, 3) and source.dtype.kind == "i" and np.all(source >= -1),
        "invalid strict sources",
    )
    require(
        observed.shape == (n, 3, 3)
        and observed.dtype == bool
        and np.array_equal(observed, (source >= 0).sum(2) >= 2),
        "wrong strict support flags",
    )
    require(
        label.shape == (n, 3)
        and label.dtype == bool
        and review.shape == (n, 3)
        and review.dtype == bool,
        "wrong original acceptance masks",
    )
    require(
        (np.array_equal(label, ~review) if method == "new" else not review.any()),
        "original review semantics differ",
    )
    # Duplicate declarations remain a failure even when the node is unsupported.
    for i, j in ((0, 1), (0, 2), (1, 2)):
        require(
            not np.any((source[:, i] >= 0) & (source[:, i] == source[:, j])),
            "strict source shared by fish",
        )
    return frames, source[..., 0], observed[..., 0], label


def score_blocks(blocks, anchors, *, method):
    require(
        method in METHODS and len(blocks) == (STOP - FIRST) // CHUNK, "wrong method block count"
    )
    checked = [
        validate_block(block, FIRST + i * CHUNK, FIRST + (i + 1) * CHUNK, method=method)
        for i, block in enumerate(blocks)
    ]
    frames, source, observed, label = (
        np.concatenate([row[k] for row in checked]) for k in range(4)
    )
    return dict(
        name=method,
        strict_raw=score_female_identity(frames, source, observed, anchors),
        strict_with_original_acceptance=score_female_identity(
            frames, source, observed, anchors, review=~label
        ),
    )


def evaluate(comparison_root, input_root, output):
    comparison_root, input_root, output = (
        Path(p).resolve() for p in (comparison_root, input_root, output)
    )
    require(output.is_relative_to(REPO) and not output.exists(), "new workspace output required")
    evidence = Evidence()
    seal = evidence.read(comparison_root / "comparison.provenance.json")
    require(
        seal.get("status") == "ok" and seal.get("stage") == COMPARE_STAGE,
        "successful geometry comparison required",
    )
    require(
        seal["params"].get("all_used_input_source_snapshot_output_sha_unchanged") is True,
        "comparison final guard missing",
    )
    source_paths = [
        Path(__file__).resolve(),
        REPO / "work/three_body/long_baseline_support.py",
        REPO / "src/medaka3d/eval/generative.py",
        REPO / "src/medaka3d/types.py",
        REPO / "src/medaka3d/provenance.py",
        REPO / "src/medaka3d/calib.py",
        REPO / "work/three_body/FEMALE_ENDPOINT_160S_PLAN.md",
    ]
    for path in source_paths:
        evidence.bind(path)
    sealed_outputs = {}
    for item in seal["outputs"].values():
        path = evidence.bind(item["path"], item["sha256"])
        require(
            path.is_relative_to(comparison_root) and path.stat().st_size == item["size"],
            "comparison output escaped or changed size",
        )
        sealed_outputs[path] = item["sha256"]
    summary_path = comparison_root / "summary.json"
    require(summary_path in sealed_outputs, "comparison summary is unsealed")
    summary = evidence.read(summary_path, sealed_outputs[summary_path])
    for key, value in dict(
        status="ok",
        frame_range=[FIRST, STOP],
        native_frames=STOP - FIRST,
        block_frames=CHUNK,
        candidate_sha256=NEW_CANDIDATE_SHA,
    ).items():
        require(
            type(summary.get(key)) is type(value) and summary[key] == value,
            "comparison summary differs: " + key,
        )
    methods = validate_methods(summary)
    catalog_path = input_root / "staging_manifest.json"
    catalog = evidence.read(catalog_path, CATALOG_SHA256)
    entries = {relative(item["relative"]): item for item in catalog["files"]}
    require(len(entries) == len(catalog["files"]), "duplicate catalog paths")
    compared_inputs = {
        str(Path(item["path"]).resolve()): item["sha256"] for item in seal["inputs"].values()
    }
    require(
        compared_inputs.get(str(NEW_CANDIDATE)) == NEW_CANDIDATE_SHA,
        "geometry comparison uses another provisional candidate",
    )
    evidence.bind(NEW_CANDIDATE, NEW_CANDIDATE_SHA)
    shared = summary["shared_SL_and_calibration_sha256"]
    require(isinstance(shared, dict) and len(shared) == 7, "shared SL/calibration scope differs")
    calibration = []
    for name, digest in shared.items():
        path = evidence.bind(name, digest)
        require(compared_inputs.get(str(path)) == digest, "shared input not in comparison seal")
        if path.suffix == ".json":
            calibration.append(path)
    require(
        len(calibration) == 4
        and sum(p.name == "calibration_manifest.json" for p in calibration) == 1,
        "four shared calibration inputs required",
    )

    def registered(name, *, compared=True):
        item = entries[name]
        path = evidence.bind(input_root / name, item["sha256"])
        require(path.stat().st_size == item["bytes"], "registered input size mismatch")
        require(
            not compared or compared_inputs.get(str(path)) == item["sha256"],
            "not the SLEAP input of the geometry comparison",
        )
        return path

    gt = registered(GT_RELATIVE, compared=False)
    require(evidence.hashes[str(gt)] == GT_SHA, "wrong female annotation version")
    paths = {
        v: registered(f"work/c01_s16_predictions/bottomup/{EXPERIMENT}__{v}.npz") for v in VIEWS
    }
    require(
        all(shared.get(str(p)) == evidence.hashes[str(p)] for p in paths.values()),
        "loaded SL differs from declared shared calibration context",
    )
    nodes = RegisteredNodes(paths, np.arange(FIRST, STOP, dtype=np.int64))
    with gt.open(encoding="utf-8-sig", newline="") as stream:
        anchors = match_mouth_anchors(
            csv.DictReader(stream), lambda v, f: nodes.at(v, f)[:, 0], FIRST, STOP, match_px=5.0
        )
    results = []
    for method in methods:
        blocks = []
        for item in method["blocks"]:
            path = (comparison_root / relative(item["path"])).resolve(strict=True)
            require(
                path in sealed_outputs and sealed_outputs[path] == item["sha256"],
                "method block is not the sealed comparison output",
            )
            with np.load(path, allow_pickle=False) as data:
                blocks.append(
                    {
                        k: data[k]
                        for k in (
                            "frame_idx",
                            "sources",
                            "observed",
                            "original_label_available",
                            "original_identity_review",
                        )
                    }
                )
        results.append(score_blocks(blocks, anchors, method=method["name"]))
    result = dict(
        schema="long_female_endpoints_160s_v1",
        frame_range=[FIRST, STOP],
        source_seconds=160,
        requested_anchors=len(anchors),
        comparison_seal_sha256=evidence.hashes[str(comparison_root / "comparison.provenance.json")],
        methods=results,
        identities_repaired=False,
        all_three_identity_accuracy_measured=False,
        full_sequence_complete=False,
        prefix_backtrace_provisional=True,
        scope="Sparse female endpoints; S counts intervals with different endpoint slots.",
        limitations=[
            "No male or dense identity truth. P can contain returning swaps.",
            "Unmatched and reviewed anchors remain U; no bridging over unknown anchors.",
            "C34 uses whole-recording future information and has derived mouth geometry.",
            "All seven settings retained; no best lambda or slot permutation selected.",
        ],
    )
    with provenance("long_female_endpoint_evaluation_160s", output, name="evaluation") as record:
        for path, digest in evidence.hashes.items():
            record.input(path, path, sha256=digest)
        for path in source_paths:
            target = output / "source_snapshot" / path.relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            require(sha(target) == evidence.hashes[str(path)], "source snapshot changed")
            record.output(str(target.relative_to(output)), target, sha256=sha(target))
        copied = output / "original_female_sex_gt.csv"
        shutil.copyfile(gt, copied)
        require(sha(copied) == GT_SHA, "annotation copy differs")
        record.output("original_annotations", copied, sha256=sha(copied))
        for name, value in (("summary", result), ("anchor_matches", [asdict(a) for a in anchors])):
            path = output / (name + ".json")
            path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
            record.output(name, path, sha256=sha(path))
        evidence.guard()
        for item in record.outputs.values():
            require(sha(item["path"]) == item["sha256"], "evaluation output changed")
        record.param("all_input_source_output_sha_unchanged", True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("comparison-root", "input-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.comparison_root, args.input_root, args.output)
    print(json.dumps({"anchors": result["requested_anchors"], "methods": result["methods"]}))


if __name__ == "__main__":
    main()
