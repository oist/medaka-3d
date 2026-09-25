"""CPU-only direct-source support audit of the completed first 1660-frame prefix.

Reuse only verify_all_support/direct_counts from the prior auditor. Its CLI,
comparison loader, human annotations and old candidates are never evaluated.
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path

import audit_persistent_minute as existing
import numpy as np

from medaka3d.provenance import provenance

BASE = existing.base
REPO = BASE.REPO
LAB = BASE.LAB
RUN = REPO / "work/three_body/long_continuation_first1660_v1"
OUT = REPO / "work/three_body/long_continuation_first1660_support_audit_v2"
FRAMES = np.arange(49800, 51460, dtype=np.int64)
CANDIDATE_SHA = "1226f750494069784cba7abf7f1f24e9b37d21dd0b912ae91beea60dcf133f84"


def require(value, message):
    if not value:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def code_paths():
    paths = {Path(__file__).resolve(), Path(existing.__file__).resolve()}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == ".py" and (
                path.is_relative_to(LAB) or path.is_relative_to(REPO / "src/medaka3d")
            ):
                paths.add(path)
    return sorted(paths)


class Digests:
    """One full SHA per file identity in each before/after pass, including raw aliases."""

    def __init__(self):
        self.paths = {}
        self.canonical = {}

    def add(self, path, expected=None):
        path = Path(path).resolve(strict=True)
        require(path.is_relative_to(REPO), "audit input is outside the workspace")
        stat = path.stat()
        identity = stat.st_dev, stat.st_ino
        if identity not in self.canonical:
            self.canonical[identity] = (str(path), BASE.sha256(path))
        canonical, actual = self.canonical[identity]
        require(os.path.samefile(path, canonical), "file identity collision")
        require(expected is None or actual == expected, "SHA mismatch: " + str(path))
        require(self.paths.get(str(path), actual) == actual, "conflicting file binding")
        self.paths[str(path)] = actual
        return actual

    def final_guard(self):
        for canonical, expected in self.canonical.values():
            require(BASE.sha256(canonical) == expected, "file changed during support audit")
        for path, expected in self.paths.items():
            stat = Path(path).stat()
            canonical, digest = self.canonical[(stat.st_dev, stat.st_ino)]
            require(os.path.samefile(path, canonical) and digest == expected, "alias changed")


def verify_run(evidence):
    seal_path = RUN / "continuation.provenance.json"
    evidence.add(seal_path)
    seal = read_json(seal_path)
    require(
        seal.get("status") == "ok" and seal.get("stage") == "three_body_long_continuation",
        "completed long prefix seal is required before opening arrays",
    )
    params = seal["params"]
    require(
        params["through_frame"] == 51460
        and params["resume_phase"] is None
        and params["all_input_source_snapshot_output_sha_unchanged"] is True,
        "unexpected prefix or incomplete producer guards",
    )
    for category in ("inputs", "outputs"):
        for key, item in seal[category].items():
            path = Path(item["path"]).resolve(strict=True)
            named = {
                "candidate": "candidate.npz",
                "summary": "summary.json",
                "progress": "progress.jsonl",
                "consumed_chunks": "consumed_chunks.json",
                "checkpoint_manifest": "checkpoint/manifest.json",
            }
            target = RUN / named[key] if category == "outputs" and key in named else Path(key)
            require(target.resolve() == path, "producer role path mismatch")
            if category == "outputs":
                require(path.is_relative_to(RUN), "producer output escaped run")
            evidence.add(path, item["sha256"])
            require(path.stat().st_size == item["size"], "producer size mismatch")
    binding = params["bindings"]
    for source, expected in binding["code_sha256"].items():
        evidence.add(source, expected)
        snapshot = RUN / "code_snapshot" / Path(source).relative_to(REPO)
        require(str(snapshot) in seal["outputs"], "producer source snapshot absent")
        evidence.add(snapshot, expected)
    require(binding["inputs"]["full_frame_range"] == [49800, 348600], "full target changed")
    seed = binding["inputs"]["seed"]
    require(
        seed["registration_sha256"]
        == "34da64c0c61a4c4e589b3e38a01627baff0e934b1d1a4325a280645e8a132b40",
        "unexpected first-frame registration",
    )
    evidence.add(seed["registration_path"], seed["registration_sha256"])
    input_root = Path(seed["input_root"]).resolve(strict=True)
    require(len(seed["SL_and_calibration_sha256"]) == 7, "missing SL/calibration evidence")
    for path, digest in seed["SL_and_calibration_sha256"].items():
        require(
            str(Path(path).resolve()) in seal["inputs"], "support evidence not a producer input"
        )
        evidence.add(path, digest)
    expected_sl = {
        str(input_root / f"work/c01_s16_predictions/bottomup/{BASE.EXPERIMENT}__{view}.npz")
        for view in BASE.VIEWS
    }
    actual_sl = {p for p in seed["SL_and_calibration_sha256"] if p.endswith(".npz")}
    require(expected_sl == actual_sl, "SL view or source path mismatch")
    calibration = input_root / BASE.CALIBRATION_DIR
    require(
        {str(p) for p in calibration.glob("*.json")}
        == {p for p in seed["SL_and_calibration_sha256"] if p.endswith(".json")},
        "actual calibration files differ from input binding",
    )
    consumed = read_json(RUN / "consumed_chunks.json")
    require(
        len(consumed) == 1 and consumed[0]["start"] == 49800 and consumed[0]["stop"] == 51460,
        "consumed chunk prefix differs",
    )
    stage = Path(binding["inputs"]["input_stage_root"])
    chunk = stage / consumed[0]["attempt_path"]
    evidence.add(chunk / "manifest.json", consumed[0]["manifest_sha256"])
    evidence.add(chunk / "staging.provenance.json", consumed[0]["provenance_sha256"])
    require(read_json(chunk / "staging.provenance.json")["status"] == "ok", "chunk incomplete")
    for item in consumed[0]["output_hashes"].values():
        evidence.add(item["path"], item["sha256"])
    for item in consumed[0]["raw_hardlinks"].values():
        require(
            os.path.samefile(item["path"], item["source_path"]), "consumed raw hardlink differs"
        )
    summary = read_json(RUN / "summary.json")
    require(
        summary["technical_completion"] is True
        and summary["processed_interval"] == [49800, 51460]
        and summary["processed_total"] == 1660
        and summary["prefix_backtrace_provisional"] is True
        and summary["full_sequence_complete"] is False,
        "prefix status differs",
    )
    require(evidence.add(RUN / "candidate.npz", CANDIDATE_SHA) == CANDIDATE_SHA, "wrong candidate")
    return summary, input_root, calibration, seal


def main():
    require(not OUT.exists(), "refusing to overwrite support audit")
    evidence = Digests()
    code = code_paths()
    for path in code:
        evidence.add(path)
    OUT.mkdir(parents=True, exist_ok=False)
    with provenance("long_continuation_direct_support_independent_audit", OUT, name="audit") as rec:
        started = time.perf_counter()
        summary, input_root, calibration, producer = verify_run(evidence)
        with np.load(RUN / "candidate.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        require(len(arrays) == 21, "expected original 21 arrays")
        require(np.array_equal(arrays["frame_idx"], FRAMES), "native frame sequence differs")
        require(
            arrays["pose_xyz"].shape == (1660, 3, 3, 3) and np.isfinite(arrays["pose_xyz"]).all(),
            "latent pose shape/finite check",
        )
        require(
            arrays["sources"].shape == (1660, 3, 3, 3) and arrays["sources"].dtype.kind == "i",
            "source schema",
        )
        require(
            arrays["observed"].shape == (1660, 3, 3) and arrays["observed"].dtype == bool,
            "observed schema",
        )
        require(
            arrays["identity_review"].shape == (1660, 3)
            and arrays["identity_review"].dtype == bool,
            "review schema",
        )
        for view in range(3):
            for node in range(3):
                for sources in arrays["sources"][:, :, view, node]:
                    present = sources[sources >= 0]
                    require(len(present) == len(np.unique(present)), "source exclusivity violated")
        nodes = existing.old.CachedNodes(input_root, BASE.EXPERIMENT, FRAMES)
        rig = existing.old.load_rig(calibration)
        require(tuple(rig.views) == tuple(BASE.VIEWS), "actual camera order differs")
        geometry, support = existing.verify_all_support(arrays, nodes, rig)
        counts = existing.direct_counts(arrays)
        require(
            counts["direct_pectoral_fish_frames"] == summary["direct_pectoral_count"]
            and counts["accepted_pectoral_fish_frames"] == summary["accepted_pectoral_count"]
            and counts["review_fish_frames"] == summary["identity_review_count"],
            "support or review counts differ from producer summary",
        )
        require(
            counts["direct_pectoral_per_slot"] == summary["per_slot_direct_pectoral"]
            and counts["accepted_pectoral_per_slot"] == summary["per_slot_accepted_pectoral"],
            "per-slot count differs",
        )
        result = {
            "schema": "long_continuation_direct_support_audit_v1",
            "status": "ok",
            "run_path": str(RUN),
            "candidate_sha256": CANDIDATE_SHA,
            "producer_seal_sha256": evidence.paths[str(RUN / "continuation.provenance.json")],
            "frame_range": [49800, 51460],
            "native_frames": 1660,
            "native_fps": 166,
            "full_sequence_complete": False,
            "prefix_backtrace_provisional": True,
            "all_node_source_geometry": geometry,
            "counts": counts,
            "source_exclusivity_all_views_nodes": True,
            "independent_ID_truth": False,
            "producer_input_files": len(producer["inputs"]),
            "producer_output_files": len(producer["outputs"]),
            "source_calibration_binding": str(calibration),
            "support_math": "Perpendicular ray distance with the original float32 origin/direction "
            "cache, then float64 residual; native refractive projection in float64; "
            "ray <= 2 mm and pixel <= 12; observed node requires >= 2 views.",
            "limits": [
                "Only declared nonnegative source assignments are verified. Absent assignments "
                "are not inferred from the latent pose or treated as observed.",
                "Current geometry support and unflagged acceptance are not identity accuracy. "
                "No human labels, baseline coordinates, old candidates, GPU or tracking were used.",
                "Full checkpoint/history/previous-image validation is delegated to the parallel "
                "main audit. This result covers direct geometry on the provisional prefix path.",
            ],
            "elapsed_seconds_before_final_SHA": time.perf_counter() - started,
        }
        np.savez_compressed(OUT / "native_support.npz", **support)
        (OUT / "summary.json").write_text(
            json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
        )
        for path in code:
            target = OUT / "code_snapshot" / path.relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            evidence.add(target, evidence.paths[str(path)])
        report = (
            "# Direct support of the completed first 1660-frame prefix\n\n"
            f"All {geometry['source_rows_checked']:,} declared per-view node assignments "
            f"passed the fixed 2 mm ray and 12 px native projection gates. "
            f"The saved {geometry['observed_nodes_checked']:,} observed node flags exactly "
            "match the requirement for at least two supporting views. Sources remain "
            "exclusive per view and node.\n\n"
            "Maximum supported ray distance: "
            f"{geometry['maximum_supported_ray_distance_mm']:.9f} mm. "
            "Maximum supported pixel error: "
            f"{geometry['maximum_supported_pixel_error']:.9f} px.\n\n"
            f"Direct pectorals: {counts['direct_pectoral_fish_frames']}/4980; "
            f"accepted pectorals: {counts['accepted_pectoral_fish_frames']}/4980; "
            f"review flags: {counts['review_fish_frames']}/4980. These are availability and "
            "internal review measures, not biological identity accuracy.\n\n"
            "This is the provisional 49800:51460 prefix only. Future observations may "
            "change its winning backtrace; the full thirty-minute track is incomplete. "
            "No tracker, GPU, baseline coordinate comparison or human annotation was used.\n"
        )
        (OUT / "RESULT.md").write_text(report, encoding="utf-8")
        for path, digest in evidence.paths.items():
            if not Path(path).is_relative_to(OUT):
                rec.input(path, path, sha256=digest)
        outputs = {}
        for path in OUT.rglob("*"):
            if path.is_file():
                outputs[str(path)] = BASE.sha256(path)
                rec.output(path.relative_to(OUT).as_posix(), path, sha256=outputs[str(path)])
        evidence.final_guard()
        require(
            all(BASE.sha256(path) == digest for path, digest in outputs.items()), "output changed"
        )
        rec.param("all_input_source_snapshot_output_sha_unchanged", True)
        rec.param("gpu_used", False)
        rec.param("geometry_refit", False)
        rec.param("human_or_GT_inputs", False)
    print(
        json.dumps({"status": "ok", "output": str(OUT), "geometry": geometry, "counts": counts}),
        flush=True,
    )


if __name__ == "__main__":
    main()
