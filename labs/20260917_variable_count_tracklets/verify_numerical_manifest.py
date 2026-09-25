"""Verify an explicit numerical artifact allowlist without touching prior records."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from medaka3d.aquarium import sha256
from medaka3d.provenance import provenance

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = ROOT / "work/variable_count_tracklets_v1"
VIEWS = ("top_cam", "side1_cam", "side2_cam")
NUMERICAL_CODE = ("baseline.py", "frozen_p0.py", "tracking.py", "run.py")


def allowed_artifacts(protocol):
    paths = ["protocol.json", "metrics.json", "summary.json", "metrics.csv", "strata.csv"]
    paths.extend("source_snapshot/" + name for name in NUMERICAL_CODE)
    for case in protocol["cases"]:
        for view in VIEWS:
            names = [f"detections_{view}.npz", f"metrics_{view}.json"]
            for method in protocol["conditions"]:
                stem = f"{method}_{view}"
                names.extend(
                    [
                        f"tracking_{stem}.npz",
                        f"tracking_{stem}.json",
                        f"metrics_{stem}.json",
                        f"timeline_{stem}.npz",
                    ]
                )
                for table in ("edges", "events"):
                    names.extend(f"{table}_{stem}.{suffix}" for suffix in ("json", "csv"))
                if method in ("S1", "S2"):
                    names.append(f"readout_audit_{stem}.npz")
                if method.startswith("P1"):
                    for table in ("predictions", "reconnections", "node_quality"):
                        names.extend(f"{table}_{stem}.{suffix}" for suffix in ("json", "csv"))
            paths.extend(case["name"] + "/" + name for name in names)
    assert len(paths) == len(set(paths))
    return sorted(paths)


def main():
    destination = OUTPUT / "numerical_manifest_verified.json"
    if destination.exists():
        raise FileExistsError("Refusing to replace the verified manifest")
    prior_path = OUTPUT / "numerical_manifest.json"
    summary_record = OUTPUT / "variable_count_summary.provenance.json"
    preserved = {str(path): sha256(path) for path in (prior_path, summary_record)}
    protocol = json.loads((OUTPUT / "protocol.json").read_text(encoding="utf-8"))
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    prior = {key.replace("\\", "/"): value for key, value in prior.items()}
    with provenance("verified_numerical_manifest", OUTPUT) as rec:
        rec.input("verification_code", Path(__file__), sha256=sha256(Path(__file__)))
        rec.input("initial_manifest", prior_path, sha256=preserved[str(prior_path)])
        rec.input(
            "initial_summary_provenance", summary_record, sha256=preserved[str(summary_record)]
        )
        artifacts = {}
        for relative in allowed_artifacts(protocol):
            path = OUTPUT / relative
            actual = sha256(path)
            expected = prior[relative]
            if actual != expected:
                raise AssertionError(
                    f"Numerical artifact changed since initial snapshot: {relative}"
                )
            artifacts[relative] = {
                "sha256": actual,
                "size_bytes": path.stat().st_size,
                "matches_initial_numerical_manifest": True,
            }
        frozen_code = {}
        for name in NUMERICAL_CODE:
            path = HERE / name
            expected = protocol["code_sha256"][name]
            actual = sha256(path)
            if actual != expected or artifacts[f"source_snapshot/{name}"]["sha256"] != expected:
                raise AssertionError(f"Frozen numerical source changed: {name}")
            frozen_code[name] = {"sha256": actual, "matches_protocol_and_snapshot": True}
            rec.input("frozen_code_" + name, path, sha256=actual)
        sources = {}
        for view in VIEWS:
            record_path = OUTPUT / f"run_{view}.provenance.json"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record["status"] != "ok":
                raise AssertionError(f"Numerical run did not finish successfully: {view}")
            rec.input("run_record_" + view, record_path, sha256=sha256(record_path))
            for role in ("raw", "saved_c27"):
                entry = record["inputs"][role]
                path = Path(entry["path"])
                actual = sha256(path)
                if actual != entry["sha256"]:
                    raise AssertionError(f"Raw input source changed: {view} {role}")
                key = f"{view}_{role}"
                sources[key] = {
                    "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "sha256": actual,
                    "matches_original_run_input": True,
                }
                rec.input(key, path, sha256=actual)
        if any(sha256(Path(path)) != digest for path, digest in preserved.items()):
            raise AssertionError("Historical manifest or summary provenance was altered")
        result = {
            "status": "verified",
            "verified_at_utc": datetime.now(UTC).isoformat(),
            "scope": "Explicit immutable numerical filenames only; no recursive directory scan",
            "includes": [
                "protocol, numerical summary/metrics/strata",
                "all 12 cases x 3 views x 8 conditions: original detection and tracking artifacts",
                "numerical evidence tables and timelines",
                "four protocol-frozen numerical code snapshots",
                "three raw C01 and three saved C27 input references",
            ],
            "excludes": [
                "HTML, media, graphs, derived review reports, browser state, test logs",
                "report/render/audit source snapshots and subsequently generated artifacts",
            ],
            "artifact_count": len(artifacts),
            "all_artifact_hashes_verified": True,
            "all_artifacts_match_initial_manifest": True,
            "all_frozen_numerical_code_matches_protocol": True,
            "all_raw_sources_match_original_run_provenance": True,
            "historical_records_preserved": preserved,
            "historical_manifest_role": (
                "Initial snapshot produced at numerical summary time; preserved unchanged. "
                "Use this verified explicit allowlist for the immutable numerical artifact scope."
            ),
            "artifacts": artifacts,
            "frozen_numerical_code": frozen_code,
            "input_references": sources,
        }
        destination.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        rec.param("artifact_count", len(artifacts))
        rec.param("scope", result["scope"])
        rec.param("historical_records_unchanged", True)
        rec.output("verified_manifest", destination, sha256=sha256(destination))
    print(f"Verified {len(artifacts)} immutable numerical artifacts and 6 raw input references")


if __name__ == "__main__":
    main()
