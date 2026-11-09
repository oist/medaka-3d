"""Export one frozen experimental identity candidate without combining candidates.

This is an artifact-validation/export helper, not a promoted identity algorithm.
The separate S and G commands supply the candidate; no CLI flag switches it.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from medaka3d.provenance import provenance


def _validate(payload: bytes) -> int:
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        names = archive["name"]
        xyz = archive["xyz_by_identity"]
        times = archive["recording_time_s"]
        labels = archive["identity_labels"]
    if names.ndim != 2 or names.shape[1] != 3 or names.dtype.kind not in "iu":
        raise ValueError("name must be an integer (frames, 3 slots) array")
    n = len(names)
    if n < 2 or xyz.shape != (n, 3, 3) or times.shape != (n,) or labels.shape != (3,):
        raise ValueError("Candidate arrays have incompatible dimensions")
    if not np.isin(names, [-1, 0, 1, 2]).all():
        raise ValueError("Identity codes must be -1 (withheld), 0, 1, or 2")
    if labels.dtype.kind not in "US" or len(set(labels.tolist())) != 3:
        raise ValueError("identity_labels must contain three distinct strings")
    if not np.isfinite(times).all() or not (np.diff(times) > 0).all():
        raise ValueError("Recording times must be finite and strictly increasing")
    if (names[times < 300] >= 0).any():
        raise ValueError("Candidate must withhold identity before recording time 300 s")
    if np.isinf(xyz).any():
        raise ValueError("Positions may be finite or NaN, never infinite")
    for code in range(3):
        count = (names == code).sum(axis=1)
        if (count > 1).any():
            raise ValueError("A frame assigns the same identity to multiple slots")
        if not np.isnan(xyz[count == 0, code]).all():
            raise ValueError("An unnamed identity has nonmissing coordinates")
    return n


def export_candidate(run: Path, output: Path, *, candidate: str, expected_sha256: str) -> Path:
    """Copy exactly one checked NPZ to a new directory, with explicit identity policy.

    Read only the requested candidate. In particular, S export never opens G,
    and missing G is an error rather than a fallback to S. Source bytes and all
    withheld codes/positions are preserved, including on reference-unknown time.
    """
    if candidate not in ("S", "G"):
        raise ValueError("Only separate S and G exports are supported")
    source = Path(run).resolve(strict=True) / "evaluation" / f"identity_tracks_{candidate}.npz"
    payload = source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256.lower():
        raise ValueError(f"Frozen candidate checksum mismatch: {source}")
    n = _validate(payload)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    target = output / source.name
    with provenance(f"export_identity_{candidate.lower()}", output) as record:
        record.input("candidate", source, sha256=digest)
        code_path = Path(__file__)
        record.input(
            "export_code", code_path, sha256=hashlib.sha256(code_path.read_bytes()).hexdigest()
        )
        record.param("candidate", candidate)
        record.param("source_run_id", source.parent.parent.name)
        record.param("combine_candidates", False)
        record.param("fill_withheld_from_other_candidate", False)
        target.write_bytes(payload)
        record.output("candidate", target, sha256=digest)
        policy = {
            "schema_version": 1,
            "candidate": candidate,
            "role": "experimental_default" if candidate == "S" else "optional_comparison",
            "identity_status": "experimental_candidate_not_independently_verified",
            "source_run_id": source.parent.parent.name,
            "source_sha256": digest,
            "frames": n,
            "combine_candidates": False,
            "fill_withheld_from_other_candidate": False,
            "withheld_code": -1,
            "physical_names": "inherited_from_source_not_verified_by_export",
            "analysis_start_s": 300.0,
            "coverage": "not_certified_by_export; retain_full_post_preparation_denominator",
        }
        policy_path = output / "identity_policy.json"
        policy_path.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
        record.output("identity_policy", policy_path)
    return target


def export_cli(candidate: str, argv: list[str] | None = None) -> None:
    """Shared argument handling for two fixed-candidate entry points."""
    parser = argparse.ArgumentParser(
        description=f"Export frozen {candidate} identity tracks alone."
    )
    parser.add_argument("--run", type=Path, required=True, help="Explicit evaluated appearance run")
    parser.add_argument(
        "--sha256", required=True, help="Expected hash from the frozen run provenance"
    )
    parser.add_argument("--output", type=Path, required=True, help="New export directory")
    args = parser.parse_args(argv)
    print(export_candidate(args.run, args.output, candidate=candidate, expected_sha256=args.sha256))
