"""Operational separation of frozen S/G candidates and preservation of abstentions."""

import hashlib
import json

import numpy as np
import pytest

from medaka3d.identity_candidate_export import export_candidate


def candidate_file(run, candidate, names):
    folder = run / "evaluation"
    folder.mkdir(parents=True, exist_ok=True)
    names = np.array(names, dtype=np.int8)
    xyz = np.full((len(names), 3, 3), np.nan, dtype=np.float32)
    for code in range(3):
        xyz[(names == code).any(axis=1), code] = code + 1
    path = folder / f"identity_tracks_{candidate}.npz"
    np.savez_compressed(
        path,
        name=names,
        xyz_by_identity=xyz,
        recording_time_s=300 + np.arange(len(names), dtype=float),
        identity_labels=np.array(["id0", "id1", "id2"]),
    )
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_separate_exports_preserve_source_bytes_and_s_abstentions(tmp_path):
    run = tmp_path / "frozen_run"
    s, shash = candidate_file(run, "S", [[0, 1, 2], [0, -1, -1]])
    g, ghash = candidate_file(run, "G", [[0, 1, 2], [0, 2, 1]])
    for candidate, source, digest in (("S", s, shash), ("G", g, ghash)):
        dest = tmp_path / f"export_{candidate}"
        result = export_candidate(run, dest, candidate=candidate, expected_sha256=digest)
        assert result.read_bytes() == source.read_bytes()
        assert list(dest.glob("*.npz")) == [result]
        policy = json.loads((dest / "identity_policy.json").read_text())
        assert policy["candidate"] == candidate
        assert policy["fill_withheld_from_other_candidate"] is False
        receipt = json.loads(
            (dest / f"export_identity_{candidate.lower()}.provenance.json").read_text()
        )
        assert receipt["status"] == "ok"
        assert receipt["inputs"]["candidate"]["sha256"] == digest
    with np.load(tmp_path / "export_S" / s.name) as data:
        assert data["name"][1].tolist() == [0, -1, -1]
        assert np.isnan(data["xyz_by_identity"][1, 1:]).all()


def test_s_needs_no_g_and_missing_g_never_falls_back(tmp_path):
    run = tmp_path / "frozen_run"
    _, digest = candidate_file(run, "S", [[0, 1, 2], [-1, -1, -1]])
    export_candidate(run, tmp_path / "s", candidate="S", expected_sha256=digest)
    with pytest.raises(FileNotFoundError):
        export_candidate(run, tmp_path / "g", candidate="G", expected_sha256=digest)
    assert not (tmp_path / "g").exists()


def test_checksum_failure_and_existing_export_are_not_overwritten(tmp_path):
    run = tmp_path / "frozen_run"
    _, digest = candidate_file(run, "S", [[0, 1, 2], [-1, -1, -1]])
    dest = tmp_path / "export"
    with pytest.raises(ValueError, match="checksum mismatch"):
        export_candidate(run, dest, candidate="S", expected_sha256="0" * 64)
    assert not dest.exists()
    result = export_candidate(run, dest, candidate="S", expected_sha256=digest)
    before = {p.name: p.read_bytes() for p in dest.iterdir()}
    with pytest.raises(FileExistsError):
        export_candidate(run, dest, candidate="S", expected_sha256=digest)
    assert {p.name: p.read_bytes() for p in dest.iterdir()} == before
    assert result.exists()


@pytest.mark.parametrize("names", [[[0, 0, 2], [0, 1, 2]], [[0, 1, 3], [0, 1, 2]]])
def test_ambiguous_or_invalid_names_are_rejected(tmp_path, names):
    run = tmp_path / "frozen_run"
    _, digest = candidate_file(run, "S", names)
    with pytest.raises(ValueError):
        export_candidate(run, tmp_path / "out", candidate="S", expected_sha256=digest)
    assert not (tmp_path / "out").exists()
