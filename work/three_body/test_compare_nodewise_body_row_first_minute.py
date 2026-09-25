"""Bounded adapter contracts; full5 metrics and upstream audits are not retested."""

from pathlib import Path

import compare_nodewise_body_row_first_minute as m
import numpy as np
import pytest

NP_LOAD = np.load


@pytest.fixture(autouse=True)
def isolation(tmp_path, monkeypatch):
    for key in ("WORK", "TAIL", "NODEWISE", "MILESTONE", "TAIL_COMPARISON", "INPUT_ROOT"):
        monkeypatch.setattr(m, key, tmp_path / key)
    monkeypatch.setattr(m, "WORK", tmp_path)
    for key, value in dict(FIRST=100, START=104, STOP=108, CHUNK=2).items():
        monkeypatch.setattr(m, key, value)

    def forbidden(*args, **kwargs):
        pytest.fail("real input or old full5 workflow reached")

    for module, names in (
        (m, ("source_code", "load_headers")),
        (
            m.metrics,
            ("load_headers", "load_candidate", "validate_candidate", "compare_arrays", "run"),
        ),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    original = Path.open

    def guard(path):
        assert Path(path).resolve().is_relative_to(tmp_path), "real file reached"

    def opened(path, *args, **kwargs):
        guard(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", opened)
    reads = []

    class Archive:
        def __init__(self, path, *args, **kwargs):
            guard(path)
            self.name = Path(path).name
            reads.append((self.name, "OPEN"))
            self.inner = NP_LOAD(path, *args, **kwargs)
            self.files = self.inner.files

        def __getitem__(self, key):
            reads.append((self.name, key))
            return self.inner[key]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.inner.close()

    monkeypatch.setattr(np, "load", Archive)
    return reads


def candidate():
    n = m.STOP - m.FIRST
    pose = np.zeros((n, 3, 3, 3))
    pose[..., 0] = np.arange(3)[None, :, None] * 20
    pose[..., 1] = np.arange(3)[None, None, :]
    return dict(
        frame_idx=np.arange(m.FIRST, m.STOP, dtype=np.int64),
        pose_xyz=pose,
        observed=np.ones((n, 3, 3), bool),
        sources=np.zeros((n, 3, 3, 3), np.int32),
        identity_review=np.zeros((n, 3), bool),
        proposal_kind=np.full(n, "predict", dtype="U22"),
    )


@pytest.mark.parametrize("damage", ["native", "shape", "nonfinite", "support"])
def test_candidate_rejects(damage):
    row = candidate()
    if damage == "native":
        row["frame_idx"][2] += 1
    elif damage == "shape":
        row["sources"] = row["sources"][:, :, :, 0]
    elif damage == "nonfinite":
        row["pose_xyz"][0, 0, 0, 0] = np.nan
    else:
        row["sources"][0, 0, :2, 0] = -1
    with pytest.raises(ValueError):
        m.validate_candidate(row)


def test_connected_sixfield_read_and_threefield_science_output(tmp_path, monkeypatch, isolation):
    a, b = candidate(), candidate()
    inverse = np.array([2, 0, 1])  # desired A-slot -> B-slot correspondence is P120
    for key in ("pose_xyz", "observed", "sources", "identity_review"):
        b[key][4:] = b[key][4:, inverse]
    b["pose_xyz"][0] += 5  # changed selected prefix does not rewrite stored history
    refs = {}
    for arm, row in (("tail", a), ("nodewise", b)):
        values = {**row, **{k: np.zeros(1) for k in m.metrics.ALL21 - set(row)}}
        path = tmp_path / f"{arm}.npz"
        np.savez_compressed(path, **values)
        refs[arm] = dict(
            candidate=dict(
                path=str(path), sha256=m.proof.sha(path), size_bytes=path.stat().st_size
            ),
            signatures=None if arm == "tail" else m.common.signature(values),
        )
    monkeypatch.setattr(m, "source_code", lambda e: {"synthetic_trust": "sources"})
    monkeypatch.setattr(
        m,
        "load_headers",
        lambda digest, e: (
            refs,
            dict(
                native_interval=[m.START, m.STOP],
                retained_interval=[m.FIRST, m.STOP],
                terminal_horizon=m.STOP,
                same_original_metadata_and39_signatures=True,
                whole_five_minute_completion_claimed=False,
            ),
        ),
    )
    metrics_constants = m.metrics.FIRST, m.metrics.START, m.metrics.STOP
    assert m.run("a" * 64, tmp_path / "out")["status"] == "ok"
    assert metrics_constants == (m.metrics.FIRST, m.metrics.START, m.metrics.STOP)
    assert isolation == [(arm + ".npz", key) for arm in m.ARMS for key in ("OPEN", *m.FIELDS)]
    report = m.proof.read_json(tmp_path / "out/summary.json")
    assert report["methods"]["tail"]["frames"] == 4
    assert report["methods"]["tail"]["adjacent_pairs_per_slot_node"] == 3
    assert report["geometric_comparison"]["chest"]["unique_winner_counts"] == [0, 0, 0, 4, 0, 0]
    assert (
        report["geometric_comparison"]["selected_pre_science_backtrace_difference_frames"][
            "pose_xyz"
        ]
        == 1
    )
    assert not report["whole_five_minute_completion_claimed"]
    with NP_LOAD(tmp_path / "out/science/nodewise.npz") as z:
        assert set(z.files) == set(m.SCIENCE_FIELDS)
        assert z["frame_idx"].tolist() == [104, 105, 106, 107]
        assert (
            m.common.signature({k: z[k] for k in z.files})
            == report["science_signatures"]["nodewise"]
        )


def test_common_prefix_rejects_a_different_sealed_manifest():
    chunks, inputs = [], {}
    for i in range(40):
        root = m.INPUT_ROOT / f"c{i}"
        chunks.append(
            dict(
                index=i,
                start=m.FIRST + i * m.CHUNK,
                stop=m.FIRST + (i + 1) * m.CHUNK,
                attempt_path=f"c{i}",
                manifest_sha256="a" * 64,
                provenance_sha256="b" * 64,
                output_hashes={},
                raw_hardlinks={},
            )
        )
        inputs[str((root / "manifest.json").resolve())] = {"sha256": "a" * 64}
        inputs[str((root / "staging.provenance.json").resolve())] = {"sha256": "b" * 64}
    index = dict(original_binding={"same": True}, prefix_chunks=chunks)
    assert len(m.verify_prefix(index, dict(inputs=inputs), index["original_binding"])) == 4
    inputs[str((m.INPUT_ROOT / "c2/manifest.json").resolve())]["sha256"] = "c" * 64
    with pytest.raises(ValueError, match="manifest lineage"):
        m.verify_prefix(index, dict(inputs=inputs), index["original_binding"])


def test_output_guard_reaches_no_real_input(tmp_path):
    with pytest.raises(ValueError, match="fresh workspace"):
        m.run("a" * 64, tmp_path)
