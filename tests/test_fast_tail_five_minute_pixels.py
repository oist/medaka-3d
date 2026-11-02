"""Five-minute schedule, exact reuse, and source lineage without real mask decode."""

from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "work/three_body"))
import prepare_fast_tail_five_minute_pixels as m


def samples():
    fit = np.tile(np.column_stack((np.arange(1024), np.zeros(1024))), (3, 1, 1)).astype(float)
    unused = np.tile(np.column_stack((np.arange(256), np.ones(256))), (3, 1, 1)).astype(float)
    return dict(
        fit_pixels=fit,
        fit_target=np.zeros((3, 1024), np.float32),
        fit_weights=np.zeros((3, 1024), np.float32),
        fit_foreground=np.zeros((3, 1024), bool),
        pixels=unused,
        target=np.zeros((3, 256), np.float32),
        weights=np.zeros((3, 256), np.float32),
        reused_fitted_coordinate=np.zeros((3, 256), bool),
    )


def test_fixed300_preserves_prior120_then_exact_three_new_minutes():
    assert m.FRAMES[:120] == m.prior.FRAMES
    assert len(m.FRAMES) == 300
    assert m.FRAMES[120] == 86403 and m.FRAMES[-1] == 116117
    assert all(b - a == 166 for a, b in zip(m.FRAMES[:-1], m.FRAMES[1:], strict=True))
    chunks = range(m.REUSED_STOP, m.STOP, m.CHUNK)
    assert len(chunks) == 18
    assert [sum(f <= n < f + m.CHUNK for n in m.FRAMES[120:]) for f in chunks] == [10] * 18


@pytest.mark.parametrize("fault", ["missing_view", "duplicate_view", "order", "extra"])
def test_membership_never_filters_missing_or_neutral_views(fault):
    rows = [dict(frame=f, view=v) for f in m.FRAMES[:2] for v in m.old.VIEWS]
    m.validate_membership_order(rows, m.FRAMES[:2])
    if fault == "missing_view":
        rows.pop(1)
    elif fault == "duplicate_view":
        rows[1] = rows[0]
    elif fault == "order":
        rows[1], rows[2] = rows[2], rows[1]
    else:
        rows.append(rows[-1])
    with pytest.raises(ValueError, match="membership order"):
        m.validate_membership_order(rows, m.FRAMES[:2])


@pytest.mark.requires_artifacts(
    m.ORIGINAL / "continuation.provenance.json",
    m.PREPARED / "preparation.provenance.json",
)
def test_real_small_metadata_binds90_masks_without_mask_pose_or_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected numerical payload or CUDA")

    monkeypatch.setattr(np, "load", forbidden)
    monkeypatch.setattr(m.torch.cuda, "init", forbidden)
    evidence = m.proof.Evidence()
    original, previous, summary, config, sources = m.metadata(evidence)
    assert len(sources) == 90 and len(summary["source_mask_sha256"]) == 36
    assert original["seal"]["params"]["through_frame"] == 116200
    assert previous["seal"]["stage"] == m.prior.STAGE
    assert config["mode"] == "detectron2"
    assert not any(Path(p).suffix == ".npz" for p in evidence.paths)
    assert not m.torch.cuda.is_initialized()


@pytest.mark.parametrize("fault", ["schedule", "config", "overlap", "truth", "diagnostic"])
def test_previous_protocol_mutation_rejected(fault):
    config = dict(
        mode="detectron2",
        width_ratio=0.06,
        positive_samples=384,
        near_samples=384,
        global_samples=256,
    )
    summary = dict(
        schema=m.prior.STAGE,
        status="ok",
        native_interval=[m.START, m.REUSED_STOP],
        scored_frames=list(m.FRAMES[:120]),
        reused_frames=list(m.FRAMES[:60]),
        newly_sampled_frames=list(m.FRAMES[60:120]),
        sampler_config=copy.deepcopy(config),
        counts=[96, 96, 64],
        seed_offset=8803010,
        original_seal_sha256=m.prior.ORIGINAL_SHA,
        reused_preparation_sha256=m.prior.PREPARED_SHA,
        native_decoded_mask_checks=180,
        reused_sample_arrays_BYTEEXACT=True,
        fitted_coordinate_overlap=0,
        pixel_draws_per_arm=92160,
        fit_coordinates_reconstructed_deterministically=True,
        saved_run_fit_diagnostics_compared=False,
        pose_values_read=False,
        raw_BGR_frames_decoded=0,
        models=0,
        GPU_initialized=False,
        temporal_holdout=False,
        identity_truth=False,
    )
    m.validate_previous(summary, config)
    if fault == "schedule":
        summary["scored_frames"][119] -= 1
    elif fault == "config":
        config["global_samples"] -= 1
    elif fault == "overlap":
        summary["fitted_coordinate_overlap"] = 1
    elif fault == "truth":
        summary["identity_truth"] = True
    else:
        summary["saved_run_fit_diagnostics_compared"] = True
    with pytest.raises(ValueError, match="fixed120 protocol"):
        m.validate_previous(summary, config)


@pytest.mark.parametrize("fault", ["first_minute_sha", "second_minute_edge", "extra_mask"])
def test_reused_mask_lineage_corruption_rejected(monkeypatch, fault):
    sources = {}
    old_inputs = {}
    mapping = {}
    for first in range(m.START, m.STOP, m.CHUNK):
        for view in m.old.VIEWS:
            path = m.WORK / "inputs" / f"fake_{first}_{view}.npz"
            mapping[(first, view)] = (path, "a" * 64)
            if first < m.REUSED_STOP:
                sources[str(path)] = "a" * 64
                if first >= m.prior.HALF:
                    old_inputs[str(path)] = "a" * 64
    monkeypatch.setattr(m, "mask_input", lambda seal, first, view: mapping[first, view])
    summary, previous = {"source_mask_sha256": sources}, {"inputs": old_inputs}
    assert len(m.bind_sources({}, previous, summary)) == 90
    if fault == "first_minute_sha":
        sources[str(mapping[m.START, m.old.VIEWS[0]][0])] = "b" * 64
    elif fault == "second_minute_edge":
        old_inputs.pop(str(mapping[m.prior.HALF, m.old.VIEWS[0]][0]))
    else:
        sources["extra"] = "a" * 64
    with pytest.raises(ValueError, match=r"(mask differs|consumption differs|membership differs)"):
        m.bind_sources({}, previous, summary)


def sealed_fixture(tmp_path, frames):
    out = m.proof.Output(tmp_path / "source")
    rows = [dict(frame=f, view=v, neutral_view=True) for f in frames for v in m.old.VIEWS]
    out.write("membership.json", rows)
    for f in frames:
        out.write(f"frame_{f:06d}.npz", samples())
    seal = dict(stage=m.prior.STAGE, status="ok", inputs={}, outputs=out.records)
    p = out.root / "preparation.provenance.json"
    p.write_bytes(m.proof.json_bytes(seal))
    return out.root, seal, hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.mark.parametrize("fault", ["path_escape", "bool_size", "negative_size", "wrong_stage"])
def test_custom_seal_rejects_unsafe_members(tmp_path, fault):
    root, seal, _ = sealed_fixture(tmp_path, m.FRAMES[:1])
    row = seal["outputs"]["membership.json"]
    if fault == "path_escape":
        row["path"] = str(root.parent / "outside.json")
    elif fault == "bool_size":
        row["size_bytes"] = True
    elif fault == "negative_size":
        row["size_bytes"] = -1
    else:
        seal["stage"] = "other"
    p = root / "preparation.provenance.json"
    p.write_bytes(m.proof.json_bytes(seal))
    with pytest.raises(ValueError):
        m.custom_seal(root, hashlib.sha256(p.read_bytes()).hexdigest(), m.proof.Evidence())


def test_reuse_byte_exact_keeps_neutral_rows_and_never_resamples(tmp_path, monkeypatch):
    frames = m.FRAMES[:2]
    root, _, digest = sealed_fixture(tmp_path, frames)
    evidence = m.proof.Evidence()
    previous = m.custom_seal(root, digest, evidence)
    output = m.proof.Output(tmp_path / "copied")
    monkeypatch.setattr(m, "FRAMES", frames)
    monkeypatch.setattr(m, "REUSED_COUNT", 2)

    def forbidden(*args, **kwargs):
        raise AssertionError("reused sample must not be resampled or decoded")

    monkeypatch.setattr(m.old, "fit_arrays", forbidden)
    monkeypatch.setattr(m.old, "unused_samples", forbidden)
    monkeypatch.setattr(m.old, "decode_frame_masks", forbidden)
    rows, completed = m.reuse_samples(previous, evidence, output)
    assert completed == list(frames) and len(rows) == 6
    assert all(r["neutral_view"] for r in rows)
    for f in frames:
        with (
            np.load(root / f"frame_{f:06d}.npz") as left,
            np.load(output.root / f"frame_{f:06d}.npz") as right,
        ):
            assert left.files == right.files
            for name in left.files:
                m.proof.same(left[name], right[name], "copied sample")


def test_custom_member_detects_same_size_corruption(tmp_path):
    root, _, digest = sealed_fixture(tmp_path, m.FRAMES[:1])
    evidence = m.proof.Evidence()
    previous = m.custom_seal(root, digest, evidence)
    p = root / "membership.json"
    raw = bytearray(p.read_bytes())
    raw[-2] ^= 1
    p.write_bytes(raw)
    with pytest.raises(ValueError, match="input SHA differs"):
        m.custom_member(previous, "membership.json", evidence)
