"""Baseline preservation at the experimental Linker protocol boundary."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from medaka3d.eval.generative import observed_motion_summary
from medaka3d.stages import Linker
from medaka3d.types import Instance3D

LAB = Path(__file__).resolve().parents[1] / "labs" / "20260916_bidirectional_reconnect"


@pytest.fixture(scope="module")
def adapter():
    if not (LAB / "linker.py").exists():
        pytest.skip("bidirectional reconnect experiment has been retired")
    originals = {name: sys.modules.get(name) for name in ("directions", "reconnect")}
    for name in ("directions", "reconnect", "linker"):
        module_name = "_bidirectional_linker" if name == "linker" else name
        spec = importlib.util.spec_from_file_location(module_name, LAB / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    yield module.BaselineDiagnosticLinker
    for name, original in originals.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


def straight(n=180):
    initial = np.array([[0.0, 0, 0], [0, 20.0, 0], [0, 0, 30.0]], dtype=np.float32)
    return initial + np.arange(n, dtype=np.float32)[:, None, None] * np.array(
        [0.1, 0.02, 0], dtype=np.float32
    )


class StubBaseline:
    name = "explicit_test_baseline"

    def __init__(self, tracks):
        self.tracks = tracks
        self.received = None

    def link(self, instances):
        self.received = instances
        return self.tracks


def test_protocol_adapter_records_proposal_without_accepting_it(adapter):
    tracks = straight()
    tracks[80:] = tracks[80:, [1, 0, 2]]
    original = tracks.copy()
    frames = np.arange(1000, 1000 + len(tracks))
    baseline = StubBaseline(tracks)
    linker = adapter(baseline, frames)
    assert isinstance(linker, Linker)
    # These arbitrary input slots must not be copied or interpreted as tracks.
    instances = [Instance3D(1000, np.array([-999, -999, -999]), 41)]
    result = linker.link(instances)
    assert baseline.received is instances
    np.testing.assert_array_equal(result, original)
    np.testing.assert_array_equal(tracks, original)
    assert result.dtype == tracks.dtype
    assert not np.shares_memory(result, tracks)
    report = linker.result
    assert report["accepted_identity_corrections"] == 0
    assert report["baseline_preserved"]
    proposals = [g["persistent_audit"] for g in report["episodes"]]
    assert any(p["suggested_permutation"] == [1, 0, 2] for p in proposals)
    assert all(not p["independent_identity_validated"] for p in proposals)
    assert observed_motion_summary(result, frames) == observed_motion_summary(original, frames)


def test_nan_coverage_source_and_input_instances_are_unchanged(adapter):
    tracks = straight()
    tracks[40:45, 1] = np.nan
    source = {"top_cam": 2, "side1_cam": 0}
    instance = Instance3D(100, np.array([1.0, 2.0, 3.0]), 5, sources=source.copy())
    frames = np.arange(100, 100 + len(tracks))
    linker = adapter(StubBaseline(tracks), frames)
    result = linker.link([instance])
    np.testing.assert_array_equal(np.isnan(result), np.isnan(tracks))
    np.testing.assert_array_equal(result, tracks)
    assert instance.sources == source
    assert instance.slot == 5
    np.testing.assert_array_equal(instance.xyz, [1, 2, 3])


def test_unsupported_exchange_remains_recorded_refusal(adapter):
    tracks = straight()
    tracks[80:] = tracks[80:, [1, 0, 2]]
    support = np.ones((len(tracks), 3), dtype=bool)
    support[80, 0] = False
    linker = adapter(StubBaseline(tracks), np.arange(len(tracks)), support)
    # Caller mutation after binding cannot silently alter evidence.
    support[:] = True
    result = linker.link([])
    audit = linker.result["episodes"][0]["persistent_audit"]
    assert audit["status"] == "rejected_support"
    assert audit["suggested_permutation"] is None
    np.testing.assert_array_equal(result, tracks)


def test_edge_episode_has_explicit_reference_refusal(adapter):
    tracks = straight(12)
    tracks[1, 0, 2] += 20
    linker = adapter(StubBaseline(tracks), np.arange(len(tracks)))
    result = linker.link([])
    audit = linker.result["episodes"][0]["persistent_audit"]
    assert audit["status"] == "insufficient_reference"
    assert audit["reason"] == "padded_episode_outside_bound_frame_schedule"
    np.testing.assert_array_equal(result, tracks)


def test_mismatched_output_fails_loudly_without_stale_diagnostics(adapter):
    baseline = StubBaseline(straight())
    linker = adapter(baseline, np.arange(len(baseline.tracks)))
    linker.link([])
    assert linker.result is not None
    baseline.tracks = baseline.tracks[:-1]
    with pytest.raises(ValueError, match="bound"):
        linker.link([])
    assert linker.result is None


def test_schedule_and_baseline_must_be_explicit(adapter):
    with pytest.raises(TypeError, match="explicit baseline"):
        adapter(object(), np.arange(10))
    with pytest.raises(ValueError, match="contiguous"):
        adapter(StubBaseline(straight(10)), np.arange(10) * 2)
    with pytest.raises(ValueError, match="inside"):
        adapter(StubBaseline(straight(10)), np.arange(10), core_start=5, core_stop=15)
