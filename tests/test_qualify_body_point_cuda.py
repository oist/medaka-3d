"""CPU specification/failure preservation for the future CUDA qualification."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "work/three_body"))
import qualify_body_point_cuda as m


@pytest.fixture(autouse=True)
def forbid_gpu(monkeypatch):
    old = torch.get_num_threads()
    torch.set_num_threads(1)

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU specification must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    yield
    torch.set_num_threads(old)


def test_cpu_specification_exercises_both_terms_with_declared_missing_points():
    report = m.check_fixture_cpu()
    assert report["query_counts"] == [3, 2, 1]
    assert report["query_ray_valid_counts"] == [2, 2, 1]
    assert report["candidate_anchor"] == [2, 0] != report["proposal_parent"]
    assert report["base_pixels_per_view"] == 4 and report["batch"] == 2
    assert report["nonzero_owner_and_gap_gradients"] and report["finite"]
    assert not report["GPU_used"]
    assert report["fit_calls"] == report["graph_calls"] == report["optimizer_steps"] == 0


@pytest.mark.parametrize("fault", [None, "large", "nan", "dtype", "shape"])
def test_comparison_enforces_f64_finiteness_shape_and_fixed_tolerance(fault):
    expected = np.array([0.0, 100.0], np.float64)
    actual = expected + np.array([5e-11, 5e-9])
    if fault == "large":
        actual[0] = 2e-10
    elif fault == "nan":
        actual[0] = np.nan
    elif fault == "dtype":
        actual = actual.astype(np.float32)
    elif fault == "shape":
        actual = actual[None]
    result = m.comparison(expected, actual)
    assert result["passed"] is (fault is None)
    assert result["atol"] == result["rtol"] == 1e-10


def test_support_and_best_iteration_comparisons_require_discrete_exactness():
    for expected, actual in [
        (np.array([True, False]), np.array([True, True])),
        (np.array([0, 14], np.int64), np.array([1, 14], np.int64)),
    ]:
        assert not m.comparison(expected, actual)["passed"]
        assert m.comparison(expected, expected.copy())["passed"]


def test_numerical_mismatch_is_written_before_exception():
    records = []

    class Output:
        def write(self, name, value):
            records.append((name, value))

    expected, actual = {"score": np.array([1.0])}, {"score": np.array([2.0])}
    with pytest.raises(ValueError, match="numerical mismatch"):
        m.save_comparison(Output(), "partial", expected, actual)
    assert [r[0] for r in records] == ["partial.npz", "partial.json"]
    assert records[0][1]["actual_score"].tobytes() == actual["score"].tobytes()
    assert not records[1][1]["passed"]


def test_inspection_restores_all_original_hooks_on_failure_before_fit(monkeypatch):
    fixture = m.synthetic_fixture()
    before = m.fitter.GraphFit, m.fitter.RefractiveFishRenderer, torch.optim.Adam
    error = RuntimeError("synthetic fixture rejected before fit")

    def fail(*args, **kwargs):
        assert m.fitter.GraphFit is not before[0]
        assert m.fitter.RefractiveFishRenderer is not before[1]
        assert torch.optim.Adam is not before[2]
        raise error

    monkeypatch.setattr(m.fitter, "fit_with_body_points", fail)
    with pytest.raises(RuntimeError) as caught:
        m.Inspection(fixture, None).execute()
    assert caught.value is error
    assert (m.fitter.GraphFit, m.fitter.RefractiveFishRenderer, torch.optim.Adam) == before
