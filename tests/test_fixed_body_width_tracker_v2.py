"""CUDA alias normalization without a CUDA query, allocation, renderer or fit."""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

WORK = Path(__file__).resolve().parents[1] / "work/three_body"
sys.path.insert(0, str(WORK))
import fixed_body_width_tracker_v2 as m  # noqa: E402


@pytest.fixture(autouse=True)
def forbid_cuda_and_tensor_work(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("device alias normalization must not query or allocate CUDA")

    for name in ("is_available", "current_device", "device_count", "init", "_lazy_init"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    for name in ("empty", "zeros", "tensor", "as_tensor"):
        monkeypatch.setattr(torch, name, forbidden)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("cuda", "cuda:0"),
        (torch.device("cuda"), "cuda:0"),
        ("cuda:0", "cuda:0"),
        (torch.device("cuda:2"), "cuda:2"),
        ("cpu", "cpu"),
        (torch.device("cpu:0"), "cpu:0"),
    ],
)
def test_only_unspecified_cuda_is_resolved_and_resolution_is_idempotent(value, expected):
    before = str(value)
    result = m.resolved_device(value)
    assert isinstance(result, torch.device)
    assert result == torch.device(expected)
    assert m.resolved_device(result) == result
    assert str(value) == before


@pytest.mark.parametrize("device", ["cuda", "cuda:2", "cpu"])
def test_policy_sees_resolved_device_before_registration_and_preserves_config(monkeypatch, device):
    instance = object.__new__(m.FixedBodyWidthTracker)
    instance.device = torch.device(device)
    instance.config = SimpleNamespace(device=device, width_ratio=0.06)
    instance.scientific_context = {"config": {"device": device}, "input_sha256": "a" * 64}
    seed_token = object()
    instance._fixed_width_reference_pose = seed_token
    before_config = copy.deepcopy(vars(instance.config))
    before_context = copy.deepcopy(instance.scientific_context)
    returned = {"reference_device": "parent result must be returned untouched"}
    calls = []

    def parent_policy(self):
        assert self is instance
        expected = "cuda:0" if device == "cuda" else device
        assert self.device == torch.device(expected)
        assert self._fixed_width_reference_pose is seed_token
        calls.append(self.device)
        return returned

    monkeypatch.setattr(m.PreviousTracker, "_width_policy", parent_policy)
    assert instance._width_policy() is returned
    assert instance._width_policy() is returned
    assert len(calls) == 2 and calls[0] == calls[1]
    assert vars(instance.config) == before_config
    assert instance.scientific_context == before_context


def test_parent_reference_rejection_propagates_without_retry_or_masking(monkeypatch):
    instance = object.__new__(m.FixedBodyWidthTracker)
    instance.device = torch.device("cuda")
    instance.config = SimpleNamespace(device="cuda")
    rejection = ValueError("fixed-width reference policy changed")
    calls = []

    def parent_policy(self):
        calls.append(self.device)
        raise rejection

    monkeypatch.setattr(m.PreviousTracker, "_width_policy", parent_policy)
    with pytest.raises(ValueError) as caught:
        instance._width_policy()
    assert caught.value is rejection
    assert calls == [torch.device("cuda:0")]
    assert instance.config.device == "cuda"
