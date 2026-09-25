"""Synthetic camera-serialization/retry delta only; no real numeric payloads."""

from __future__ import annotations

import ast
import dataclasses
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import diagnose_temporal_refractive_geometry as old
import diagnose_temporal_refractive_geometry_v2 as new
import numpy as np
import pytest


@dataclasses.dataclass
class Camera:
    center: np.ndarray
    rotation: np.ndarray
    distortion: tuple


def fixture():
    camera = Camera(np.array([1.0, -0.0, 3.0]), np.eye(3), (0.0, 0.1))
    rig = SimpleNamespace(cameras=[camera], views=tuple(new.VIEWS), world_units="mm")
    cameras = json.loads(new.coh._canonical([dataclasses.asdict(camera)]))
    return rig, dict(context=dict(cameras=cameras, views=list(new.VIEWS), world_units="mm"))


def test_actual_serializer_shape_passes_but_old_tolist_fails():
    rig, metadata = fixture()
    before = deepcopy(metadata)
    assert set(metadata["context"]["cameras"][0]["center"]) == {"shape", "dtype", "sha256"}
    with pytest.raises(ValueError, match="context"):
        old.context_geometry(metadata, rig)
    new.context_geometry(metadata, rig)
    assert metadata == before


@pytest.mark.parametrize("fault", ["value", "dtype", "signed_zero", "views", "units"])
def test_serializer_still_rejects_real_binding_drift(fault):
    rig, metadata = fixture()
    if fault == "value":
        rig.cameras[0].center[0] += 1
    elif fault == "dtype":
        rig.cameras[0].center = rig.cameras[0].center.astype(np.float32)
    elif fault == "signed_zero":
        rig.cameras[0].center[1] = 0.0
    elif fault == "views":
        metadata["context"]["views"] = list(reversed(new.VIEWS))
    else:
        metadata["context"]["world_units"] = "cm"
    with pytest.raises(ValueError, match="context"):
        new.context_geometry(metadata, rig)


def test_all_scientific_functions_ast_unchanged():
    def definitions(module):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        return {
            n.name: ast.dump(n, include_attributes=False)
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        }

    a, b = definitions(old), definitions(new)
    assert set(b) - set(a) == {"retry_binding"}
    assert {k for k in a if a[k] != b[k]} == {"context_geometry", "run"}
    assert new.STAGE == "temporal_refractive_geometry_diagnosis_v2"
    assert new.PLAN_SHA == old.PLAN_SHA
    assert new.REFERENCES == old.REFERENCES


def failure():
    source = new.WORK / "diagnose_temporal_refractive_geometry.py"
    return dict(
        status="failed",
        stage=old.STAGE,
        error="registered refractive context differs",
        inputs={str(source): new.PINS[source]},
    )


def test_retry_is_planned_and_does_not_adopt_partial(monkeypatch):
    monkeypatch.setattr(new.base, "read_json", lambda path: failure())
    added = []
    result = new.retry_binding(SimpleNamespace(add=lambda *args: added.append(args)))
    assert len(added) == 1
    assert result["full_two_window_reexecution_planned"] is True
    assert result["prior_partial_is_success"] is False
    assert result["prior_numerical_outputs_reused"] is False


@pytest.mark.parametrize("fault", ["status", "stage", "error", "source", "candidate"])
def test_retry_rejects_other_failure_origin(monkeypatch, fault):
    value = failure()
    if fault in ("status", "stage", "error"):
        value[fault] = "different"
    elif fault == "source":
        value["inputs"][str(new.WORK / "diagnose_temporal_refractive_geometry.py")] = "0" * 64
    else:
        value["inputs"][str(new.WORK / "candidate.npz")] = "0" * 64
    monkeypatch.setattr(new.base, "read_json", lambda path: value)
    with pytest.raises(ValueError):
        new.retry_binding(SimpleNamespace(add=lambda *args: None))
