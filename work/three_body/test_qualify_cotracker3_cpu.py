"""Pure metadata/type/failure guards only; no Torch or official model imports."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import qualify_cotracker3_cpu as m


class Tensor:
    def __init__(self, device="cpu"):
        self.device = SimpleNamespace(type=device)


TORCH = SimpleNamespace(Tensor=Tensor)


def test_plain_and_wrapped_states_preserve_tensor_identity():
    tensor = Tensor()
    state = {"weight": tensor}
    plain, flag = m.unwrap_checkpoint(state, TORCH)
    assert plain is state and plain["weight"] is tensor and flag is False
    wrapped, flag = m.unwrap_checkpoint({"model": state, "epoch": 100}, TORCH)
    assert wrapped is state and wrapped["weight"] is tensor and flag is True


@pytest.mark.parametrize(
    "state",
    [
        None,
        [],
        {},
        {"model": {}},
        {"model": []},
        {"weight": 1},
        {0: Tensor()},
        {"weight": Tensor("cuda")},
    ],
)
def test_non_state_and_non_cpu_content_rejected(state):
    with pytest.raises(ValueError):
        m.unwrap_checkpoint(state, TORCH)


@pytest.fixture
def closure(tmp_path):
    files = {}
    modules = {"unrelated_package": SimpleNamespace()}
    for name in ("cotracker", "cotracker.predictor"):
        path = tmp_path / (name + ".py")
        path.write_text("# inert fixture\n", encoding="utf-8")
        files[name] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        modules[name] = SimpleNamespace(__file__=str(path))
    return files, modules


def test_actual_loaded_modules_require_exact_names_origins_and_bytes(closure):
    expected, modules = closure
    assert m.validate_loaded_modules(expected, modules) is expected


@pytest.mark.parametrize("drift", ["missing", "extra", "no_file", "wrong_origin", "changed_bytes"])
def test_loaded_code_drift_rejected(closure, drift, tmp_path):
    expected, modules = closure
    if drift == "missing":
        del modules["cotracker.predictor"]
    elif drift == "extra":
        modules["cotracker.unregistered"] = SimpleNamespace(__file__=expected["cotracker"]["path"])
    elif drift == "no_file":
        modules["cotracker.predictor"] = SimpleNamespace()
    elif drift == "wrong_origin":
        other = tmp_path / "other.py"
        other.write_bytes(Path(expected["cotracker.predictor"]["path"]).read_bytes())
        modules["cotracker.predictor"].__file__ = str(other)
    else:
        Path(expected["cotracker.predictor"]["path"]).write_text("# changed\n")
    with pytest.raises(ValueError):
        m.validate_loaded_modules(expected, modules)


def test_existing_output_rejected_before_import_or_mutation(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "WORK", tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "marker.txt"
    marker.write_text("preserve")
    with pytest.raises(ValueError, match="new direct workspace"):
        m.qualify(output)
    assert marker.read_text() == "preserve" and list(output.iterdir()) == [marker]


def test_asset_failure_is_sealed_before_any_model_import(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "WORK", tmp_path)

    def bad_assets(_):
        raise ValueError("synthetic changed asset")

    def forbidden_import(_):
        raise AssertionError("no runtime import permitted after failed asset check")

    monkeypatch.setattr(m, "verify_assets", bad_assets)
    monkeypatch.setattr(m.importlib, "import_module", forbidden_import)
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="synthetic changed asset"):
        m.qualify(output)
    seal = json.loads((output / "qualification.provenance.json").read_text())
    failure = json.loads((output / "failure.json").read_text())
    assert seal["status"] == failure["status"] == "failed"
    assert seal["no_GPU"] is True and seal["no_model_forward"] is None
    assert failure["cuda_initialized"] is None
    assert not (output / "summary.json").exists()
    assert set(seal["outputs"]) == {
        str(Path("code_snapshot") / Path(m.__file__).name),
        "failure.json",
    }
