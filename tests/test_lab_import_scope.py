"""Flat lab imports must be isolated during collection and test execution."""

import importlib
import sys
from types import ModuleType

import pytest

from conftest import LabImportScope


def make_lab(directory, label):
    directory.mkdir()
    (directory / "observations.py").write_text(f"ORIGIN = {label!r}\n", encoding="utf-8")
    (directory / "consumer.py").write_text(
        "def origin():\n    from observations import ORIGIN\n    return ORIGIN\n",
        encoding="utf-8",
    )
    return LabImportScope(directory)


@pytest.mark.parametrize("order", [(0, 1, 0), (1, 0, 1)])
def test_alternating_labs_bind_runtime_imports_and_keep_their_module_identity(tmp_path, order):
    scopes = [make_lab(tmp_path / f"lab_{index}", index) for index in range(2)]
    consumers = {}
    for index in order:
        with scopes[index].activate():
            consumer = importlib.import_module("consumer")
            assert consumer.origin() == index
            if index in consumers:
                assert consumer is consumers[index]
            consumers[index] = consumer


def test_prior_module_and_search_path_are_restored_even_when_import_raises(tmp_path, monkeypatch):
    prior = ModuleType("observations")
    monkeypatch.setitem(sys.modules, "observations", prior)
    scope = make_lab(tmp_path / "lab", "isolated")
    (scope.directory / "broken.py").write_text(
        "import observations\nraise RuntimeError('failed lab import')\n", encoding="utf-8"
    )
    scope = LabImportScope(scope.directory)
    previous_path = sys.path[:]
    with pytest.raises(RuntimeError, match="failed lab import"), scope.activate():
        importlib.import_module("broken")
    assert sys.modules["observations"] is prior
    assert "broken" not in sys.modules
    assert sys.path == previous_path
