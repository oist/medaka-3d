"""Tests for the stage protocols and component registry."""

from __future__ import annotations

import numpy as np
import pytest

from medaka3d import stages
from medaka3d.types import Identity


class DummyLinker:
    name = "dummy"

    def link(self, instances):
        return np.zeros((len(instances), 3, 3))


class DummyIdentifier:
    name = "abstainer"

    def identify(self, tracks, frames):
        return {f: dict.fromkeys(range(tracks.shape[1]), Identity.UNKNOWN) for f in frames}


class NotALinker:
    name = "broken"


@pytest.fixture(autouse=True)
def _clean_registry():
    stages.clear()
    yield
    stages.clear()


class TestRegistry:
    def test_register_and_get(self):
        linker = DummyLinker()
        stages.register("linker", "dummy", linker)
        assert stages.get("linker", "dummy") is linker
        assert stages.available("linker") == {"linker": ["dummy"]}

    def test_duplicate_name_is_rejected_not_overwritten(self):
        """Silent shadowing makes results unattributable."""
        stages.register("linker", "dummy", DummyLinker())
        with pytest.raises(stages.RegistryError, match="already registered"):
            stages.register("linker", "dummy", DummyLinker())

    def test_component_must_satisfy_the_protocol(self):
        with pytest.raises(stages.RegistryError, match="does not satisfy"):
            stages.register("linker", "broken", NotALinker())

    def test_unknown_stage_is_rejected(self):
        with pytest.raises(stages.RegistryError, match="unknown stage"):
            stages.register("nonsense", "x", DummyLinker())

    def test_missing_component_lists_alternatives(self):
        stages.register("linker", "dummy", DummyLinker())
        with pytest.raises(stages.RegistryError, match="Registered: dummy"):
            stages.get("linker", "absent")

    def test_helpful_message_when_nothing_registered(self):
        with pytest.raises(stages.RegistryError, match="none registered"):
            stages.get("identifier", "anything")

    def test_all_four_stages_exist(self):
        assert set(stages.PROTOCOLS) == {"detector", "associator", "linker", "identifier"}


class TestProtocolConformance:
    def test_abstaining_identifier_is_valid(self):
        """A component that only ever says UNKNOWN is legitimate, not broken."""
        identifier = DummyIdentifier()
        stages.register("identifier", "abstainer", identifier)
        result = identifier.identify(np.zeros((10, 3, 3)), [0, 1])
        assert result[0][0] is Identity.UNKNOWN
