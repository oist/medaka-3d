"""Pipeline stage contracts and the component registry.

The pipeline is four swappable stages:

    Detector    video          -> per-view 2D candidates
    Associator  2D candidates  -> 3D instances at one frame (no identity)
    Linker      3D instances   -> temporally continuous tracks (still no identity)
    Identifier  tracks         -> biological identity, or UNKNOWN

Two things follow from this decomposition, and both are the point of it.

**Experiments plug in at exactly one stage.** Testing segmentation-instance
tracking replaces the Linker; testing a new appearance classifier replaces the
Identifier; a scene model that suppresses reflections replaces the Detector.
None of them requires touching the other three, so a result is attributable to
the thing that changed.

**Every implementation is scored by the same harness.** In the retired repo each
experiment brought its own evaluation code, so numbers were not comparable and
the comparison was settled by argument. Here a component that satisfies a
Protocol is runnable by the shared benchmark, unmodified.

Separating Linker from Identifier is deliberate. Temporal continuity ("this is
the same animal as last frame") and biological identity ("this animal is the
female") are different claims with different evidence and different failure
modes. The retired repo fused them, so a geometry failure and a naming failure
were indistinguishable in the output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    import numpy as np

    from medaka3d.types import Detection2D, Identity, Instance3D


@runtime_checkable
class Detector(Protocol):
    """Finds candidate animals in one camera view."""

    name: str

    def detect(self, video: Path, frames: Sequence[int]) -> Iterable[Detection2D]:
        """Yield candidates for the requested frames.

        May return more candidates than there are animals. Over-detection is
        expected and useful: the Associator resolves it with multi-view
        geometry, and suppressing candidates here loses information that cannot
        be recovered later.
        """
        ...


@runtime_checkable
class Associator(Protocol):
    """Combines per-view candidates into 3D instances at a single frame."""

    name: str

    def associate(
        self, detections: dict[str, Sequence[Detection2D]], frame: int
    ) -> list[Instance3D]:
        """Return 3D instances for one frame, with arbitrary slot indices.

        Must not use temporal information — that is the Linker's job. Keeping
        this stage memoryless is what allows a linking failure to be diagnosed
        separately from an association failure.
        """
        ...


@runtime_checkable
class Linker(Protocol):
    """Joins per-frame 3D instances into temporally continuous tracks."""

    name: str

    def link(self, instances: Sequence[Instance3D]) -> np.ndarray:
        """Return an (n_frames, n_animals, 3) array of linked trajectories.

        Slots are arbitrary but must be *consistent over time* — that is the
        entire content of this stage. NaN where an animal is unobserved. A
        linker should prefer breaking a track to bridging a gap it cannot
        justify: a NaN is recoverable downstream, a wrong join is not.
        """
        ...


@runtime_checkable
class Identifier(Protocol):
    """Assigns biological identity to linked tracks."""

    name: str

    def identify(self, tracks: np.ndarray, frames: Sequence[int]) -> dict[int, dict[int, Identity]]:
        """Map ``{frame: {slot: Identity}}``.

        Returning ``Identity.UNKNOWN`` is a legitimate, and often preferable,
        answer. The benchmark scores abstention separately from error precisely
        so that this stage is never pushed toward confident guessing.
        """
        ...


#: Stage name -> Protocol. Used by the registry and the labs hygiene test.
PROTOCOLS: dict[str, type] = {
    "detector": Detector,
    "associator": Associator,
    "linker": Linker,
    "identifier": Identifier,
}

_REGISTRY: dict[str, dict[str, object]] = {stage: {} for stage in PROTOCOLS}


class RegistryError(RuntimeError):
    """Raised on an unknown stage, a duplicate name, or a missing component."""


def register(stage: str, name: str, component: object) -> None:
    """Register a component under a stage.

    Duplicate names are rejected rather than overwritten. Silent shadowing is
    how two different things end up with one name and results become
    unattributable.
    """
    if stage not in _REGISTRY:
        raise RegistryError(f"unknown stage {stage!r}; expected one of {sorted(PROTOCOLS)}")
    if name in _REGISTRY[stage]:
        raise RegistryError(
            f"{stage} {name!r} is already registered. Pick a distinct name — "
            f"overwriting would make results unattributable."
        )
    protocol = PROTOCOLS[stage]
    if not isinstance(component, protocol):
        raise RegistryError(
            f"{name!r} does not satisfy the {protocol.__name__} protocol; "
            f"it must expose a 'name' attribute and the stage method."
        )
    _REGISTRY[stage][name] = component


def get(stage: str, name: str) -> object:
    """Look up a registered component."""
    if stage not in _REGISTRY:
        raise RegistryError(f"unknown stage {stage!r}; expected one of {sorted(PROTOCOLS)}")
    try:
        return _REGISTRY[stage][name]
    except KeyError:
        known = sorted(_REGISTRY[stage]) or ["<none registered>"]
        raise RegistryError(f"no {stage} named {name!r}. Registered: {', '.join(known)}") from None


def available(stage: str | None = None) -> dict[str, list[str]]:
    """List registered component names, by stage."""
    stages = [stage] if stage else list(_REGISTRY)
    return {s: sorted(_REGISTRY[s]) for s in stages}


def clear() -> None:
    """Empty the registry. Tests only."""
    for stage in _REGISTRY:
        _REGISTRY[stage].clear()
