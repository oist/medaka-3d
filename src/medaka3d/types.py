"""Core data types shared by every pipeline stage.

Deliberately small. These are the only types that cross a stage boundary, so
adding one is a design decision, not a convenience.

The identity vocabulary matters. Three distinct notions were conflated in the
retired repo and that conflation caused real scientific errors:

- **slot** — an arbitrary integer from geometry. Stable only within a run.
  Carries no biological meaning. (The old repo's `fish_id`.)
- **identity** — a biological individual: FEMALE, MALE_1, MALE_2. A claim
  about the world that must be earned from evidence.
- **UNKNOWN** — an explicit refusal to claim. Not a failure state.

Abstention is first class. For a pipeline whose top priority is minimising
identity swaps, a component that says UNKNOWN is strictly better than one that
guesses wrong, and the metrics score it that way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

VIEWS: tuple[str, ...] = ("top_cam", "side1_cam", "side2_cam")
FPS: float = 166.0


class Identity(StrEnum):
    """A biological individual, or an explicit refusal to name one."""

    FEMALE = "F"
    MALE_1 = "M1"
    MALE_2 = "M2"
    UNKNOWN = "?"

    @property
    def is_known(self) -> bool:
        return self is not Identity.UNKNOWN

    @property
    def is_male(self) -> bool:
        return self in (Identity.MALE_1, Identity.MALE_2)


#: The three fish actually present in every experiment.
COHORT: tuple[Identity, ...] = (Identity.FEMALE, Identity.MALE_1, Identity.MALE_2)


@dataclass(frozen=True, slots=True)
class Detection2D:
    """One candidate animal in one camera at one frame.

    ``keypoints`` is (n_nodes, 2) in pixels with NaN for missing nodes; ``mask``
    is an optional boolean array from a segmentation backend. A detector may
    supply either or both — downstream stages must tolerate a missing mask,
    because segmentation is a confidence signal and never a hard veto.
    """

    view: str
    frame: int
    keypoints: np.ndarray | None = None
    mask: np.ndarray | None = None
    bbox: tuple[float, float, float, float] | None = None
    score: float = 1.0

    def __post_init__(self) -> None:
        if self.view not in VIEWS:
            raise ValueError(f"unknown view {self.view!r}; expected one of {VIEWS}")


@dataclass(frozen=True, slots=True)
class Instance3D:
    """One reconstructed animal at one frame, before any identity claim.

    ``slot`` is arbitrary. ``residual`` is in the calibration backend's own
    unit — never compare residuals across backends, and never carry a threshold
    tuned for one backend into another.
    """

    frame: int
    xyz: np.ndarray
    slot: int
    residual: float = float("nan")
    sources: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Encounter:
    """A window in which two animals were close enough for identity to be at risk.

    Bounded on **both** sides by sustained separation. That is not a detail: an
    audit in the retired repo was invalidated because its clips began after the
    animals were already close, so a human could not establish who was who
    before the ambiguity started. An encounter that does not start and end in
    separation cannot be adjudicated, and is therefore not an encounter.
    """

    experiment: str
    start_frame: int
    end_frame: int
    entry_frame: int
    exit_frame: int
    participants: tuple[int, ...]
    min_distance_mm: float

    @property
    def duration_frames(self) -> int:
        return self.exit_frame - self.entry_frame

    @property
    def duration_s(self) -> float:
        return self.duration_frames / FPS

    @property
    def clip_frames(self) -> int:
        """Full adjudicable span, including the separated context on each side."""
        return self.end_frame - self.start_frame


@dataclass(frozen=True, slots=True)
class EncounterTruth:
    """Human ground truth for one encounter — deliberately method-agnostic.

    The human watches a clip that starts and ends in separation, and marks
    where each animal is at entry and again at exit, preserving correspondence.
    That is the irreducible human judgement: *this* fish leaving is *that* fish
    that arrived.

    Crucially this records only what the human saw, never a tracker's opinion.
    Any current or future method is scored against it without re-annotation —
    which is what makes a few hours of labelling a permanent asset rather than
    a per-method cost. The retired repo's audits were method-specific and had to
    be redone from scratch for each candidate; two were run and both were
    thrown away.

    Positions are top-camera pixels, the view the annotator actually works in.
    A 3D method is scored by reprojecting its tracks to the top camera.
    """

    experiment: str
    entry_frame: int
    exit_frame: int
    entry_xy: dict[Identity, tuple[float, float]]
    exit_xy: dict[Identity, tuple[float, float]]
    annotator: str = ""
    confident: bool = True
    note: str = ""

    def __post_init__(self) -> None:
        if self.exit_frame <= self.entry_frame:
            raise ValueError(
                f"exit_frame {self.exit_frame} must follow entry_frame {self.entry_frame}"
            )
        if set(self.entry_xy) != set(self.exit_xy):
            raise ValueError(
                "entry and exit must label the same individuals; got "
                f"{sorted(i.value for i in self.entry_xy)} vs "
                f"{sorted(i.value for i in self.exit_xy)}"
            )
        if any(i is Identity.UNKNOWN for i in self.entry_xy):
            raise ValueError(
                "ground truth cannot contain UNKNOWN. If the annotator could not "
                "follow an animal, mark the encounter confident=False and omit it "
                "from scoring rather than recording a guess."
            )

    @property
    def identities(self) -> tuple[Identity, ...]:
        return tuple(sorted(self.entry_xy, key=lambda i: i.value))
