"""Scoring a method against human encounter ground truth.

The headline number is **encounter survival**: the fraction of human-adjudicated
encounters a method crosses without swapping any identity.

Three design choices, each answering a specific failure of the retired repo.

**1. Encounters, not frames.** At 166 fps a frame-wise identity accuracy is
dominated by the long easy stretches where the animals are far apart, so a
method can swap at every single crossing and still report >99%. §2.2 of the
findings dossier records exactly this: track-quality QC metrics were
structurally blind to swaps. Encounter survival has no such blind spot — it
measures only the moments where identity is actually at risk.

**2. Abstention is not an error.** A method that returns UNKNOWN through a
crossing has not swapped anyone; it has declined to guess. For a pipeline whose
stated top priority is minimising swaps, that is a *good* outcome and is scored
separately from a wrong answer. Reporting a single accuracy number would push
development toward confident guessing, which is the opposite of what is wanted.

**3. Swaps are counted per identity pair.** Female-male and male-male swaps are
different problems with different difficulty and different scientific cost, and
are never pooled into one figure.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from medaka3d.types import Identity

#: A predicted position further than this from the human mark is no match.
DEFAULT_MATCH_RADIUS_PX = 60.0


@dataclass(frozen=True, slots=True)
class EncounterOutcome:
    """How one method fared on one encounter."""

    experiment: str
    entry_frame: int
    survived: bool
    abstained: bool
    swapped_pairs: tuple[tuple[Identity, Identity], ...] = ()
    unmatched: tuple[Identity, ...] = ()

    @property
    def is_error(self) -> bool:
        """A wrong claim. Abstention and unmatched detections are not errors."""
        return bool(self.swapped_pairs)


@dataclass(slots=True)
class Scorecard:
    """Aggregate result for one method over one benchmark."""

    method: str = ""
    outcomes: list[EncounterOutcome] = field(default_factory=list)

    @property
    def n_encounters(self) -> int:
        return len(self.outcomes)

    @property
    def n_adjudicated(self) -> int:
        """Encounters where the method committed to an answer."""
        return sum(1 for o in self.outcomes if not o.abstained)

    @property
    def n_survived(self) -> int:
        return sum(1 for o in self.outcomes if o.survived)

    @property
    def n_swapped(self) -> int:
        return sum(1 for o in self.outcomes if o.is_error)

    @property
    def n_abstained(self) -> int:
        return sum(1 for o in self.outcomes if o.abstained)

    @property
    def survival_rate(self) -> float:
        """Headline metric: share of all encounters crossed without a swap.

        Abstentions count against it — declining to answer is safe, not correct.
        """
        return self.n_survived / self.n_encounters if self.n_encounters else float("nan")

    @property
    def precision_when_committed(self) -> float:
        """Share of *committed* answers that were right.

        Read alongside ``survival_rate``: a method with high precision and heavy
        abstention is a usable partial solution; high abstention alone is not.
        """
        return self.n_survived / self.n_adjudicated if self.n_adjudicated else float("nan")

    @property
    def abstention_rate(self) -> float:
        return self.n_abstained / self.n_encounters if self.n_encounters else float("nan")

    def swaps_by_pair(self) -> dict[str, int]:
        """Swap counts keyed by the identity pair involved, never pooled."""
        counter: Counter[str] = Counter()
        for outcome in self.outcomes:
            for a, b in outcome.swapped_pairs:
                counter["-".join(sorted((a.value, b.value)))] += 1
        return dict(counter)

    def summary(self) -> dict[str, object]:
        return {
            "method": self.method,
            "encounters": self.n_encounters,
            "survived": self.n_survived,
            "swapped": self.n_swapped,
            "abstained": self.n_abstained,
            "survival_rate": round(self.survival_rate, 4) if self.n_encounters else None,
            "precision_when_committed": (
                round(self.precision_when_committed, 4) if self.n_adjudicated else None
            ),
            "abstention_rate": round(self.abstention_rate, 4) if self.n_encounters else None,
            "swaps_by_pair": self.swaps_by_pair(),
        }


def _assign(
    predicted: dict[Identity, tuple[float, float]],
    truth: dict[Identity, tuple[float, float]],
    radius: float,
) -> dict[Identity, Identity | None]:
    """Map each true identity to the predicted label nearest its human mark.

    Greedy nearest-first, which is exact enough here: three animals, and a
    correct method has them far apart at entry and exit by construction.
    """
    pairs = [
        (float(np.hypot(p[0] - t[0], p[1] - t[1])), true_id, pred_id)
        for true_id, t in truth.items()
        for pred_id, p in predicted.items()
    ]
    pairs.sort(key=lambda x: x[0])

    out: dict[Identity, Identity | None] = dict.fromkeys(truth)
    used: set[Identity] = set()
    for distance, true_id, pred_id in pairs:
        if distance > radius or out[true_id] is not None or pred_id in used:
            continue
        out[true_id] = pred_id
        used.add(pred_id)
    return out


def score_encounter(
    truth,
    entry_pred: dict[Identity, tuple[float, float]],
    exit_pred: dict[Identity, tuple[float, float]],
    *,
    match_radius_px: float = DEFAULT_MATCH_RADIUS_PX,
) -> EncounterOutcome:
    """Score one method's identity assignment across one encounter.

    Args:
        truth: An ``EncounterTruth``.
        entry_pred: Method's labelled positions at ``truth.entry_frame``.
        exit_pred: Method's labelled positions at ``truth.exit_frame``.

    The method survives if every animal the human could follow carries the same
    predicted label at exit as at entry. If the method labelled an animal
    UNKNOWN at either end, or produced nothing within ``match_radius_px``, the
    encounter is an abstention rather than a failure.

    Note what this deliberately does **not** penalise: a method that calls the
    true M1 "M2" at both entry and exit scores as survived. That is correct.
    The M1/M2 letters are arbitrary — the findings dossier records the
    assignment as explicitly provisional — so a consistent global relabelling is
    a naming difference, not a tracking failure. Only a *change* in assignment
    across the encounter is a swap.

    The same leniency applies to the female, which means sexing errors are
    invisible here. That is a separate question with a separate metric; do not
    read a high survival rate as evidence that the female was correctly
    identified.
    """
    at_entry = _assign(entry_pred, truth.entry_xy, match_radius_px)
    at_exit = _assign(exit_pred, truth.exit_xy, match_radius_px)

    unmatched = tuple(
        i
        for i in truth.identities
        if at_entry[i] is None
        or at_exit[i] is None
        or at_entry[i] is Identity.UNKNOWN
        or at_exit[i] is Identity.UNKNOWN
    )
    if unmatched:
        return EncounterOutcome(
            experiment=truth.experiment,
            entry_frame=truth.entry_frame,
            survived=False,
            abstained=True,
            unmatched=unmatched,
        )

    # One exchange involves two animals and must be counted once, not twice:
    # a single M1<->M2 swap is one failure, not two.
    swapped = tuple(
        sorted(
            {
                tuple(sorted((i, at_exit[i]), key=lambda x: x.value))  # type: ignore[arg-type]
                for i in truth.identities
                if at_entry[i] != at_exit[i]
            }
        )
    )
    return EncounterOutcome(
        experiment=truth.experiment,
        entry_frame=truth.entry_frame,
        survived=not swapped,
        abstained=False,
        swapped_pairs=swapped,  # type: ignore[arg-type]
    )
