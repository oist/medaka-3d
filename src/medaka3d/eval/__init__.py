"""Benchmark harness: encounter segmentation and identity-swap scoring.

This package is shared by production and by every lab. No experiment writes its
own evaluation code — that is what made the retired repo's method comparisons
incomparable and left them to be settled by argument rather than measurement.
"""

from medaka3d.eval.encounters import (
    EncounterConfig,
    find_encounters,
    find_pair_encounters,
    summarise,
)
from medaka3d.eval.metrics import EncounterOutcome, Scorecard, score_encounter
from medaka3d.eval.tracking import (
    ViewTracks,
    fragmentation,
    identity_preservation,
    modal_agreement,
    switch_candidates,
    track_jumps,
)

__all__ = [
    "EncounterConfig",
    "EncounterOutcome",
    "Scorecard",
    "ViewTracks",
    "find_encounters",
    "find_pair_encounters",
    "fragmentation",
    "identity_preservation",
    "modal_agreement",
    "score_encounter",
    "summarise",
    "switch_candidates",
    "track_jumps",
]
