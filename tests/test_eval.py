"""Tests for encounter segmentation and swap scoring."""

from __future__ import annotations

import numpy as np
import pytest

from medaka3d.eval.encounters import (
    EncounterConfig,
    find_encounters,
    find_pair_encounters,
    summarise,
)
from medaka3d.eval.metrics import Scorecard, score_encounter
from medaka3d.types import EncounterTruth, Identity

CFG = EncounterConfig(close_mm=20.0, separation_mm=30.0, min_separation_frames=10, context_frames=5)


def trace(*segments: tuple[float, int]) -> np.ndarray:
    """Build a distance trace from (value, n_frames) segments."""
    return np.concatenate([np.full(n, v, dtype=float) for v, n in segments])


class TestEncounterSegmentation:
    def test_finds_a_bounded_approach(self):
        d = trace((50, 20), (5, 8), (50, 20))
        (enc,) = find_pair_encounters(d, config=CFG)
        assert (enc.entry_frame, enc.exit_frame) == (20, 28)
        assert enc.min_distance_mm == 5.0
        assert enc.start_frame == 15 and enc.end_frame == 33  # context on both sides

    def test_unbounded_ambiguity_is_not_an_encounter(self):
        """The defect that invalidated the retired repo's audit.

        A clip that starts already-close cannot be adjudicated: the annotator
        never saw who arrived. Such a window is deliberately not emitted.
        """
        assert find_pair_encounters(trace((5, 30), (50, 20)), config=CFG) == []
        assert find_pair_encounters(trace((50, 20), (5, 30)), config=CFG) == []

    def test_brief_separation_does_not_split_one_encounter(self):
        """Under-long separation must not inflate the encounter count.

        The retired repo's audit counted 7 "crossings" that a proper sustained
        separation rule collapsed to 2 encounters -- a 3.5x overstatement of
        sample size.
        """
        d = trace((50, 20), (5, 8), (50, 4), (5, 8), (50, 20))  # 4-frame gap < 10
        encounters = find_pair_encounters(d, config=CFG)
        assert len(encounters) == 1
        assert (encounters[0].entry_frame, encounters[0].exit_frame) == (20, 40)

    def test_sufficient_separation_does_split(self):
        d = trace((50, 20), (5, 8), (50, 15), (5, 8), (50, 20))  # 15-frame gap >= 10
        assert len(find_pair_encounters(d, config=CFG)) == 2

    def test_passing_nearby_without_contact_is_ignored(self):
        assert find_pair_encounters(trace((50, 20), (25, 8), (50, 20)), config=CFG) == []

    def test_nan_is_not_evidence_of_separation(self):
        """A dropped detection cannot prove the animals were apart."""
        d = trace((50, 20), (np.nan, 30), (50, 20))
        assert find_pair_encounters(d, config=CFG) == []

    def test_hysteresis_is_required(self):
        with pytest.raises(ValueError, match="must exceed close_mm"):
            EncounterConfig(close_mm=20.0, separation_mm=20.0)

    def test_frame_offset_is_applied(self):
        (enc,) = find_pair_encounters(
            trace((50, 20), (5, 8), (50, 20)), config=CFG, frame_offset=1000
        )
        assert (enc.entry_frame, enc.exit_frame) == (1020, 1028)

    def test_multi_animal_reports_the_pair(self):
        n = 60
        tracks = np.zeros((n, 3, 3))
        tracks[:, 1, 0] = 100.0  # animal 1 far away on x
        tracks[:, 2, 0] = 100.0  # animal 2 likewise
        tracks[25:33, 2, 0] = 5.0  # animal 2 approaches animal 0
        encounters = find_encounters(tracks, config=CFG, experiment="EXP")
        assert [e.participants for e in encounters] == [(0, 2)]
        assert encounters[0].experiment == "EXP"

    def test_summarise_estimates_annotation_cost(self):
        d = trace((50, 20), (5, 8), (50, 20))
        s = summarise(find_pair_encounters(d, config=CFG))
        assert s["count"] == 1
        assert s["annotation_minutes_estimate"] == 1.5
        assert summarise([])["count"] == 0


def truth(entry, exit_, **kw) -> EncounterTruth:
    return EncounterTruth(
        experiment="EXP",
        entry_frame=100,
        exit_frame=200,
        entry_xy=entry,
        exit_xy=exit_,
        **kw,
    )


F, M1, M2 = Identity.FEMALE, Identity.MALE_1, Identity.MALE_2
SEPARATED = {F: (0.0, 0.0), M1: (500.0, 0.0), M2: (0.0, 500.0)}


class TestEncounterTruth:
    def test_rejects_unknown_in_ground_truth(self):
        with pytest.raises(ValueError, match="cannot contain UNKNOWN"):
            truth({Identity.UNKNOWN: (0.0, 0.0)}, {Identity.UNKNOWN: (0.0, 0.0)})

    def test_rejects_mismatched_individuals(self):
        with pytest.raises(ValueError, match="same individuals"):
            truth({F: (0.0, 0.0)}, {M1: (0.0, 0.0)})

    def test_rejects_reversed_frames(self):
        with pytest.raises(ValueError, match="must follow"):
            EncounterTruth(
                experiment="E",
                entry_frame=200,
                exit_frame=100,
                entry_xy={F: (0.0, 0.0)},
                exit_xy={F: (0.0, 0.0)},
            )


class TestScoring:
    def test_clean_crossing_survives(self):
        out = score_encounter(truth(SEPARATED, SEPARATED), SEPARATED, SEPARATED)
        assert out.survived and not out.abstained and not out.is_error

    def test_male_male_swap_is_caught(self):
        """The failure mode the whole benchmark exists to detect."""
        swapped = {F: (0.0, 0.0), M1: (0.0, 500.0), M2: (500.0, 0.0)}
        out = score_encounter(truth(SEPARATED, SEPARATED), SEPARATED, swapped)
        assert not out.survived and out.is_error
        # one exchange, counted once -- not once per animal involved
        assert out.swapped_pairs == ((M1, M2),)

    def test_consistent_relabelling_survives_but_is_misnamed(self):
        """Continuity and naming are separate failures.

        M1/M2 are anchored to the experimenter's colour descriptors (group K:
        white vs red), so a consistent global relabelling is a real error --
        but it is a *naming* error, not a tracking one. The method never lost
        an animal, so it survived.
        """
        relabelled = {F: (0.0, 0.0), M2: (500.0, 0.0), M1: (0.0, 500.0)}
        out = score_encounter(truth(SEPARATED, SEPARATED), relabelled, relabelled)
        assert out.survived and not out.is_error  # continuity intact
        assert set(out.misnamed) == {M1, M2}  # naming wrong
        assert not out.correctly_named

    def test_correct_naming_is_recorded(self):
        out = score_encounter(truth(SEPARATED, SEPARATED), SEPARATED, SEPARATED)
        assert out.correctly_named and out.misnamed == ()
        assert out.n_named == 3

    def test_unknown_is_abstention_not_error(self):
        partial = {F: (0.0, 0.0), Identity.UNKNOWN: (500.0, 0.0)}
        out = score_encounter(truth(SEPARATED, SEPARATED), partial, partial)
        assert out.abstained and not out.is_error and not out.survived

    def test_missing_prediction_is_abstention(self):
        out = score_encounter(truth(SEPARATED, SEPARATED), {F: (0.0, 0.0)}, {F: (0.0, 0.0)})
        assert out.abstained and set(out.unmatched) == {M1, M2}

    def test_far_prediction_does_not_match(self):
        far = {F: (9000.0, 9000.0), M1: (500.0, 0.0), M2: (0.0, 500.0)}
        out = score_encounter(truth(SEPARATED, SEPARATED), far, far)
        assert out.abstained and out.unmatched == (F,)


class TestScorecard:
    def build(self) -> Scorecard:
        card = Scorecard(method="demo")
        swapped = {F: (0.0, 0.0), M1: (0.0, 500.0), M2: (500.0, 0.0)}
        partial = {F: (0.0, 0.0), Identity.UNKNOWN: (500.0, 0.0)}
        card.outcomes = [
            score_encounter(truth(SEPARATED, SEPARATED), SEPARATED, SEPARATED),
            score_encounter(truth(SEPARATED, SEPARATED), SEPARATED, swapped),
            score_encounter(truth(SEPARATED, SEPARATED), partial, partial),
        ]
        return card

    def test_rates(self):
        card = self.build()
        assert (card.n_survived, card.n_swapped, card.n_abstained) == (1, 1, 1)
        assert card.survival_rate == pytest.approx(1 / 3)
        # abstention excluded from the denominator here
        assert card.precision_when_committed == pytest.approx(1 / 2)
        assert card.abstention_rate == pytest.approx(1 / 3)

    def test_swaps_reported_per_pair_never_pooled(self):
        assert self.build().swaps_by_pair() == {"M1-M2": 1}

    def test_naming_accuracy_is_independent_of_survival(self):
        """A perfect tracker that names every animal wrong: survival 1, naming 0."""
        relabelled = {F: (500.0, 0.0), M1: (0.0, 500.0), M2: (0.0, 0.0)}
        card = Scorecard(method="misnamer")
        card.outcomes = [score_encounter(truth(SEPARATED, SEPARATED), relabelled, relabelled)]
        assert card.survival_rate == 1.0
        assert card.naming_accuracy == 0.0

    def test_naming_accuracy_ignores_abstentions(self):
        card = self.build()  # 1 correct, 1 swapped, 1 abstained
        # 2 committed encounters x 3 animals; the swap misnames 2 of them
        assert card.naming_accuracy == pytest.approx(4 / 6)

    def test_summary_is_json_ready(self):
        import json

        json.dumps(self.build().summary())

    def test_empty_scorecard_does_not_divide_by_zero(self):
        card = Scorecard(method="none")
        assert np.isnan(card.survival_rate)
        assert card.summary()["survival_rate"] is None
