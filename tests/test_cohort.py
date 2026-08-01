"""Tests for experiment naming, trial structure and fish descriptors."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest

from medaka3d.cohort import (
    CohortError,
    ExperimentId,
    FishDescriptor,
    group_trials,
    load_groups,
)
from medaka3d.types import Identity

# Mirrors the real sheet, including its mixed ASCII / full-width commas.
GROUP_CSV = """SET DATE,GROUP NAME,FEMALE,MALE1,MALE2,DETAIL
20251222,K,yellow,white,red,
20251222,L,yellow,small,big,
20251223,M,smallest,tail yellow,tail red,
20251216,G,yellow,"white,small","yellow,big",
20251216,B,smallest,"white,small",yellow，big,give up at 20251217
20251299,Z,yellow,white,white,ambiguous males
"""


@pytest.fixture
def groups(tmp_path):
    path = tmp_path / "group_list.csv"
    path.write_text(GROUP_CSV, encoding="utf-8")
    return load_groups(path)


class TestExperimentId:
    def test_parses_the_convention(self):
        e = ExperimentId.parse("20251225_K_1_2_FCMC")
        assert (e.group, e.trial, e.day, e.paradigm) == ("K", 1, 2, "FCMC")
        assert e.set_date == "20251225"

    def test_days_of_one_trial_share_a_trial_key(self):
        """The load-bearing fact: K_1_1..K_1_3 are ONE experimental unit."""
        keys = {ExperimentId.parse(f"20251225_K_1_{d}_FCMC").trial_key for d in (1, 2, 3)}
        assert keys == {("K", 1)}

    def test_different_trials_of_one_group_are_distinct_units(self):
        a = ExperimentId.parse("20251225_K_1_1_FCMC")
        b = ExperimentId.parse("20260109_K_2_1_FCMC")
        assert a.trial_key != b.trial_key

    def test_suffixed_paradigm_is_kept(self):
        assert ExperimentId.parse("20250106_E_2_1_FCMC_P2").paradigm == "FCMC_P2"

    def test_rejects_non_conforming_names(self):
        for bad in ("G_test", "cali_251217_way1", "I_TEST"):
            with pytest.raises(CohortError, match="cannot parse"):
                ExperimentId.parse(bad)


class TestFishDescriptor:
    @pytest.mark.parametrize(
        ("raw", "colours", "sizes"),
        [
            ("white", {"white"}, set()),
            ("red", {"red"}, set()),
            ("white,small", {"white"}, {"small"}),
            ("yellow，big", {"yellow"}, {"big"}),  # full-width comma, as in the sheet
            ("yellow, small", {"yellow"}, {"small"}),
            ("tail yellow", {"yellow"}, set()),
            ("biggest", set(), {"biggest"}),
            ("smallest", set(), {"smallest"}),
            ("", set(), set()),
        ],
    )
    def test_parses_experimenter_vocabulary(self, raw, colours, sizes):
        d = FishDescriptor.parse(Identity.MALE_1, raw)
        assert set(d.colours) == colours
        assert set(d.sizes) == sizes

    def test_biggest_does_not_also_match_big(self):
        d = FishDescriptor.parse(Identity.MALE_2, "biggest")
        assert set(d.sizes) == {"biggest"}

    def test_empty_descriptor_is_flagged(self):
        assert FishDescriptor.parse(Identity.MALE_1, "").is_empty


class TestGroups:
    def test_loads_all_groups(self, groups):
        assert set(groups) == {"K", "L", "M", "G", "B", "Z"}

    def test_k_males_are_white_and_red(self, groups):
        """The most-studied group is colour-separated -- a good first target."""
        k = groups["K"]
        assert set(k.male_1.colours) == {"white"}
        assert set(k.male_2.colours) == {"red"}
        assert k.descriptor(Identity.FEMALE).colours == frozenset({"yellow"})

    @pytest.mark.parametrize(
        ("group", "cue"),
        [("K", "colour"), ("M", "colour"), ("L", "size"), ("G", "colour+size"), ("Z", "none")],
    )
    def test_male_cue_classification(self, groups, group, cue):
        assert groups[group].male_cue == cue

    def test_size_only_groups_are_identifiable_as_weak(self, groups):
        """Size varies with pose and depth, and fish grow between days."""
        weak = [g.name for g in groups.values() if g.male_cue in ("size", "none")]
        assert set(weak) == {"L", "Z"}

    def test_unknown_identity_has_no_descriptor(self, groups):
        with pytest.raises(CohortError, match="no descriptor"):
            groups["K"].descriptor(Identity.UNKNOWN)

    def test_duplicate_group_is_rejected(self, tmp_path):
        path = tmp_path / "dup.csv"
        path.write_text(GROUP_CSV + "20251222,K,yellow,white,red,\n", encoding="utf-8")
        with pytest.raises(CohortError, match="appears twice"):
            load_groups(path)


class TestTrials:
    FOLDERS: ClassVar[list[str]] = [
        "20251225_K_1_1_FCMC",
        "20251225_K_1_2_FCMC",
        "20251225_K_1_3_FCMC",
        "20260109_K_2_1_FCMC",
        "20251220_E_1_1_FCMC",
        "G_test",  # not a recording
    ]

    def test_groups_days_into_trials(self):
        trials = group_trials(self.FOLDERS)
        assert set(trials) == {("K", 1), ("K", 2), ("E", 1)}
        assert trials[("K", 1)].n_days == 3

    def test_days_are_ordered(self):
        trial = group_trials(list(reversed(self.FOLDERS)))[("K", 1)]
        assert [e.day for e in trial.days] == [1, 2, 3]

    def test_non_conforming_folders_are_skipped_by_default(self):
        assert "G_test" not in {
            e.folder for t in group_trials(self.FOLDERS).values() for e in t.days
        }

    def test_strict_mode_raises(self):
        with pytest.raises(CohortError):
            group_trials(self.FOLDERS, strict=True)

    def test_holdout_split_is_by_day(self):
        """Cross-day evaluation: same animals, different day, no leakage."""
        trial = group_trials(self.FOLDERS)[("K", 1)]
        train, test = trial.holdout_split(test_day=3)
        assert [e.day for e in train] == [1, 2]
        assert test.day == 3
        assert test.folder not in {e.folder for e in train}

    def test_holdout_rejects_a_missing_day(self):
        with pytest.raises(CohortError, match="no day 9"):
            group_trials(self.FOLDERS)[("K", 1)].holdout_split(test_day=9)


def test_real_group_list_is_parseable():
    """Guard against the shipped sheet drifting out of the expected shape."""

    real = Path("Z:/StephensU/medaka/assets/catalog/experiment_list/group_list.csv")
    if not real.is_file():
        pytest.skip("asset root not mounted")
    groups = load_groups(real)
    assert len(groups) >= 14
    assert groups["K"].male_cue == "colour"
    # every group must give the two males *some* distinguishing descriptor
    unusable = [g.name for g in groups.values() if g.male_cue == "none"]
    assert not unusable, f"groups with indistinguishable males: {unusable}"
