"""Experiment naming, trial structure, and the physical identity of each fish.

Two things live here that the retired repo left implicit, at real scientific
cost.

**The group is the individual.** A folder name encodes
``<set_date>_<GROUP>_<trial>_<day>_<paradigm>``. A group name denotes one trio of
fish for its entire lifetime, so *every* experiment sharing a group name — across
days and across trials — is a repeated measure of the same three animals. The
white male of ``K_1`` is the same fish as the white male of ``K_2``, 15 days
later.

Consequences: ``K_1_1``, ``K_1_2``, ``K_1_3`` are one trio observed three times,
so the retired repo's headline "female mate choice in all 3 trials
(0.61 / 0.71 / 0.80)" is n=1, not n=3 — its own dossier recorded the question as
INCONCLUSIVE while the answer sat in the experiment list. And the independent
unit is the **group**, not the trial: ``Trio`` exists so that this is a type
rather than a convention someone has to remember.

The same fact is an asset. Six recording days spanning weeks, of known-identical
individuals, is a re-identification benchmark that needs no extra annotation:
train on one trial, test on another.

**M1 and M2 denote specific animals.** The experimenter recorded a physical
descriptor for each — colour and relative size — in ``group_list.csv`` (group K:
males *white* and *red*), assigned by expert morphological examination at setup,
never from behaviour.

Note the limit, which matters for ground truth: those descriptors were assigned
under close examination, and the experimenter reports being **unable to apply
them reliably to a single video frame**. Continuity through a crossing can be
adjudicated by scrubbing; absolute naming of a male from one frame generally
cannot. See ``docs/DEVELOPMENT_STRATEGY.md`` for how ground truth is built
around that constraint.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from medaka3d.types import Identity

FOLDER = re.compile(
    r"^(?P<set_date>\d{8})_(?P<group>[A-Z]+)_(?P<trial>\d+)_(?P<day>\d+)_(?P<paradigm>.+)$"
)

#: Descriptor vocabulary used by the experimenter, in priority order.
COLOURS: tuple[str, ...] = ("white", "yellow", "red")
SIZES: tuple[str, ...] = ("biggest", "smallest", "big", "small")

# The sheet mixes ASCII and full-width commas (the latter typed on a Japanese
# IME). Both are real separators in the source data; do not "fix" this.
_SEPARATORS = re.compile(r"[,，]")  # noqa: RUF001


class CohortError(ValueError):
    """Raised on an unparseable folder name or group record."""


@dataclass(frozen=True, slots=True)
class ExperimentId:
    """A parsed recording folder name."""

    folder: str
    set_date: str
    group: str
    trial: int
    day: int
    paradigm: str

    @classmethod
    def parse(cls, folder: str) -> ExperimentId:
        match = FOLDER.match(folder)
        if not match:
            raise CohortError(
                f"cannot parse {folder!r}; expected "
                f"<set_date>_<GROUP>_<trial>_<day>_<paradigm>, e.g. 20251225_K_1_2_FCMC"
            )
        g = match.groupdict()
        return cls(
            folder=folder,
            set_date=g["set_date"],
            group=g["group"],
            trial=int(g["trial"]),
            day=int(g["day"]),
            paradigm=g["paradigm"],
        )

    @property
    def trio_key(self) -> str:
        """**The independent experimental unit.**

        A group name denotes one trio of individuals for its whole lifetime. The
        white male of ``K_1`` is the *same fish* as the white male of ``K_2``,
        recorded 15 days later. So every experiment sharing a group name is a
        repeated measure of the same three animals, and statistics must be
        grouped by this, never by folder and never by trial.
        """
        return self.group

    @property
    def trial_key(self) -> tuple[str, int]:
        """A recording block: consecutive days with one setup.

        Useful for session-level grouping, but **not** an independent unit —
        two trials of one group are still the same fish.
        """
        return (self.group, self.trial)


@dataclass(frozen=True, slots=True)
class FishDescriptor:
    """The experimenter's physical description of one animal."""

    role: Identity
    raw: str
    colours: frozenset[str]
    sizes: frozenset[str]

    @classmethod
    def parse(cls, role: Identity, raw: str) -> FishDescriptor:
        text = (raw or "").strip().lower()
        tokens = [t.strip() for t in _SEPARATORS.split(text) if t.strip()]
        colours = {c for c in COLOURS if any(c in t for t in tokens)}
        sizes: set[str] = set()
        for token in tokens:
            for size in SIZES:  # 'biggest' before 'big' so the longer wins
                if size in token:
                    sizes.add(size)
                    break
        return cls(role=role, raw=raw.strip(), colours=frozenset(colours), sizes=frozenset(sizes))

    @property
    def is_empty(self) -> bool:
        return not self.colours and not self.sizes


@dataclass(frozen=True, slots=True)
class Group:
    """One stock of three fish, as recorded by the experimenter."""

    name: str
    set_date: str
    female: FishDescriptor
    male_1: FishDescriptor
    male_2: FishDescriptor
    note: str = ""

    def descriptor(self, identity: Identity) -> FishDescriptor:
        match identity:
            case Identity.FEMALE:
                return self.female
            case Identity.MALE_1:
                return self.male_1
            case Identity.MALE_2:
                return self.male_2
            case _:
                raise CohortError(f"no descriptor for {identity!r}")

    @property
    def male_cue(self) -> str:
        """What visually separates the two males: colour, size, both, or neither.

        A group whose males differ only in *size* is a weak target for appearance
        re-identification: apparent size varies with pose, depth and camera, and
        the animals grow between days. Prefer colour-separated groups when
        building the first training set, and treat ``"none"`` groups as unusable
        for male identity ground truth.
        """
        colour_differs = self.male_1.colours != self.male_2.colours and (
            self.male_1.colours or self.male_2.colours
        )
        size_differs = self.male_1.sizes != self.male_2.sizes and (
            self.male_1.sizes or self.male_2.sizes
        )
        if colour_differs and size_differs:
            return "colour+size"
        if colour_differs:
            return "colour"
        if size_differs:
            return "size"
        return "none"


def load_groups(path: str | Path) -> dict[str, Group]:
    """Parse ``group_list.csv`` into per-group fish descriptors.

    Expected columns: ``SET DATE``, ``GROUP NAME``, ``FEMALE``, ``MALE1``,
    ``MALE2``, ``DETAIL``.
    """
    groups: dict[str, Group] = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            name = (row.get("GROUP NAME") or "").strip()
            if not name:
                continue
            if name in groups:
                raise CohortError(f"group {name!r} appears twice in {path}")
            groups[name] = Group(
                name=name,
                set_date=(row.get("SET DATE") or "").strip(),
                female=FishDescriptor.parse(Identity.FEMALE, row.get("FEMALE", "")),
                male_1=FishDescriptor.parse(Identity.MALE_1, row.get("MALE1", "")),
                male_2=FishDescriptor.parse(Identity.MALE_2, row.get("MALE2", "")),
                note=(row.get("DETAIL") or "").strip(),
            )
    return groups


#: Notes in `group_list.csv` implying a group did NOT get three fresh fish.
_REUSE_HINTS = ("only change", "give up female", "change female")


def groups_with_possible_shared_individuals(groups: dict[str, Group]) -> dict[str, str]:
    """Groups whose note suggests they reuse animals from another group.

    Returns ``{group_name: note}``. **Confirm each with the experimenter before
    treating these as independent units.**

    Concretely: group C is noted "give up female at 20260105" and group P, set on
    exactly 20260105, is noted "only change female" — and both list males *red*
    and *white*. If P is C's males with a new female, then C and P are not two
    independent replicates, and the two males appear across roughly three weeks.

    That is a statistical hazard and a re-identification opportunity at the same
    time, which is why this reports rather than decides.
    """
    return {
        name: group.note
        for name, group in sorted(groups.items())
        if any(hint in group.note.lower() for hint in _REUSE_HINTS)
    }


@dataclass(frozen=True, slots=True)
class Trial:
    """One trio of fish observed across consecutive days.

    **This is the independent unit for statistics.** Its days are repeated
    measures of the same animals.
    """

    group: str
    trial: int
    days: tuple[ExperimentId, ...]

    @property
    def key(self) -> tuple[str, int]:
        return (self.group, self.trial)

    @property
    def n_days(self) -> int:
        return len(self.days)

    def day(self, day: int) -> ExperimentId:
        for experiment in self.days:
            if experiment.day == day:
                return experiment
        raise CohortError(f"trial {self.key} has no day {day}")

    def holdout_split(self, test_day: int) -> tuple[tuple[ExperimentId, ...], ExperimentId]:
        """Split into training days and one held-out test day.

        This is the cross-day protocol: fit on some days, evaluate on another.
        Because it is the same animals on different days, a model that transfers
        has learned the individual rather than a within-video shortcut — which
        is exactly the leak that invalidated every appearance dataset in the
        retired repo (§4.4, "circular and leaky").
        """
        held = self.day(test_day)
        return tuple(e for e in self.days if e.day != test_day), held


@dataclass(frozen=True, slots=True)
class Trio:
    """All recordings of one group — i.e. of one set of three individuals.

    **This is the independent unit for statistics.** Everything inside it is a
    repeated measure.
    """

    group: str
    trials: tuple[Trial, ...]

    @property
    def experiments(self) -> tuple[ExperimentId, ...]:
        return tuple(e for t in self.trials for e in t.days)

    @property
    def n_days(self) -> int:
        return len(self.experiments)

    def trial(self, trial: int) -> Trial:
        for candidate in self.trials:
            if candidate.trial == trial:
                return candidate
        raise CohortError(f"group {self.group} has no trial {trial}")

    def holdout_by_trial(
        self, test_trial: int
    ) -> tuple[tuple[ExperimentId, ...], tuple[ExperimentId, ...]]:
        """Split by recording block: train on one trial, test on another.

        The strongest re-identification test available without new annotation.
        Trials of one group are separated by days to weeks — different sessions,
        different lighting, the fish having grown — yet provably the same
        animals. A model that transfers across this has learned the individual
        and nothing else.
        """
        held = self.trial(test_trial)
        train = tuple(e for t in self.trials if t.trial != test_trial for e in t.days)
        if not train:
            raise CohortError(f"group {self.group} has only trial {test_trial}; no training split")
        return train, held.days


def group_trios(folders: list[str], *, strict: bool = False) -> dict[str, Trio]:
    """Group recording folders into trios — the independent statistical unit."""
    trials = group_trials(folders, strict=strict)
    buckets: dict[str, list[Trial]] = {}
    for (group, _), trial in trials.items():
        buckets.setdefault(group, []).append(trial)
    return {
        group: Trio(group=group, trials=tuple(sorted(items, key=lambda t: t.trial)))
        for group, items in sorted(buckets.items())
    }


def group_trials(folders: list[str], *, strict: bool = False) -> dict[tuple[str, int], Trial]:
    """Group recording folders into trials.

    Args:
        strict: Raise on an unparseable folder instead of skipping it. Test
            recordings and calibration folders do not follow the convention.
    """
    buckets: dict[tuple[str, int], list[ExperimentId]] = {}
    for folder in folders:
        try:
            experiment = ExperimentId.parse(folder)
        except CohortError:
            if strict:
                raise
            continue
        buckets.setdefault(experiment.trial_key, []).append(experiment)
    return {
        key: Trial(group=key[0], trial=key[1], days=tuple(sorted(items, key=lambda e: e.day)))
        for key, items in sorted(buckets.items())
    }
