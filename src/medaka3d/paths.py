"""Machine-specific path resolution.

Design rule inherited from the post-mortem of `medaka-mating-pipeline`: a path
that no longer exists must fail loudly at load time. The old repo shipped a
config pointing at SLEAP models that had been deleted months earlier and nothing
ever complained, because every consumer resolved paths lazily and fell back on a
default. Validation here is eager and total.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

# Keys every machine must define. Adding one here makes it mandatory everywhere.
REQUIRED_KEYS: tuple[str, ...] = (
    "raw_data_dir",  # per-experiment recording folders
    "asset_root",  # curated models + training data + ground truth
    "work_dir",  # this machine's scratch for pipeline outputs
)

OPTIONAL_KEYS: tuple[str, ...] = (
    "sleap_output_dir",
    "archive_dir",
    "boris_events_dir",
)


class PathConfigError(RuntimeError):
    """Raised when the path config is missing, malformed, or points at nothing."""


@dataclass(frozen=True, slots=True)
class Paths:
    """Resolved, existence-checked roots for one machine."""

    raw_data_dir: Path
    asset_root: Path
    work_dir: Path
    sleap_output_dir: Path | None = None
    archive_dir: Path | None = None
    boris_events_dir: Path | None = None
    source: Path | None = None

    def experiment_dir(self, experiment: str) -> Path:
        """Directory for one recording session, verified to exist."""
        path = self.raw_data_dir / experiment
        if not path.is_dir():
            raise PathConfigError(f"No such experiment {experiment!r} under {self.raw_data_dir}")
        return path

    def video(self, experiment: str, view: str) -> Path:
        """The single .mp4 for one view of one experiment.

        Views are the recording folder names: ``top_cam``, ``side1_cam``,
        ``side2_cam``. Exactly one video must match; ambiguity is an error
        rather than a silent ``[0]``.
        """
        view_dir = self.experiment_dir(experiment) / view
        if not view_dir.is_dir():
            raise PathConfigError(f"No such view {view!r} in {self.experiment_dir(experiment)}")
        videos = sorted(view_dir.glob("*.mp4"))
        if len(videos) != 1:
            raise PathConfigError(
                f"Expected exactly 1 .mp4 in {view_dir}, found {len(videos)}: "
                f"{[v.name for v in videos]}"
            )
        return videos[0]


def load_paths(config: str | Path | None = None) -> Paths:
    """Load and validate the local path config.

    Args:
        config: Path to a TOML file. Defaults to ``configs/paths.local.toml``
            relative to the repository root.

    Raises:
        PathConfigError: If the file is absent, a required key is missing, or
            any configured directory does not exist on this machine.
    """
    path = Path(config) if config else _repo_root() / "configs" / "paths.local.toml"
    if not path.is_file():
        raise PathConfigError(
            f"No path config at {path}. Copy configs/paths.example.toml to "
            f"configs/paths.local.toml and fill in this machine's roots."
        )

    with path.open("rb") as handle:
        table = tomllib.load(handle).get("paths", {})

    missing = [key for key in REQUIRED_KEYS if not table.get(key)]
    if missing:
        raise PathConfigError(f"{path}: missing required key(s): {', '.join(missing)}")

    unknown = set(table) - set(REQUIRED_KEYS) - set(OPTIONAL_KEYS)
    if unknown:
        raise PathConfigError(
            f"{path}: unrecognised key(s): {', '.join(sorted(unknown))}. "
            f"Known keys: {', '.join(REQUIRED_KEYS + OPTIONAL_KEYS)}"
        )

    resolved: dict[str, Path] = {}
    absent: list[str] = []
    for key in REQUIRED_KEYS + OPTIONAL_KEYS:
        raw = table.get(key)
        if not raw:
            continue
        directory = Path(raw)
        # work_dir is the one root we are allowed to create.
        if key == "work_dir":
            directory.mkdir(parents=True, exist_ok=True)
        if not directory.is_dir():
            absent.append(f"  {key} = {directory}")
            continue
        resolved[key] = directory

    if absent:
        raise PathConfigError(
            f"{path}: configured director"
            f"{'y does' if len(absent) == 1 else 'ies do'} not exist on this machine:\n"
            + "\n".join(absent)
            + "\n\nIf a network share is simply not mounted, mount it; do not edit the "
            "config to point somewhere else."
        )

    return Paths(source=path, **resolved)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]
