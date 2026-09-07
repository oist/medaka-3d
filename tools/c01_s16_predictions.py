"""C01 — read a full-length 3-node inference run once, cache it as arrays.

Two runs exist and both are read by this module, selected with ``--run``:

* ``bottomup`` — `s16`, `sleap_output_3node/s16_bu_K_1_all/predictions/`
* ``topdown`` — `c05`, `sleap_output_3node/c05_td_K_1_all/predictions/`

Each is **126 chunk ``.slp`` files**: 3 experiments x 3 views x 348,600 frames,
skeleton ``mouth -> pectoral -> tail``, **no tracks**, over the identical frames,
so the two architectures are comparable frame for frame. Detection is finished;
the unsolved problem is correspondence over time, and every tool in Phase C reads
the same detections. Re-parsing 126 ``.slp`` files per tool would cost ~5 minutes
each time and, worse, would give every tool its own chance to mis-key a frame.

**The cache is namespaced by run and the run is part of the directory path**, so
a tool that forgets to pass one gets the default rather than a silent mixture of
two architectures' predictions of the same frame.

**Keys are ``(experiment, view, frame_idx)`` and are built by
``s06_locked_benchmark_3node.frame_key``, never by index.** ``side1_cam`` and
``side2_cam`` share every ``frame_idx`` in this corpus, so a key that is right by
index and wrong by camera silently matches one camera's prediction to the other
camera's data. That mistake has already been made here once, produced 600-900 px
of fictitious error, and reached a published conclusion. The cache stores one
file per (experiment, view) and re-derives the pair from the file name *and* from
``frame_key`` on every chunk, refusing any chunk whose contents disagree with
either.

Nothing in this module scores anything. It is a reader.

Build the cache (~6 min per run, once)::

    .venv\\Scripts\\python.exe tools/c01_s16_predictions.py --build --run bottomup
    .venv\\Scripts\\python.exe tools/c01_s16_predictions.py --build --run topdown
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sleap_io as sio

sys.path.insert(0, str(Path(__file__).resolve().parent))

from s06_locked_benchmark_3node import frame_key

from medaka3d import load_paths
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS

#: The full-length inference runs this reads, relative to the bucket root that
#: holds `work`. Both cover the identical 3 experiments x 3 views x 348,600
#: frames, so they are comparable frame for frame rather than sample against
#: sample. The cache is namespaced by run: two architectures' predictions of the
#: same frame must never land in the same file.
RUNS: dict[str, Path] = {
    "bottomup": Path("sleap_output_3node") / "s16_bu_K_1_all" / "predictions",
    "topdown": Path("sleap_output_3node") / "c05_td_K_1_all" / "predictions",
}
DEFAULT_RUN = "bottomup"

#: The skeleton both runs emit, in order. Asserted per chunk, never assumed.
NODES: tuple[str, ...] = ("mouth", "pectoral", "tail")

#: Every experiment in the s16 run. All group K: one trio, three consecutive days.
EXPERIMENTS: tuple[str, ...] = (
    "20251225_K_1_1_FCMC",
    "20251225_K_1_2_FCMC",
    "20251225_K_1_3_FCMC",
)

#: Frames per experiment per view, from the s16 record. Checked, not trusted.
N_FRAMES = 348_600

CHUNK_RE = re.compile(
    r"^(?P<experiment>\S+?)__(?P<view>\w+_cam)__frames_(?P<start>\d+)-(?P<end>\d+)\.slp$"
)


@dataclass(frozen=True, slots=True)
class Detections:
    """Every predicted instance of one (experiment, view), sorted by frame.

    ``points`` is (n_instances, n_nodes, 2) in pixels, NaN for a node the model
    did not place. ``start``/``stop`` index into it: instances of frame ``i`` are
    ``points[start[i]:stop[i]]``. Frames with no detection have ``start == stop``,
    which is the honest representation of the 464 frames s16 left empty.
    """

    experiment: str
    view: str
    frame_idx: np.ndarray  # (n_instances,) int32
    points: np.ndarray  # (n_instances, n_nodes, 2) float32
    score: np.ndarray  # (n_instances,) float32
    n_frames: int
    nodes: tuple[str, ...] = NODES

    @property
    def start(self) -> np.ndarray:
        return np.searchsorted(self.frame_idx, np.arange(self.n_frames), side="left")

    @property
    def stop(self) -> np.ndarray:
        return np.searchsorted(self.frame_idx, np.arange(self.n_frames), side="right")

    def node(self, name: str) -> np.ndarray:
        """(n_instances, 2) pixels for one node."""
        return self.points[:, self.nodes.index(name), :]

    def counts(self) -> np.ndarray:
        """(n_frames,) instances per frame."""
        return np.bincount(self.frame_idx, minlength=self.n_frames)[: self.n_frames]


def chunk_files(root: Path) -> dict[tuple[str, str], list[Path]]:
    """Group chunk files by (experiment, view), ordered by start frame."""
    grouped: dict[tuple[str, str], list[tuple[int, Path]]] = {}
    for path in sorted(root.glob("*.slp")):
        match = CHUNK_RE.match(path.name)
        if match is None:
            raise SystemExit(f"unrecognised chunk name {path.name}")
        key = (match["experiment"], match["view"])
        grouped.setdefault(key, []).append((int(match["start"]), path))
    return {key: [p for _, p in sorted(items)] for key, items in grouped.items()}


def read_pair(paths: list[Path], experiment: str, view: str, n_frames: int) -> Detections:
    """Read every chunk of one (experiment, view) into flat arrays.

    Every chunk is checked against ``frame_key`` — the file name says which
    camera it is and so does the embedded video path, and they must agree. A
    chunk whose contents belong to the other side camera is refused rather than
    scored.
    """
    frames: list[np.ndarray] = []
    points: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    seen: set[int] = set()

    for path in paths:
        labels = sio.load_slp(str(path))
        node_names = tuple(node.name for node in labels.skeletons[0].nodes)
        if node_names != NODES:
            raise SystemExit(f"{path.name}: skeleton {node_names} is not {NODES}")
        if labels.tracks:
            raise SystemExit(
                f"{path.name}: already carries {len(labels.tracks)} tracks. These "
                f"predictions are meant to be untracked; scoring a tracker against "
                f"pre-tracked input would measure the wrong thing."
            )
        for labeled in labels.labeled_frames:
            key_experiment, key_view, key_frame = frame_key(labeled)
            if key_experiment != experiment or key_view != view:
                raise SystemExit(
                    f"{path.name}: frame_key resolves a frame to "
                    f"({key_experiment}, {key_view}) but the file name says "
                    f"({experiment}, {view}). Refusing to mix cameras."
                )
            if key_frame in seen:
                raise SystemExit(f"{path.name}: frame {key_frame} appears in two chunks")
            seen.add(key_frame)
            for instance in labeled.instances:
                frames.append(key_frame)
                points.append(np.asarray(instance.numpy(), dtype=np.float32))
                scores.append(float(getattr(instance, "score", np.nan)))

    frame_idx = np.asarray(frames, dtype=np.int32)
    order = np.argsort(frame_idx, kind="stable")
    return Detections(
        experiment=experiment,
        view=view,
        frame_idx=frame_idx[order],
        points=np.asarray(points, dtype=np.float32)[order],
        score=np.asarray(scores, dtype=np.float32)[order],
        n_frames=n_frames,
    )


def cache_path(work_dir: Path, experiment: str, view: str, run: str = DEFAULT_RUN) -> Path:
    """One file per (run, experiment, view). The run is in the *path*, not the name,
    so a tool that forgets to pass it cannot silently read the other architecture."""
    if run not in RUNS:
        raise SystemExit(f"unknown run {run!r}; known runs are {sorted(RUNS)}")
    return work_dir / "c01_s16_predictions" / run / f"{experiment}__{view}.npz"


def save(detections: Detections, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        experiment=detections.experiment,
        view=detections.view,
        frame_idx=detections.frame_idx,
        points=detections.points,
        score=detections.score,
        n_frames=detections.n_frames,
        nodes=np.asarray(detections.nodes),
    )


def load(work_dir: Path, experiment: str, view: str, run: str = DEFAULT_RUN) -> Detections:
    """Load one cached (run, experiment, view). Raises if the cache is not built."""
    path = cache_path(work_dir, experiment, view, run)
    if not path.is_file():
        raise SystemExit(
            f"no cache at {path}. Build it once with:\n"
            f"    .venv\\Scripts\\python.exe tools/c01_s16_predictions.py --build --run {run}"
        )
    data = np.load(path, allow_pickle=False)
    nodes = tuple(str(n) for n in data["nodes"])
    cached_view = str(data["view"])
    if cached_view != view or str(data["experiment"]) != experiment:
        raise SystemExit(
            f"{path} holds ({data['experiment']}, {cached_view}), not ({experiment}, {view})"
        )
    return Detections(
        experiment=experiment,
        view=view,
        frame_idx=data["frame_idx"],
        points=data["points"],
        score=data["score"],
        n_frames=int(data["n_frames"]),
        nodes=nodes,
    )


def load_all(work_dir: Path, experiments=EXPERIMENTS, views=VIEWS, run=DEFAULT_RUN) -> dict:
    """``{(experiment, view): Detections}`` for one whole run."""
    return {(e, v): load(work_dir, e, v, run) for e in experiments for v in views}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--build", action="store_true", help="Parse the .slp chunks and cache.")
    parser.add_argument("--run", choices=sorted(RUNS), default=DEFAULT_RUN,
                        help="Which full-length inference run to read.")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Override the s16 predictions directory (default: beside work_dir).",
    )
    args = parser.parse_args()

    paths = load_paths()
    root = args.predictions or (paths.work_dir.parent / RUNS[args.run])
    if not root.is_dir():
        raise SystemExit(f"no s16 predictions at {root}")

    out_dir = paths.work_dir / "c01_s16_predictions" / args.run
    grouped = chunk_files(root)
    print(f"{len(grouped)} (experiment, view) pairs across "
          f"{sum(len(v) for v in grouped.values())} chunk files at {root}")

    if not args.build:
        for (experiment, view) in sorted(grouped):
            path = cache_path(paths.work_dir, experiment, view, args.run)
            print(f"  {experiment} {view:<10} {'cached' if path.is_file() else 'NOT BUILT'}")
        return

    with provenance("s16_predictions", out_dir, name=f"c01_{args.run}") as record:
        record.params_from(args)
        record.param("predictions_root", str(root))
        record.param("run", args.run)
        summary = {}
        for (experiment, view), files in sorted(grouped.items()):
            started = time.time()
            detections = read_pair(files, experiment, view, N_FRAMES)
            path = cache_path(paths.work_dir, experiment, view, args.run)
            save(detections, path)
            counts = detections.counts()
            summary[f"{experiment}__{view}"] = {
                "chunks": len(files),
                "instances": int(detections.points.shape[0]),
                "frames_with_detections": int((counts > 0).sum()),
                "frames_empty": int((counts == 0).sum()),
                "mean_instances": float(counts.mean()),
                "eq3": int((counts == 3).sum()),
            }
            record.output(f"{experiment}__{view}", path)
            print(
                f"  {experiment} {view:<10} {len(files):>2} chunks  "
                f"{detections.points.shape[0]:>9,} instances  "
                f"empty {int((counts == 0).sum()):>4}  "
                f"=3 {100 * (counts == 3).mean():5.1f}%  "
                f"{time.time() - started:5.1f}s"
            )
        record.param("summary", summary)
    print(f"\ncached to {out_dir}")


if __name__ == "__main__":
    main()
