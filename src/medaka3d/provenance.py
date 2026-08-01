"""Provenance records written beside every pipeline output.

The old repo could not answer "which model, which calibration, which code
produced this CSV" without reading prose documentation that had drifted out of
date. Every stage here writes a sidecar record so the answer is always next to
the artifact.

Usage::

    with provenance("reconstruct_3d", paths, out_dir) as rec:
        rec.input("sleap_top", top_slp)
        rec.param("max_jump_mm", 8)
        ...
        rec.output("tracks", tracks_csv)

The record is written on exit, including on failure (with the traceback and
``status="failed"``), because a crashed run you cannot diagnose is worse than no
run at all.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SIDECAR_SUFFIX = ".provenance.json"


@dataclass
class Record:
    """Accumulates what a stage consumed, chose, and produced."""

    stage: str
    started_at: str
    inputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    status: str = "running"
    error: str | None = None

    def input(self, name: str, path: str | Path, *, sha256: str | None = None) -> None:
        """Register a consumed file. Size is recorded; hashing is opt-in (cost)."""
        self.inputs[name] = _describe(path, sha256=sha256)

    def output(self, name: str, path: str | Path, *, sha256: str | None = None) -> None:
        """Register a produced file."""
        self.outputs[name] = _describe(path, sha256=sha256)

    def param(self, name: str, value: Any) -> None:
        """Register a decision that affects the result."""
        self.params[name] = value

    def params_from(self, namespace: Any) -> None:
        """Register every public attribute of an argparse Namespace."""
        for key, value in vars(namespace).items():
            if not key.startswith("_"):
                self.params[key] = value if _jsonable(value) else str(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": _now(),
            "inputs": self.inputs,
            "outputs": self.outputs,
            "params": self.params,
            "environment": self.environment,
            **({"error": self.error} if self.error else {}),
        }


@contextmanager
def provenance(stage: str, out_dir: str | Path, *, name: str | None = None) -> Iterator[Record]:
    """Write a provenance sidecar for one stage invocation.

    Args:
        stage: Stage identifier, e.g. ``"reconstruct_3d"``.
        out_dir: Directory the sidecar is written into.
        name: Sidecar basename. Defaults to ``stage``.
    """
    record = Record(stage=stage, started_at=_now())
    record.environment = _environment()
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"{name or stage}{SIDECAR_SUFFIX}"
    try:
        yield record
    except BaseException as exc:
        record.status = "failed"
        record.error = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        raise
    else:
        record.status = "ok"
    finally:
        target.write_text(
            json.dumps(record.to_dict(), indent=2, sort_keys=False, default=str),
            encoding="utf-8",
        )


def _describe(path: str | Path, *, sha256: str | None = None) -> dict[str, Any]:
    resolved = Path(path)
    entry: dict[str, Any] = {"path": str(resolved)}
    if resolved.is_file():
        stat = resolved.stat()
        entry["size"] = stat.st_size
        entry["mtime"] = datetime.fromtimestamp(stat.st_mtime, UTC).isoformat()
    else:
        entry["missing"] = True
    if sha256:
        entry["sha256"] = sha256
    return entry


def _environment() -> dict[str, Any]:
    return {
        "argv": sys.argv,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "git_rev": _git_rev(),
        "git_dirty": _git_dirty(),
    }


def _git_rev() -> str | None:
    return _git("rev-parse", "HEAD")


def _git_dirty() -> bool | None:
    status = _git("status", "--porcelain")
    return None if status is None else bool(status)


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _jsonable(value: Any) -> bool:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


def _now() -> str:
    return datetime.now(UTC).isoformat()
