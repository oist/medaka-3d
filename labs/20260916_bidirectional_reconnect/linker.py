"""Shared Linker wrapper retaining a baseline while recording unaccepted audits.

The Linker protocol accepts arbitrary-slot instances rather than linked tracks.
Consequently this adapter requires an explicit baseline Linker; it never treats
``Instance3D.slot`` as pre-established temporal continuity. The source-aware
frozen C34 experiment remains ``run.py``: source rows, slot_track, and all pose
nodes are outside the Linker return contract and are not rewritten here.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
from directions import DirectionConfig, diagnose, episodes
from reconnect import ReconnectConfig, audit_transient, audit_window

from medaka3d.stages import Linker


class BaselineDiagnosticLinker:
    """Pass through an explicit baseline and attach geometry-only diagnostics.

    Bind the native frame schedule of the baseline output at construction. The
    shared benchmark receives an unchanged copy of its trajectories; proposals
    live only in ``result`` and cannot change benchmark identity or coverage.
    This is an offline audit and makes no claim of online causality or new IDs.
    """

    name = "bidirectional_baseline_diagnostic_v1"

    def __init__(
        self,
        baseline,
        frame_idx,
        support=None,
        *,
        direction_config=None,
        reconnect_config=None,
        core_start=None,
        core_stop=None,
    ):
        if not isinstance(baseline, Linker):
            raise TypeError("an explicit baseline implementing the Linker protocol is required")
        frames = np.asarray(frame_idx)
        if (
            frames.ndim != 1
            or not len(frames)
            or not np.isfinite(frames).all()
            or np.any(frames != np.floor(frames))
            or np.any(np.diff(frames) != 1)
        ):
            raise ValueError("frame_idx must be nonempty contiguous integer native frames")
        self.baseline = baseline
        self.frame_idx = frames.astype(np.int64, copy=True)
        self.frame_idx.flags.writeable = False
        self.support = None if support is None else np.array(support, dtype=bool, copy=True)
        if self.support is not None:
            if self.support.shape != (len(frames), 3):
                raise ValueError("support must have shape (frames, 3)")
            self.support.flags.writeable = False
        self.direction_config = direction_config or DirectionConfig()
        self.reconnect_config = reconnect_config or ReconnectConfig()
        self.core_start = int(frames[0]) if core_start is None else core_start
        self.core_stop = int(frames[-1]) + 1 if core_stop is None else core_stop
        if (
            not isinstance(self.core_start, (int, np.integer))
            or not isinstance(self.core_stop, (int, np.integer))
            or not frames[0] <= self.core_start < self.core_stop <= frames[-1] + 1
        ):
            raise ValueError("core interval must lie inside the bound native frame schedule")
        self.result = None

    def link(self, instances):
        """Return unchanged baseline trajectories and retain audits in ``result``."""
        # Do not retain a previous successful report when a new invocation fails.
        self.result = None
        tracks = np.asarray(self.baseline.link(instances))
        if tracks.shape != (len(self.frame_idx), 3, 3):
            raise ValueError("baseline output must match the bound (frames, 3, 3) schedule")
        unchanged = tracks.copy()
        directional = diagnose(tracks, self.frame_idx, self.direction_config)
        groups = episodes(directional, self.core_start, self.core_stop)
        first_frame = int(self.frame_idx[0])
        for group in groups:
            lo = group["start_frame"] - first_frame
            hi = group["stop_frame_exclusive"] - first_frame - 1
            group["transient_audits"] = []
            if lo < 0 or hi >= len(tracks):
                group["persistent_audit"] = {
                    "status": "insufficient_reference",
                    "reason": "padded_episode_outside_bound_frame_schedule",
                    "suggested_permutation": None,
                    "independent_identity_validated": False,
                }
                continue
            group["persistent_audit"] = audit_window(
                tracks, self.frame_idx, lo, hi, self.support, self.reconnect_config
            )
            for entry in sorted({edge["frame"] for edge in group["edges"]}):
                for length in (1, 2, 3):
                    start, stop = entry - first_frame, entry - first_frame + length - 1
                    group["transient_audits"].append({
                        "entry_frame": entry,
                        "length": length,
                        "audit": audit_transient(
                            tracks,
                            self.frame_idx,
                            start,
                            stop,
                            self.support,
                            self.reconnect_config,
                            reference_window=(lo, hi),
                        ),
                    })
        self.result = {
            "algorithm": self.name,
            "baseline": self.baseline.name,
            "frame_idx": self.frame_idx.copy(),
            "core_start": int(self.core_start),
            "core_stop": int(self.core_stop),
            "direction_config": asdict(self.direction_config),
            "reconnect_config": asdict(self.reconnect_config),
            "directional": directional,
            "episodes": groups,
            "accepted_identity_corrections": 0,
            "baseline_preserved": True,
            "scope": "offline geometry proposals; no independently validated identity repair",
            "source_metadata": "outside Linker return contract; authoritative audit is run.py",
        }
        return unchanged
