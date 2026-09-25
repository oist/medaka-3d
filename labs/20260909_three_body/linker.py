"""Shared Linker protocol adapter with explicitly bound independent observations."""

from __future__ import annotations

import numpy as np
from tracker import ThreeBodyTracker


class ThreeBodyLinker:
    """Use first-frame skeletons as seeds and return supported, unflagged tracks.

    ``observation_factory`` opens the independent native-frame stream when the
    shared harness invokes ``link``. Only the first frame's instances can seed
    the model; later instances never enter its state or observation likelihood.
    Extra diagnostics are retained in ``result``, not interpreted as identity
    truth by this adapter. Bind a new adapter for another experimental run.
    """

    name = "three_body_analysis_by_synthesis"

    def __init__(self, rig, first_frame, observation_factory, config=None):
        if not isinstance(first_frame, int) or first_frame < 0:
            raise ValueError("first_frame must be a nonnegative source frame integer")
        if not callable(observation_factory):
            raise TypeError("bind a factory for an independent native observation stream")
        self.rig = rig
        self.first_frame = first_frame
        self.observation_factory = observation_factory
        self.config = config
        self.result = None
        self.tracker = None

    def link(self, instances):
        if self.tracker is not None:
            raise RuntimeError("use a new linker for each run; no hidden reset is permitted")
        seeds = sorted((s for s in instances if s.frame == self.first_frame), key=lambda s: s.slot)
        if len(seeds) != 3 or len({s.slot for s in seeds}) != 3:
            raise ValueError("exactly three distinct first-frame skeletons are required")
        initial = np.stack([s.xyz for s in seeds])
        self.tracker = ThreeBodyTracker(self.rig, initial, self.config)

        def observations():
            for t, frame in enumerate(self.observation_factory()):
                if frame.frame_idx != self.first_frame + t:
                    raise ValueError("observation schedule must start at the registered seed frame")
                yield frame

        self.result = self.tracker.track(observations())
        return self.result["accepted_xyz"]
