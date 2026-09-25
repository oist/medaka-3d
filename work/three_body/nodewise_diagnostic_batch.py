"""Bounded lossless batches of frozen instance and Nodewise body diagnostics.

Only diagnostic exports and serialization are changed. All frames and all codec
fields remain, with the old compact candidate/count rows retained. The writer
does not advance, fit, restore, finish, change costs or alter tracker diagnostics.
"""

from __future__ import annotations

import math

import numpy as np
import run_nodewise_body_row_negative_continuation as frozen

codec, proof, require = frozen.diagnostic, frozen.proof, frozen.require
FROZEN_CALLER_SHA = "4f2d31a383a8d59320ef0c09f8dcddf645dfb20e0849bd2f1dc27398d719669e"
CODEC_SHA = "91a66646f89a6abb59b6d60bc1b357b383167c21eb9d40bc47491802d05e9544"
SCHEMA = "nodewise_lossless_diagnostic_batch_v1"
TIMING_FIELDS = (
    "inherited_proposal_seconds",
    "pool_generation_seconds",
    "scene_generation_seconds",
    "generation_seconds",
)


def compact_row(instance, body, frame_idx):
    """Old save_frame_diagnostics row plus separately named body timing scalars."""
    require(
        instance["schema"] == "instance_rescore_frame_diagnostics_v1"
        and body["schema"] == "body_row_frame_diagnostics_v1"
        and instance["frame_idx"] == body["frame_idx"] == frame_idx
        and body["frame_completed"] is True,
        "both complete diagnostics must describe the requested native frame",
    )
    counts = frozen.body_row_counts(body)
    indices = np.asarray(instance["candidate_index"])
    require(
        indices.ndim == 1
        and indices.dtype.kind in "iu"
        and np.array_equal(indices, np.arange(len(indices)))
        and len(indices) == counts["original_count"] + counts["admitted"],
        "instance/body candidate indexing or admission differs",
    )
    timings = {k: body[k] for k in TIMING_FIELDS}
    for value in (
        *timings.values(),
        instance["gate_preparation_seconds"],
        instance["candidate_scoring_seconds"],
    ):
        require(
            isinstance(value, (int, float, np.integer, np.floating))
            and math.isfinite(value)
            and value >= 0,
            "finite nonnegative diagnostic timing required",
        )
    return dict(
        frame_idx=frame_idx,
        candidate_count=len(indices),
        gate_preparation_seconds=instance["gate_preparation_seconds"],
        candidate_scoring_seconds=instance["candidate_scoring_seconds"],
        eligible_views=[view for view, row in instance["views"].items() if row["eligible"]],
        body_row=counts,
        body_row_timing_seconds=timings,
    )


class NodewiseDiagnosticBatch:
    """Append consecutive completed frames; flush automatically at100 or earlier.

    batch_size may be1..100. Call flush() at every source chunk end and before
    sealing. records/totalframes count accepted frame exports; saved_frames counts
    fully verified batches. Any append/flush failure poisons the writer, leaving
    partial artifacts for the caller's failed seal and prohibiting retry.
    """

    def __init__(self, out, batch_size=100):
        require(type(batch_size) is int and 1 <= batch_size <= 100, "batch size must be1..100")
        self.out = out
        self.batch_size = batch_size
        self.records = []
        self.batches = []
        self.totalframes = 0
        self.saved_frames = 0
        self._pending = []
        self._next_frame = None
        self._failed = False

    @property
    def pending_frames(self):
        return len(self._pending)

    def _usable(self):
        require(not self._failed, "diagnostic writer previously failed; no retry")

    def append(self, tracker, frame_idx):
        self._usable()
        try:
            require(
                type(frame_idx) is int
                and frame_idx >= 0
                and (self._next_frame is None or frame_idx == self._next_frame),
                "strict consecutive nonnegative native frame order required",
            )
            require(
                tracker.previous_frame == frame_idx, "tracker is not at requested completed frame"
            )
            # Both frozen exporters deep-copy their completed current-frame data.
            # Never read or mutate the live last_* diagnostic dictionaries here.
            instance = tracker.export_instance_diagnostics()
            body = tracker.export_body_row_diagnostics()
            row = compact_row(instance, body, frame_idx)
            self._pending.append(dict(frame_idx=frame_idx, instance=instance, body_row=body))
            self.records.append(row)
            self.totalframes += 1
            self._next_frame = frame_idx + 1
            if len(self._pending) == self.batch_size:
                self.flush()
            return row
        except BaseException:
            self._failed = True
            raise

    def flush(self):
        self._usable()
        if not self._pending:
            return None
        try:
            first = self._pending[0]["frame_idx"]
            stop = self._pending[-1]["frame_idx"] + 1
            require(
                0 < len(self._pending) <= self.batch_size
                and [r["frame_idx"] for r in self._pending] == list(range(first, stop)),
                "batch native coverage differs",
            )
            tree, arrays = codec.split_diagnostic(dict(schema=SCHEMA, frames=self._pending))
            prefix = f"diagnostic_batches/f{first:06d}-{stop:06d}"
            archive_ref = self.out.write(prefix + ".npz", arrays)
            document = dict(
                schema=SCHEMA,
                first_frame=first,
                stop_frame_exclusive=stop,
                frame_count=len(self._pending),
                declared_batch_size=self.batch_size,
                diagnostic_families=["instance", "body_row"],
                all_fields_all_frames_preserved=True,
                archive=archive_ref,
                tree=tree,
            )
            json_ref = self.out.write(prefix + ".json", document)
            saved_document = proof.read_json(json_ref["path"])
            require(
                proof.json_bytes(saved_document) == proof.json_bytes(document),
                "serialized diagnostic metadata differs",
            )
            with np.load(archive_ref["path"], allow_pickle=False) as archive:
                require(
                    len(archive.files) == len(set(archive.files)) == len(arrays)
                    and set(archive.files) == set(arrays),
                    "diagnostic archive membership differs",
                )
                restored = codec.restore_diagnostic(
                    saved_document["tree"], {key: archive[key] for key in archive.files}
                )
            require(
                restored["schema"] == SCHEMA
                and [r["frame_idx"] for r in restored["frames"]] == list(range(first, stop)),
                "complete batch roundtrip changed native frame order",
            )
            for frame in restored["frames"]:
                compact_row(frame["instance"], frame["body_row"], frame["frame_idx"])
            record = dict(
                first_frame=first,
                stop_frame_exclusive=stop,
                frame_count=len(self._pending),
                array_fields=len(arrays),
                archive=archive_ref,
                metadata=json_ref,
                full_codec_roundtrip_verified=True,
            )
            self.batches.append(record)
            self.saved_frames += len(self._pending)
            self._pending.clear()
            return record
        except BaseException:
            self._failed = True
            raise
