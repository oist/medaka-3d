"""Lossless three-family batches for the observation-reserve experiment.

Reuse the frozen two-family writer's bounded lifecycle and compact counts.
Only append/flush serialization is extended; no tracker state or cost changes.
"""

from __future__ import annotations

import copy
import math

import nodewise_diagnostic_batch as legacy
import numpy as np

codec, proof, require = legacy.codec, legacy.proof, legacy.require
LEGACY_BATCH_SHA = "f494819316ea4b1000378c8672c0491951b065980a43468143d80b016c896bab"
TRACKER_SHA = "7b03698dacda5d18be87ff99c0bccd47474a94db08ae89aff0231f4dcb50c74a"
SCHEMA = "observation_reserve_lossless_diagnostic_batch_v1"
FAMILIES = ("instance", "body_row", "observation_reserve")


def compact_row(instance, body, reserve, frame_idx):
    row = legacy.compact_row(instance, body, frame_idx)
    indices = np.asarray(reserve["candidate_index"])
    eligible = np.asarray(reserve["eligible"])
    count = row["candidate_count"]
    require(
        reserve["schema"] == "observation_reserve_frame_diagnostics_v1"
        and reserve["frame_idx"] == frame_idx
        and reserve["frame_completed"] is True
        and indices.dtype.kind in "iu"
        and indices.shape == (count,)
        and np.array_equal(indices, instance["candidate_index"])
        and eligible.dtype == np.bool_
        and eligible.shape == (count,)
        and type(reserve["support_evaluation_count"]) is int
        and reserve["support_evaluation_count"] == count
        and reserve["selected_support_reused"] is True,
        "complete reserve diagnostic must match instance/body candidates and support cache",
    )
    selection = reserve["selection"]
    baseline, selected = selection["baseline_indices"], selection["selected_indices"]
    require(
        isinstance(baseline, list)
        and isinstance(selected, list)
        and 1 <= len(baseline) == len(selected) <= 6
        and all(type(i) is int and 0 <= i < count for i in baseline + selected)
        and len(set(baseline)) == len(baseline)
        and len(set(selected)) == len(selected)
        and selected[:-1] == baseline[:-1]
        and selected[0] == baseline[0] == reserve["baseline_best_index"]
        and type(selection["changed"]) is bool
        and selection["changed"] == (selected != baseline)
        and (not selection["changed"] or len(selected) == 6)
        and selection["current_best_preserved"] is True
        and selection["real_costs_modified"] is False,
        "reserve selection must preserve baseline prefix and state zero",
    )
    timings = {key: reserve[key] for key in ("support_seconds", "selection_seconds")}
    for value in timings.values():
        require(
            isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
            and math.isfinite(value)
            and value >= 0,
            "finite nonnegative reserve timing required",
        )
    row["observation_reserve"] = dict(
        eligible_candidate_count=int(eligible.sum()),
        baseline_indices=baseline.copy(),
        selected_indices=selected.copy(),
        changed=selection["changed"],
        reservation_count=int(selection["changed"]),
        baseline_best_index=reserve["baseline_best_index"],
        support_evaluation_count=reserve["support_evaluation_count"],
        reason=selection["reason"],
        replacement=copy.deepcopy(selection["replacement"]),
    )
    row["observation_reserve_timing_seconds"] = timings
    return row


class ObservationReserveDiagnosticBatch(legacy.NodewiseDiagnosticBatch):
    """Same 1..100-frame bound and poison-on-failure policy; all three exports."""

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
            # These exporters return owned copies of completed current-frame data.
            instance = tracker.export_instance_diagnostics()
            body = tracker.export_body_row_diagnostics()
            reserve = tracker.export_observation_reserve_diagnostics()
            row = compact_row(instance, body, reserve, frame_idx)
            self._pending.append(
                dict(
                    frame_idx=frame_idx,
                    instance=instance,
                    body_row=body,
                    observation_reserve=reserve,
                )
            )
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
                diagnostic_families=list(FAMILIES),
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
                compact_row(
                    frame["instance"],
                    frame["body_row"],
                    frame["observation_reserve"],
                    frame["frame_idx"],
                )
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
