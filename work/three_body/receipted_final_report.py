"""Pure final-report arithmetic on already audited candidates/support blocks.

Slot names are saved tracker slots, not biological identity truth. This module
does not validate tracking lineage or declare the final audit complete. The
caller binds this source and retains the independent support/chain evidence.
Residual input is read-only; only synthetic archives are used by its tests.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path

import audit_long_final
import numpy as np

NODES = ("mouth", "pectoral", "tail")
METRICS = ("ray_distance_mm", "native_pixel_error")
PROBABILITIES = (0.0, 0.5, 0.9, 0.95, 0.99, 1.0)
MAX_SUPPORTED_VALUES = 8_067_600


def require(condition, message):
    if not condition:
        raise ValueError(message)


def native_frames(value):
    frames = np.asarray(value)
    require(
        frames.dtype == np.int64
        and frames.ndim == 1
        and len(frames) > 0
        and frames[0] >= 0
        and frames[-1] < np.iinfo(np.int64).max
        and np.all(np.diff(frames) == 1),
        "nonempty contiguous nonnegative int64 native frames required",
    )
    return frames


def missing_intervals(frames, supported):
    """All maximal missing runs, native half-open, without block-boundary resets."""
    missing = ~supported
    transitions = np.diff(np.r_[False, missing, False].astype(np.int8))
    starts, stops = np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)
    first = int(frames[0])
    return [[first + int(lo), first + int(hi)] for lo, hi in zip(starts, stops, strict=True)]


def step_metrics(pose, supported):
    """Adjacent native frames only; neither metric bridges missing observations."""
    with np.errstate(over="ignore", invalid="ignore"):
        steps = np.linalg.norm(np.diff(pose.astype(np.float64), axis=0), axis=-1)
    require(np.isfinite(steps).all(), "nonfinite derived adjacent pose distance")
    both = supported[:-1] & supported[1:]
    return (
        dict(count=len(steps), max_mm=float(steps.max()) if len(steps) else None),
        dict(count=int(both.sum()), max_mm=float(steps[both].max()) if both.any() else None),
    )


def candidate_metrics(candidate):
    """Report all 3 slots x 3 nodes and preserve frozen direct-count semantics.

    T must be positive. Latent poses are finite; xyz and accepted_xyz must be
    whole finite or whole NaN points. The frozen helper verifies their finite
    masks against observed pectorals and review exclusion. Empty supports are
    valid and yield the entire interval as one missing run.
    """
    frames = native_frames(candidate["frame_idx"])
    count = len(frames)
    pose, observed, review = (
        np.asarray(candidate[key]) for key in ("pose_xyz", "observed", "identity_review")
    )
    require(
        pose.shape == (count, 3, 3, 3) and pose.dtype.kind == "f" and np.isfinite(pose).all(),
        "finite floating T x 3 x 3 x 3 latent poses required",
    )
    require(
        observed.shape == (count, 3, 3) and observed.dtype == np.bool_,
        "boolean T x 3 x 3 observed mask required",
    )
    require(
        review.shape == (count, 3) and review.dtype == np.bool_,
        "boolean T x 3 identity_review required",
    )
    for key in ("xyz", "accepted_xyz"):
        xyz = np.asarray(candidate[key])
        require(
            xyz.shape == (count, 3, 3)
            and xyz.dtype.kind == "f"
            and np.all(np.isfinite(xyz).all(-1) | np.isnan(xyz).all(-1)),
            key + " must contain whole finite or whole NaN pectoral points",
        )
    direct = audit_long_final.SUPPORT.direct_counts(candidate)
    slots = []
    for slot in range(3):
        nodes = []
        for node, name in enumerate(NODES):
            supported = observed[:, slot, node]
            latent, adjacent = step_metrics(pose[:, slot, node], supported)
            gaps = missing_intervals(frames, supported)
            nodes.append(
                dict(
                    node=name,
                    supported_count=int(supported.sum()),
                    missing_count=int((~supported).sum()),
                    missing_intervals=gaps,
                    longest_missing_frames=max((hi - lo for lo, hi in gaps), default=0),
                    latent_adjacent_steps=latent,
                    both_supported_adjacent_steps=adjacent,
                )
            )
        accepted = np.isfinite(candidate["accepted_xyz"][:, slot]).all(-1)
        slots.append(
            dict(
                slot=slot,
                identity_review_count=int(review[:, slot].sum()),
                accepted_pectoral_count=int(accepted.sum()),
                accepted_pectoral_missing_intervals=missing_intervals(frames, accepted),
                nodes=nodes,
            )
        )
    return dict(
        schema="receipted_final_candidate_metrics_v1",
        frame_interval=[int(frames[0]), int(frames[-1]) + 1],
        native_frames=count,
        node_denominator=count * 9,
        pectoral_denominator=count * 3,
        direct_counts=direct,
        per_slot=slots,
        gap_interval_convention="native half-open",
        step_convention="adjacent native frames only; no missing-gap bridging",
        biological_ID_truth=False,
        final_audit_complete=False,
    )


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def checked_block(record, root):
    supplied = Path(record["path"])
    path = (supplied if supplied.is_absolute() else root / supplied).resolve(strict=True)
    require(path.is_relative_to(root) and path.is_file(), "residual block outside output root")
    expected = record["sha256"]
    require(
        isinstance(expected, str)
        and len(expected) == 64
        and all(c in "0123456789abcdef" for c in expected),
        "residual block SHA256 required",
    )
    require(file_sha(path) == expected, "residual block SHA differs")
    if "size_bytes" in record:
        require(path.stat().st_size == record["size_bytes"], "residual block size differs")
    with np.load(path, allow_pickle=False) as archive:
        require(
            len(archive.files) == len(set(archive.files))
            and set(archive.files) == {"frame_idx", "valid_source", *METRICS},
            "residual block array schema differs",
        )
        yield archive
    require(file_sha(path) == expected, "residual block changed during reading")


def block_axes(archive):
    frames = native_frames(archive["frame_idx"])
    valid = archive["valid_source"]
    require(
        valid.dtype == np.bool_ and valid.shape == (len(frames), 3, 3, 3),
        "boolean T x 3 x 3 x 3 valid_source required",
    )
    return frames, valid


def metric_values(archive, name, shape):
    values = archive[name]
    require(
        values.dtype == np.float64 and values.shape == shape,
        name + " must be float64 with the valid_source shape",
    )
    return values


def residual_metrics(block_records, output_root):
    """Exact finite-supported residual quantiles from ordered, hashed NPZ blocks.

    Each record has path, sha256 and optional size_bytes. Invalid source entries
    never enter statistics, even if finite or extreme. Nonfinite valid-source
    entries are counted separately. Quantiles use NumPy's linear method with
    one in-place global float64 buffer, at most 8,067,600 values, plus the current
    source block and boolean masks. The two global metrics never coexist. No
    report files are written; the caller owns the final output/seal guards.
    """
    records = list(block_records)
    require(bool(records), "nonempty ordered residual blocks required")
    root = Path(output_root).resolve(strict=True)
    require(root.is_dir(), "residual output root must be a directory")
    counts = dict.fromkeys(METRICS, 0)
    block_counts = []
    first, stop, total, valid_count = None, None, 0, 0
    for record in records:
        with checked_block(record, root) as archive:
            frames, valid = block_axes(archive)
            require(stop is None or frames[0] == stop, "residual block native order/gap differs")
            first = int(frames[0]) if first is None else first
            stop = int(frames[-1]) + 1
            total += len(frames)
            require(total * 27 <= MAX_SUPPORTED_VALUES, "residual native denominator exceeds cap")
            valid_count += int(valid.sum())
            row_counts = {}
            for name in METRICS:
                values = metric_values(archive, name, valid.shape)
                row_counts[name] = int(np.count_nonzero(valid & np.isfinite(values)))
                counts[name] += row_counts[name]
                del values
            block_counts.append(row_counts)
    metrics = {}
    for name in METRICS:
        values = np.empty(counts[name], dtype=np.float64)
        offset = 0
        for record, row_counts in zip(records, block_counts, strict=True):
            with checked_block(record, root) as archive:
                _, valid = block_axes(archive)
                block = metric_values(archive, name, valid.shape)
                selected = valid & np.isfinite(block)
                count = row_counts[name]
                require(int(selected.sum()) == count, "residual finite-supported count changed")
                np.compress(selected.ravel(), block.ravel(), out=values[offset : offset + count])
                offset += count
                del block
        require(offset == len(values), "residual value collection incomplete")
        quantiles = (
            np.quantile(values, PROBABILITIES, method="linear", overwrite_input=True).tolist()
            if len(values)
            else None
        )
        metrics[name] = dict(
            finite_supported_count=len(values),
            nonfinite_supported_count=valid_count - len(values),
            max=quantiles[-1] if quantiles is not None else None,
            quantile_probabilities=list(PROBABILITIES),
            quantiles=quantiles,
        )
        del values
    return dict(
        schema="receipted_final_residual_metrics_v1",
        frame_interval=[first, stop],
        native_frames=total,
        blocks=len(records),
        view_node_denominator=total * 27,
        valid_source_count=valid_count,
        metrics=metrics,
        quantile_method="linear",
        invalid_source_entries_included=False,
        maximum_global_supported_values=MAX_SUPPORTED_VALUES,
        biological_ID_truth=False,
        final_audit_complete=False,
    )
