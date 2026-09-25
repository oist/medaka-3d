"""Batch the frozen pectoral-only identity comparison without changing its rule.

These are geometric permutation differences, not evidence of biological identity.
The six permutations retain itertools' lexicographic ordering, so exactly tied
costs select the same first permutation as the original tuple ``min``.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np

PERMUTATIONS = np.asarray(list(permutations(range(3))), dtype=np.int64)
PERMUTATIONS.setflags(write=False)


def identity_costs_batch(first, second):
    """Return (A,B,6) costs for first[A,3,3,3] against second[B,3,3,3].

    The frozen rule is sum(norm(delta)**2), not a squared-distance shortcut.
    In NumPy, scalar **2 can round differently from ndarray **2. Keep scalar
    powers and left-to-right additions exactly, while sharing the nine distinct
    pectoral distances among the six permutations. Batched 1x3 @ 3x1 reproduces
    the one-dimensional norm's dot product before square root, without requiring
    the newer np.vecdot API. Inputs retain their floating dtype.
    """
    first, second = np.asarray(first), np.asarray(second)
    for name, value in (("first", first), ("second", second)):
        if value.ndim != 4 or value.shape[1:] != (3, 3, 3):
            raise ValueError(f"{name} must have shape (batch,fish,node,xyz)")
        if value.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError(f"{name} must use float32 or float64 coordinates")
        if not np.isfinite(value[:, :, 1]).all():
            raise ValueError(f"{name} pectoral coordinates must be finite")
    delta = first[:, None, :, None, 1] - second[None, :, None, :, 1]
    norms = np.sqrt((delta[..., None, :] @ delta[..., :, None])[..., 0, 0])
    squared = np.fromiter((value**2 for value in norms.flat), dtype=norms.dtype,
                          count=norms.size).reshape(norms.shape)
    # Python sum starts at integer zero, then accumulates in fish index order.
    costs = 0 + squared[:, :, 0, PERMUTATIONS[:, 0]]
    costs = costs + squared[:, :, 1, PERMUTATIONS[:, 1]]
    costs = costs + squared[:, :, 2, PERMUTATIONS[:, 2]]
    return costs


def different_identity_batch(first, second):
    """Return bool[A,B,3], identical to all pairwise _different_identity calls.

    Runtime tracking supplies finite float64 poses. Missing mouth/tail coordinates
    do not enter the comparison. Exactly coincident pectorals retain the frozen
    method's lexical tie result; this helper does not resolve that ambiguity.
    """
    costs = identity_costs_batch(first, second)
    chosen = PERMUTATIONS[np.argmin(costs, axis=-1)]
    return chosen != np.arange(3)
