"""Bounded16-frame/stride8 native iterator, with no model, query logic or decoder.

The caller supplies one ordered (native_index, payload) iterator. It owns decoded
SHA/mask/availability checks. This helper holds at most16 payload references and
reuses the same overlap objects without mutation. Consumers must release each
window before requesting the next to keep their own memory bounded. This is not
a bound on a source iterator's buffering or on official full-prefix model state.
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class Window:
    call_index: int
    input_relative: tuple[int, int]
    input_native: tuple[int, int]
    maximum_seen_native: int
    records: tuple


def _windows(source, first, count):
    ring = deque()
    consumed = 0
    for call_index, lo in enumerate(range(0, count - 15, 8)):
        if call_index:
            for _ in range(8):
                ring.popleft()
        hi = lo + 16
        while consumed < hi:
            try:
                record = next(source)
            except StopIteration as error:
                raise ValueError("native stream ended before fixed stop") from error
            require(
                isinstance(record, tuple)
                and len(record) == 2
                and type(record[0]) is int
                and record[0] == first + consumed,
                "native stream order/index differs",
            )
            ring.append(record)
            consumed += 1
            require(len(ring) <= 16, "internal window bound exceeded")
        require(len(ring) == 16, "full16 window required")
        window = Window(call_index, (lo, hi), (first + lo, first + hi), first + hi - 1, tuple(ring))
        yield window
        del window
    require(consumed == count, "fixed native count differs")
    # Do not pull an extra record merely to probe EOF. The source manifest owns
    # full source extent; this iterator consumes exactly the selected interval.


@contextmanager
def window_stream(records, *, first, count=200):
    """Context owns iterator close, including early break/error; no frame count201 read.

    Count16..200 divisible by8 only. Short-tail padding and arbitrary long runs
    require separate qualification and are rejected here.
    """
    require(
        type(first) is int
        and first >= 0
        and type(count) is int
        and 16 <= count <= 200
        and count % 8 == 0
        and first + count <= 2**63 - 1,
        "bounded native16/8 interval required",
    )
    source = iter(records)
    windows = _windows(source, first, count)
    try:
        yield windows
    finally:
        windows.close()
        close = getattr(source, "close", None)
        if close is not None:
            close()
