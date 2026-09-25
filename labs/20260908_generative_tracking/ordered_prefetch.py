"""Bounded, ordered preprocessing in one producer with explicit error propagation."""

from __future__ import annotations

from contextlib import contextmanager
from queue import Empty, Full, Queue
from threading import Event, Thread


@contextmanager
def ordered_prefetch(iterable, capacity=2):
    """Yield one consumer iterator; always join the sole producer before returning.

    The producer exclusively advances the source iterator. It must not depend on
    consumer work to finish a single item. Capacity bounds queued frames, with one
    additional item being prepared; none are dropped. Errors reach the consumer.
    """
    if capacity < 1:
        raise ValueError("prefetch capacity must be positive")
    queue, stopped = Queue(maxsize=capacity), Event()

    def put(kind, value):
        while not stopped.is_set():
            try:
                queue.put((kind, value), timeout=.05)
                return True
            except Full:
                pass
        return False

    def produce():
        source = iter(iterable)
        try:
            while not stopped.is_set():
                try:
                    value = next(source)
                except StopIteration:
                    put("end", None)
                    return
                if not put("item", value):
                    return
        except BaseException as error:
            put("error", error)
        finally:
            close = getattr(source, "close", None)
            if close:
                close()

    producer = Thread(target=produce, name="ordered-image-prefetch", daemon=False)

    def consume():
        while True:
            try:
                kind, value = queue.get(timeout=.1)
            except Empty:
                if not producer.is_alive():
                    raise RuntimeError(
                        "prefetch producer exited without a terminal message") from None
                continue
            if kind == "error":
                raise value
            if kind == "end":
                return
            yield value

    producer.start()
    try:
        yield consume()
    finally:
        stopped.set()
        # Join before callers close camera handles, CUDA builders or temporary inputs.
        producer.join()
