"""Monotonic per-stage timing with optional accelerator synchronization."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Callable, Iterator, List, Optional

from .types import StageTiming


class StageTimer:
    """Records stage durations with ``time.perf_counter``.

    When ``synchronize`` is given (e.g. ``torch.cuda.synchronize``), it is called before reading
    the clock at the start and at the end of a stage marked ``sync=True``, so asynchronous kernels
    are attributed to the stage that queued them. The synchronization calls themselves are inside
    the measured interval of the end point only.
    """

    def __init__(self, synchronize: Optional[Callable[[], None]] = None) -> None:
        self._sync = synchronize
        self.records: List[StageTiming] = []

    @contextmanager
    def stage(self, name: str, includes: str, sync: bool = False) -> Iterator[None]:
        do_sync = sync and self._sync is not None
        if do_sync:
            self._sync()  # type: ignore[misc]
        start = time.perf_counter()
        try:
            yield
        finally:
            if do_sync:
                self._sync()  # type: ignore[misc]
            self.records.append(StageTiming(name, time.perf_counter() - start, do_sync, includes))
