"""Measure the cost of replaying consecutive log writes sharing an archived timestamp."""

import itertools
import math
from time import perf_counter

from .ledger import Replay
from .stream import LogStream


def percentile(values, q):
    return sorted(values)[math.ceil(len(values) * q) - 1]


def benchmark(path, catalog):
    stream = LogStream()
    replay = Replay(catalog)
    samples = []
    peak_backlog_s = 0
    busy_until = 0
    last_clock = None
    day = 0
    started = perf_counter()
    with path.open("rb") as file:
        for stamp, lines in itertools.groupby(file, key=lambda line: line[2:18]):
            group = b"".join(lines)
            try:
                h, m, s = stamp.split(b":")
                clock = int(h) * 3600 + int(m) * 60 + float(s)
            except ValueError:
                raise ValueError("invalid archive timestamp") from None
            if last_clock is not None and clock < last_clock - 43200:
                day += 86400
            last_clock = clock
            arrival = day + clock
            before = perf_counter()
            for event in stream.feed(group):
                replay.apply(event)
            elapsed = perf_counter() - before
            delay = max(0, busy_until - arrival)
            peak_backlog_s = max(peak_backlog_s, delay)
            busy_until = arrival + delay + elapsed
            samples.append((elapsed * 1000, len(group)))
    wall = perf_counter() - started
    durations, bytes_per = zip(*samples)
    return dict(
        groups=len(samples),
        bytes=sum(bytes_per),
        p99_ms=percentile(durations, 0.99),
        max_ms=max(durations),
        max_burst_bytes=max(bytes_per),
        p99_burst_bytes=percentile(bytes_per, 0.99),
        max_fifo_wait_ms=peak_backlog_s * 1000,
        wall_s=wall,
        MiB_per_s=sum(bytes_per) / 2**20 / wall,
        grouping="adjacent lines with identical printed client timestamp; disk write boundaries unavailable",
        timing="includes decoding and state application; wall includes grouping and file read; no inference or HTML",
    )
