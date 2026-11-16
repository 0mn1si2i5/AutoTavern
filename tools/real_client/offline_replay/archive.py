"""Read only existing Power.log bytes, without querying a running game."""

from pathlib import Path

from .ledger import Replay
from .stream import LogStream


def prefixes(path: Path, offsets, catalog):
    """Yield independent snapshots at archived end-exclusive byte offsets."""
    stream = LogStream()
    replay = Replay(catalog)
    position = 0
    with path.open("rb") as source:
        for offset in sorted(set(offsets)):
            if not 0 <= offset <= path.stat().st_size:
                raise ValueError("prefix outside archive")
            while position < offset:
                data = source.read(min(65536, offset - position))
                if not data:
                    raise EOFError("archive truncated")
                for event in stream.feed(data, position):
                    replay.apply(event)
                position += len(data)
            yield offset, replay.snapshot()
