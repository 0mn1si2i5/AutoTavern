"""Incremental read-only log sources; replay stops *before* each recorded send."""

import hashlib
from pathlib import Path
import subprocess
import time

from tools.real_client.offline_replay.ledger import Replay
from tools.real_client.offline_replay.stream import LogStream


class LogSource:
    def __init__(self, catalog, path=None, directory=None):
        self.catalog = catalog
        self.explicit_path = Path(path).resolve() if path else None
        self.directory = Path(directory or "/Applications/Hearthstone/Logs")
        self.path = None
        self.identity = None
        self.offset = 0
        self.changed = time.monotonic()
        self.replay = Replay(catalog)
        self.stream = LogStream()
        self.generation = 0
        self.error = None
        self.available = False
        self.unavailable_reason = "尚未发现 Power.log"

    def locate(self):
        if self.explicit_path:
            return self.explicit_path if self.explicit_path.exists() else None
        paths = list(self.directory.glob("Hearthstone_*/Power.log"))
        return max(paths, key=lambda p: p.stat().st_mtime_ns) if paths else None

    def poll(self):
        if self.explicit_path is None and subprocess.run(
                ["pgrep", "-x", "Hearthstone"], capture_output=True, check=False).returncode != 0:
            path = None
            self.unavailable_reason = "炉石客户端未运行；打开客户端后自动连接只读日志"
        else:
            path = self.locate()
            self.unavailable_reason = "找不到 Power.log；等待客户端写入日志"
        if path is None:
            changed = self.available
            if self.available:
                self.replay, self.stream = Replay(self.catalog), LogStream()
                self.path, self.identity, self.offset = None, None, 0
                self.generation += 1
            self.available = False
            return changed
        self.available = True
        stat = path.stat()
        identity = (str(path), stat.st_dev, stat.st_ino)
        if identity != self.identity or stat.st_size < self.offset:
            self.path, self.identity = path, identity
            self.offset = 0
            self.replay, self.stream = Replay(self.catalog), LogStream()
            self.generation += 1
        changed = stat.st_size > self.offset
        if changed:
            # Snapshot the observed size; appended bytes are consumed next time.
            with path.open("rb") as handle:
                handle.seek(self.offset)
                while self.offset < stat.st_size:
                    data = handle.read(min(65536, stat.st_size - self.offset))
                    if not data:
                        break
                    for event in self.stream.feed(data):
                        self.replay.apply(event)
                    self.offset += len(data)
            self.changed = time.monotonic()
        # A hotkey may arrive at an option-list EOF with no subsequent event.
        # A full newline plus stable size is a logged heuristic, never a UI send.
        tree = self.replay.tree
        if tree and not tree.complete and not self.stream.tail and time.monotonic() - self.changed >= .3:
            tree.complete = True
            changed = True
        return changed

    def info(self):
        return {"path": str(self.path) if self.path else None, "offset": self.offset,
                "generation": self.generation, "partial_line_bytes": len(self.stream.tail),
                "available": self.available, "reason": self.unavailable_reason if not self.available else None,
                "idle_seconds": round(time.monotonic() - self.changed, 1),
                "completion_policy": "non-option event or newline-terminated options stable for 300ms"}


class ReplaySource:
    def __init__(self, catalog, path):
        self.catalog, self.path = catalog, Path(path).resolve()
        self._digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.index = -1
        self.total = 0
        self.boundaries = []
        self.boundary_times = []
        stream = LogStream()
        with self.path.open("rb") as handle:
            while data := handle.read(65536):
                for event in stream.feed(data):
                    if self._boundary(event):
                        self.boundaries.append(event.start)
                        self.boundary_times.append(event.seconds)
        self.total = len(self.boundaries)
        self._reset()

    @staticmethod
    def _boundary(event):
        return event.stream == "GameState" and (
            event.method == "SendOption" or event.method == "SendChoices" and event.message.startswith("id="))

    def _reset(self):
        if getattr(self, "handle", None):
            self.handle.close()
        self.handle = self.path.open("rb")
        self.replay, self.stream = Replay(self.catalog), LogStream()
        self.offset, self.index = 0, -1
        self.last_options_time = None

    def goto(self, index):
        if not 0 <= index < self.total:
            return False
        if index < self.index:
            self._reset()
        limit = self.boundaries[index]
        while self.offset < limit:
            data = self.handle.read(min(65536, limit - self.offset))
            if not data:
                raise ValueError("Replay source truncated during playback")
            for event in self.stream.feed(data):
                self.replay.apply(event)
                if event.stream == "GameState" and event.method == "DebugPrintOptions":
                    self.last_options_time = event.seconds
            self.offset += len(data)
        # Match live's quiet-window rule using only the replay clock and prefix.
        if (self.replay.tree and not self.stream.tail and self.last_options_time is not None
                and self.boundary_times[index] - self.last_options_time >= .3):
            self.replay.tree.complete = True
        self.index = index
        return True

    def poll(self):
        return False

    def info(self):
        return {"path": str(self.path), "offset": self.offset, "index": self.index,
                "total": self.total, "sha256": self._digest, "generation": 0,
                "prefix_only": True}

    def close(self):
        self.handle.close()
