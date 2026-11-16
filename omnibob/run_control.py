"""Live control of a training process through a small JSON file.

Modes: ``normal`` runs at full speed; ``low`` uses fewer threads and works only part of the
time (``low_duty``); ``pause`` saves a checkpoint and waits; ``stop`` saves and exits. The file
is polled every few seconds. A missing, half-written or invalid file keeps the last valid
settings, so an edit in progress cannot stop or speed up a run by accident.
"""
import json
import os
from pathlib import Path
import time

MODES = ("normal", "low", "pause", "stop")
DEFAULTS = {"mode": "normal", "threads": None, "low_threads": 1, "low_duty": 0.35}


def validate(settings):
    """Settings merged over the defaults; ValueError for unknown fields or bad values."""
    if not isinstance(settings, dict):
        raise ValueError("control file must hold a JSON object")
    unknown = sorted(set(settings)-set(DEFAULTS))
    if unknown:
        raise ValueError(f"unknown control fields: {unknown}")
    merged = {**DEFAULTS, **settings}
    if merged["mode"] not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    for key in ("threads", "low_threads"):
        value = merged[key]
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ValueError(f"{key} must be a positive integer")
    duty = merged["low_duty"]
    if isinstance(duty, bool) or not isinstance(duty, (int, float)) or not .05 <= duty <= 1:
        raise ValueError("low_duty must be between 0.05 and 1")
    return merged


def replace(source, destination, attempts=20, delay=.25):
    """os.replace, retried briefly on Windows, where a virus scanner or a reader holding the
    target makes the rename fail with PermissionError for a moment."""
    for attempt in range(attempts):
        try:
            return os.replace(source, destination)
        except PermissionError:
            if os.name != "nt" or attempt == attempts-1:
                raise
            time.sleep(delay)


def read(path):
    """Validated settings from the file, or the defaults when it does not exist."""
    path = Path(path)
    if not path.exists():
        return dict(DEFAULTS)
    return validate(json.loads(path.read_text(encoding="utf-8")))


def write(path, settings):
    """Validate and atomically replace the control file; returns the stored settings."""
    path = Path(path)
    merged = validate(settings)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps({k: merged[k] for k in DEFAULTS}, indent=2)+"\n", encoding="utf-8")
    replace(temporary, path)
    return merged


def _torch_threads(count):
    import torch
    torch.set_num_threads(count)


class RunControl:
    """Polls the control file, applies thread counts and sleeps for the low-power duty cycle.

    ``threads`` is the run's own thread count, used in normal mode unless the file sets one.
    Sleeping never changes what is computed; a different thread count can change floating
    point rounding, so runs that switch modes are not bit-identical to uninterrupted ones.
    """

    def __init__(self, path, threads, poll_seconds=2.0, quantum_seconds=.5,
                 clock=time.monotonic, sleep=time.sleep, set_threads=_torch_threads):
        self.path = Path(path)
        self.base_threads = threads
        self.poll_seconds, self.quantum_seconds = poll_seconds, quantum_seconds
        self.clock, self.sleep, self.set_threads = clock, sleep, set_threads
        self.settings = dict(DEFAULTS)
        self.error = None
        self.active_threads = None
        self.throttled_seconds = 0.0
        self.paused_seconds = 0.0
        self._signature = None
        self._polled = None
        self._busy_since = clock()
        self.refresh(force=True)

    @property
    def mode(self):
        return self.settings["mode"]

    def refresh(self, force=False):
        """Re-read the file when it changed (at most once per poll interval); returns the mode."""
        now = self.clock()
        if not force and self._polled is not None and now-self._polled < self.poll_seconds:
            return self.mode
        self._polled = now
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            stat = None
        if stat is not None and (stat.st_mtime_ns, stat.st_size) != self._signature:
            try:
                self.settings = validate(json.loads(self.path.read_text(encoding="utf-8")))
                self._signature = (stat.st_mtime_ns, stat.st_size)
                self.error = None
            except (OSError, ValueError) as exc:
                # Keep the last valid settings; the next poll reads the file again.
                self.error = f"{type(exc).__name__}: {exc}"
        self._apply_threads()
        return self.mode

    def _apply_threads(self):
        wanted = (self.settings["low_threads"] if self.mode == "low"
                  else self.settings["threads"] or self.base_threads)
        if wanted != self.active_threads:
            self.set_threads(wanted)
            self.active_threads = wanted

    def throttle(self):
        """In low mode, sleep so that work takes about ``low_duty`` of the wall time."""
        now = self.clock()
        if self.mode != "low":
            self._busy_since = now
            return 0.0
        busy = now-self._busy_since
        if busy < self.quantum_seconds:
            return 0.0
        duty = self.settings["low_duty"]
        pause = busy*(1-duty)/duty
        if pause > 0:
            self.sleep(pause)
            self.throttled_seconds += pause
        self._busy_since = self.clock()
        return pause

    def wait_while_paused(self, on_pause, check):
        """While the mode is pause: call ``on_pause`` once (to save), then ``check`` every poll
        (it raises to end the run). Returns whether the run was paused."""
        if self.mode != "pause":
            return False
        on_pause()
        started = self.clock()
        while self.refresh(force=True) == "pause":
            check()
            self.sleep(self.poll_seconds)
        self.paused_seconds += self.clock()-started
        self._busy_since = self.clock()
        return True
