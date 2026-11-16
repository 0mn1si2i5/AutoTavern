from dataclasses import dataclass
import re

LINE = re.compile(
    rb"^D (\d\d):(\d\d):(\d\d\.\d+) (GameState|PowerTaskList|ChoiceCardMgr)\.(\w+)\(\) - (.*)$"
)


@dataclass(frozen=True, slots=True)
class Event:
    start: int
    end: int
    seconds: float
    stream: str
    method: str
    message: str


class LogStream:
    def __init__(self):
        self.tail = b""
        self.cursor = 0
        self.day = 0
        self.previous = None

    def feed(self, data, start_offset=None):
        if start_offset is not None and start_offset != self.cursor + len(self.tail):
            raise ValueError("non-contiguous input")
        lines = (self.tail + data).split(b"\n")
        self.tail = lines.pop()
        for line in lines:
            start = self.cursor
            self.cursor += len(line) + 1
            m = LINE.match(line.rstrip(b"\r"))
            if not m:
                continue
            seconds = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])
            if self.previous is not None and seconds < self.previous - 43200:
                self.day += 1
            self.previous = seconds
            yield Event(
                start,
                self.cursor,
                self.day * 86400 + seconds,
                m[4].decode(),
                m[5].decode(),
                m[6].decode("utf8", "replace"),
            )
