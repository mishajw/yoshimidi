from typing import Dict, List, Literal, Optional

import msgspec


class Track(msgspec.Struct):
    channels: Dict[int, "Channel"]
    metadata: "TrackMetadata"


class TrackMetadata(msgspec.Struct):
    bpm: Optional[float] = None
    time_signature: Optional[str] = None
    key: Optional[str] = None


class Channel(msgspec.Struct):
    notes: List["Note"]
    # See https://en.wikipedia.org/wiki/General_MIDI#Program_change_events.
    program_nums: List[int]


class Note(msgspec.Struct):
    note: int
    kind: Literal["on", "off"]
    velocity: int
    time_secs: float
