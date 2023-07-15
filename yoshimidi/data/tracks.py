from typing import Literal, Optional

import msgspec


class Track(msgspec.Struct):
    channels: dict[int, "Channel"]
    metadata: "TrackMetadata"


class TrackMetadata(msgspec.Struct):
    time_signature: Optional[str] = None
    key: Optional[str] = None


class Channel(msgspec.Struct):
    notes: list["Note"]
    # See https://en.wikipedia.org/wiki/General_MIDI#Program_change_events.
    program_nums: list[int]


class Note(msgspec.Struct):
    note: int
    kind: Literal["on", "off"]
    velocity: int
    time_delta_secs: float
