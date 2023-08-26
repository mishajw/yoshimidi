from typing import Literal, Optional

import msgspec


class TrackMetadata(msgspec.Struct):
    time_signature: Optional[str] = None
    key: Optional[str] = None


class Note(msgspec.Struct):
    note: int
    kind: Literal["on", "off"]
    velocity: int
    # Time to wait after the note is played.
    time_delta_secs: float


class KeySignature(msgspec.Struct):
    key: str


class Channel(msgspec.Struct):
    notes: list[Note | KeySignature]
    # See https://en.wikipedia.org/wiki/General_MIDI#Program_change_events.
    program_nums: list[int]


class Track(msgspec.Struct):
    channels: dict[int, Channel]
    metadata: "TrackMetadata"
