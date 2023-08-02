from typing import Literal, TypeAlias

import numpy as np
from jaxtyping import UInt8

from yoshimidi.data.parse import time_parsing
from yoshimidi.data.parse.tracks import Channel

KindType: TypeAlias = Literal["on", "off", "end"]
KINDS: list[KindType] = ["on", "off", "end"]

TokenFields: TypeAlias = Literal["kind", "note", "time"]
_TOKEN_FIELDS: list[TokenFields] = ["kind", "note", "time"]

TOKEN_DIM = len(_TOKEN_FIELDS)
DTYPE = np.uint8

# jaxtyping
seq, token = None, None


def from_channel(
    channel: Channel, include_end: bool = True
) -> UInt8[np.ndarray, "seq token"]:  # noqa: F722
    buffer = np.zeros(get_buffer_size(channel), dtype=DTYPE)
    from_channel_to_buffer(channel, buffer)
    if not include_end:
        buffer = buffer[:-1]
    return buffer


def from_channel_to_buffer(
    channel: Channel,
    output: UInt8[np.ndarray, "seq token"],  # noqa: F722
) -> None:
    for index, note in enumerate(channel.notes):
        _create_token_in_buffer(
            output[index],
            kind=note.kind,
            note=note.note,
            time_delta_secs=note.time_delta_secs,
        )
    _create_token_in_buffer(output[-1], kind="end", note=0, time_delta_secs=0)


def get_buffer_size(channel: Channel) -> tuple[int, int]:
    return (len(channel.notes) + 1, len(_TOKEN_FIELDS))


def get_kind(token: UInt8[np.ndarray, "token"]) -> KindType:
    index = _TOKEN_FIELDS.index("kind")
    if token[index] == 0:
        return "on"
    elif token[index] == 1:
        return "off"
    elif token[index] == 2:
        return "end"
    else:
        raise ValueError(token)


def get_note(token: UInt8[np.ndarray, "token"]) -> int:
    index = _TOKEN_FIELDS.index("note")
    assert get_kind(token) in ["on", "off"]
    return token[index]


def get_time_secs(token: UInt8[np.ndarray, "token"]) -> float:
    index = _TOKEN_FIELDS.index("time")
    assert get_kind(token) in ["on", "off"]
    return time_parsing.time_from_uint8(token[index])


def create_token(
    kind: KindType,
    *,
    note: int,
    time_delta_secs: float,
) -> UInt8[np.ndarray, "token"]:
    buffer = np.zeros(TOKEN_DIM)
    _create_token_in_buffer(buffer, kind, note=note, time_delta_secs=time_delta_secs)
    return buffer


def _create_token_in_buffer(
    mmap: UInt8[np.ndarray, "token"],
    kind: KindType,
    *,
    note: int,
    time_delta_secs: float,
) -> None:
    index = 0

    if kind == "on":
        mmap[index] = 0
    elif kind == "off":
        mmap[index] = 1
    elif kind == "end":
        mmap[index] = 2
    else:
        raise ValueError(kind)
    index += 1

    assert 0 <= note < 2**32, note
    mmap[index] = note
    index += 1

    mmap[index] = time_parsing.time_to_uint8(time_delta_secs)
    index += 1

    assert index == mmap.shape[0], (index, mmap.shape[0])
