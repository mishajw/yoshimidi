from typing import Literal

import numpy as np
from jaxtyping import Float

from yoshimidi.data.tracks import Channel

VOCAB = 28  # 4 + 12 + 11 + 1
DTYPE = np.float32

# jaxtyping
seq, vocab = None, None


def from_channel(channel: Channel) -> np.ndarray:
    buffer = np.zeros(get_buffer_size(channel), dtype=DTYPE)
    from_channel_to_buffer(channel, buffer)
    return buffer


def from_channel_to_buffer(
    channel: Channel,
    output: Float[np.ndarray, "seq vocab"],  # noqa: F722
):
    index = 0
    for note_index in range(len(channel.notes)):
        note = channel.notes[note_index]
        _create_token(output[index], kind=note.kind, note=note.note)
        index += 1
        if note_index < len(channel.notes) - 1:
            _create_token(output[index], kind="pause", time=note.time_delta_secs)
            index += 1
        else:
            _create_token(output[index], kind="end")
            index += 1
    assert index == output.shape[0]


def get_buffer_size(channel: Channel):
    return (len(channel.notes) * 2, VOCAB)


def get_kind(token: Float[np.ndarray, "vocab"]) -> Literal["on", "off", "pause", "end"]:
    if token[0] == 1:
        return "on"
    elif token[1] == 1:
        return "off"
    elif token[2] == 1:
        return "pause"
    elif token[3] == 1:
        return "end"
    else:
        raise ValueError(token)


def get_note(token: Float[np.ndarray, "vocab"]) -> int:
    assert get_kind(token) in ["on", "off"]
    return np.argmax(token[4:16]) + 12 * np.argmax(token[16:27])


def get_time_secs(token: Float[np.ndarray, "vocab"]) -> float:
    assert get_kind(token) == "pause"
    return token[27]


def _create_token(
    mmap: Float[np.memmap, "vocab"],
    kind: Literal["on", "off", "pause", "end"],
    *,
    note: int | None = None,
    time: float | None = None,
) -> None:
    index = 0

    if kind == "on":
        mmap[0] = 1
    elif kind == "off":
        mmap[1] = 1
    elif kind == "pause":
        mmap[2] = 1
    elif kind == "end":
        mmap[3] = 1
    index += 4

    if note is not None:
        assert 0 <= note < 128, note
        mmap[index + note % 12] = 1
    index += 12
    if note is not None:
        mmap[index + note // 12] = 1
    index += 11

    if time is not None:
        mmap[index] = time
    index += 1

    assert index == mmap.shape[0], (index, mmap.shape[0])
