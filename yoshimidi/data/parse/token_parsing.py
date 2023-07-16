from typing import Literal

import numpy as np
import torch
from jaxtyping import Float

from yoshimidi.data.parse.tracks import Channel
from yoshimidi.data.token_format import PIECE_LENGTHS, VOCAB

DTYPE = np.float32

# jaxtyping
seq, vocab = None, None


def from_channel(channel: Channel, include_end: bool = True) -> np.ndarray:
    buffer = np.zeros(get_buffer_size(channel), dtype=DTYPE)
    from_channel_to_buffer(channel, buffer)
    if not include_end:
        buffer = buffer[:-1]
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


def create_torch_token(
    kind: Literal["on", "off", "pause", "end"],
    *,
    note: int | None = None,
    time: float | None = None,
) -> torch.Tensor:
    result = np.zeros(VOCAB)
    _create_token(result, kind, note=note, time=time)
    return torch.Tensor(result)


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
    return np.argmax(token[4:16]).item() + 12 * np.argmax(token[16:27]).item()


def get_time_secs(token: Float[np.ndarray, "vocab"]) -> float:
    assert get_kind(token) == "pause"
    return token[27]


def _create_token(
    mmap: Float[np.ndarray, "vocab"],
    kind: Literal["on", "off", "pause", "end"],
    *,
    note: int | None = None,
    time: float | None = None,
) -> None:
    index = 0

    for piece, piece_length in PIECE_LENGTHS.items():
        if piece == "kind":
            assert piece_length == 4
            if kind == "on":
                mmap[index] = 1
            elif kind == "off":
                mmap[index + 1] = 1
            elif kind == "pause":
                mmap[index + 2] = 1
            elif kind == "end":
                mmap[index + 3] = 1

        elif piece == "note_key":
            assert piece_length == 12
            if note is not None:
                assert 0 <= note < 128, note
                mmap[index + note % 12] = 1

        elif piece == "note_octave":
            assert piece_length == 11
            if note is not None:
                mmap[index + note // 12] = 1

        elif piece == "time":
            assert piece_length == 1
            if time is not None:
                mmap[index] = time

        else:
            raise ValueError(piece)

        index += piece_length

    assert index == mmap.shape[0], (index, mmap.shape[0])
