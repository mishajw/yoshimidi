from typing import Literal, TypeAlias

import numpy as np
import torch
from jaxtyping import Float, UInt8

from yoshimidi.data.parse import time_parsing
from yoshimidi.data.parse.one_hot_parsing import get_one_hot_range, piece_range
from yoshimidi.data.parse.tracks import Channel, KeySignature, Note

TokenFields: TypeAlias = Literal["kind", "note_on", "note_off", "time", "key_signature"]
TOKEN_FIELDS: list[TokenFields] = [
    "kind",
    "note_on",
    "note_off",
    "time",
    "key_signature",
]

KindType: TypeAlias = Literal["on", "off", "pause", "end", "key_signature"]
KINDS: list[KindType] = ["on", "off", "pause", "end", "key_signature"]

TOKEN_DIM = len(TOKEN_FIELDS)
DTYPE = np.uint8

KEY_SIGNATURES = list(
    f"{note}{mode}"
    for note in [
        "A",
        "A#",
        "Ab",
        "B",
        "Bb",
        "C",
        "C#",
        "Cb",
        "D",
        "D#",
        "Db",
        "E",
        "Eb",
        "F",
        "F#",
        "G",
        "G#",
        "Gb",
    ]
    for mode in ["", "m"]
)

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
    index = 0
    for note in channel.notes:
        if isinstance(note, Note) and note.kind == "on":
            create_note_on_token(output[index], note=note.note)
            index += 1
        elif isinstance(note, Note) and note.kind == "off":
            create_note_off_token(output[index], note=note.note)
            index += 1
        elif isinstance(note, KeySignature):
            create_key_signature_token(output[index], key=note.key)
            index += 1
        else:
            raise ValueError(note)
        if isinstance(note, Note) and note.time_delta_secs > 0:
            create_pause_token(output[index], time_delta_secs=note.time_delta_secs)
            index += 1
    create_end_token(output[index])
    index += 1
    assert index == output.shape[0], (index, output.shape[0])


def from_one_hot(
    index: int,
    probs: Float[torch.Tensor, "token"],
) -> UInt8[np.ndarray, "token"]:  # noqa: F722
    one_hot_range = get_one_hot_range(index)
    start, end = piece_range(one_hot_range)
    relative_index = index - start
    result = np.zeros(TOKEN_DIM, dtype=DTYPE)
    if one_hot_range == "note_on":
        create_note_on_token(result, note=relative_index)
    elif one_hot_range == "note_off":
        create_note_off_token(result, note=relative_index)
    elif one_hot_range == "key_signature":
        create_key_signature_token(result, key=KEY_SIGNATURES[relative_index])
    elif one_hot_range == "end":
        create_end_token(result)
    elif one_hot_range == "pause":
        time_uint8 = time_parsing.time_uint8_from_support(probs[start:end])
        time_delta_secs = time_parsing.time_from_uint8(time_uint8)
        create_pause_token(result, time_delta_secs=time_delta_secs)
    else:
        raise ValueError(one_hot_range)
    return result


def get_buffer_size(channel: Channel) -> tuple[int, int]:
    num_notes = len(channel.notes)
    num_pauses = sum(
        1
        for note in channel.notes
        if isinstance(note, Note) and note.time_delta_secs > 0
    )
    num_ends = 1
    return (num_notes + num_pauses + num_ends, len(TOKEN_FIELDS))


def get_kind(token: UInt8[np.ndarray, "token"]) -> KindType:
    index = TOKEN_FIELDS.index("kind")
    kind_index = token[index]
    assert kind_index < len(KINDS), (kind_index, len(KINDS))
    return KINDS[token[index]]


def get_note_on(token: UInt8[np.ndarray, "token"]) -> int:
    assert get_kind(token) == "on"
    return token[TOKEN_FIELDS.index("note_on")]


def get_note_off(token: UInt8[np.ndarray, "token"]) -> int:
    assert get_kind(token) == "off"
    return token[TOKEN_FIELDS.index("note_off")]


def get_time_secs(token: UInt8[np.ndarray, "token"]) -> float:
    index = TOKEN_FIELDS.index("time")
    assert get_kind(token) == "pause"
    return time_parsing.time_from_uint8(token[index])


def get_key_signature(token: UInt8[np.ndarray, "token"]) -> str:
    index = TOKEN_FIELDS.index("key_signature")
    assert get_kind(token) == "key_signature"
    return KEY_SIGNATURES[token[index]]


def create_note_on_token(mmap: UInt8[np.ndarray, "token"], note: int) -> None:
    mmap[TOKEN_FIELDS.index("kind")] = KINDS.index("on")
    mmap[TOKEN_FIELDS.index("note_on")] = note


def create_note_off_token(mmap: UInt8[np.ndarray, "token"], note: int) -> None:
    mmap[TOKEN_FIELDS.index("kind")] = KINDS.index("off")
    mmap[TOKEN_FIELDS.index("note_off")] = note


def create_pause_token(
    mmap: UInt8[np.ndarray, "token"], time_delta_secs: float
) -> None:
    mmap[TOKEN_FIELDS.index("kind")] = KINDS.index("pause")
    mmap[TOKEN_FIELDS.index("time")] = time_parsing.time_to_uint8(time_delta_secs)


def create_end_token(mmap: UInt8[np.ndarray, "token"]) -> None:
    mmap[TOKEN_FIELDS.index("kind")] = KINDS.index("end")


def create_key_signature_token(mmap: UInt8[np.ndarray, "token"], key: str) -> None:
    key = key.strip()
    mmap[TOKEN_FIELDS.index("kind")] = KINDS.index("key_signature")
    assert key in KEY_SIGNATURES, repr(key)
    mmap[TOKEN_FIELDS.index("key_signature")] = KEY_SIGNATURES.index(key)
