from typing import Literal, Tuple, TypeAlias

import numpy as np

PieceType: TypeAlias = Literal["kind", "note_key", "note_octave", "time"]
KindType: TypeAlias = Literal["on", "off", "pause", "end"]

_TIME_SUPPORTS = [0, *[2**exponent for exponent in range(-8, 1)], 300]

KINDS: list[KindType] = ["on", "off", "pause", "end"]
PIECE_LENGTHS: dict[PieceType, int] = {
    "kind": 4,
    "note_key": 12,
    "note_octave": 11,
    "time": len(_TIME_SUPPORTS),
}
VOCAB = sum(PIECE_LENGTHS.values())  # 35


def piece_range(piece_type: PieceType) -> Tuple[int, int]:
    index = 0
    for piece, length in PIECE_LENGTHS.items():
        if piece == piece_type:
            return index, index + length
        index += length
    raise ValueError(piece)


def time_to_support(time: float, buffer: np.ndarray) -> None:
    assert time >= 0
    assert buffer.shape == (len(_TIME_SUPPORTS),)
    if time == 0:
        buffer[0] = 1
        return
    if time >= _TIME_SUPPORTS[-1]:
        buffer[-1] = 1
        return
    upper_idx = next(i for i, support in enumerate(_TIME_SUPPORTS) if support > time)
    lower_idx = upper_idx - 1
    lower_support = _TIME_SUPPORTS[lower_idx]
    upper_support = _TIME_SUPPORTS[upper_idx]
    assert lower_support <= time < upper_support, (lower_support, time, upper_support)
    weighting = (time - lower_support) / (upper_support - lower_support)
    buffer[lower_idx] = 1 - weighting
    buffer[upper_idx] = weighting


def support_to_time(support: np.ndarray) -> float:
    assert support.shape == (len(_TIME_SUPPORTS),)
    return np.dot(support, _TIME_SUPPORTS)
