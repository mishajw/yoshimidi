from typing import Literal, Tuple, TypeAlias

import numpy as np
import torch
from jaxtyping import Float, UInt8

from yoshimidi.data.parse import time_parsing
from yoshimidi.data.parse.token_parsing import KINDS

PieceType: TypeAlias = Literal["kind", "note_key", "note_octave", "time"]
PIECE_LENGTHS: dict[PieceType, int] = {
    "kind": len(KINDS),
    "note_key": 12,
    "note_octave": 11,
    "time": time_parsing.NUM_TIME_SUPPORTS,
}
VOCAB = sum(PIECE_LENGTHS.values())  # 38

# jaxtyping
seq, token, vocab = None, None, None


def from_tokens(
    input: UInt8[np.ndarray, "seq token"],  # noqa: F722
) -> Float[torch.Tensor, "seq vocab"]:  # noqa: F722
    output = torch.zeros((input.shape[0], VOCAB))
    index = 0
    for piece, piece_length in PIECE_LENGTHS.items():
        if piece == "kind":
            output[index : index + piece_length] = torch.nn.functional.one_hot(
                input[0],
                num_classes=VOCAB,
                out=len(KINDS),
            )
        elif piece == "note_key":
            output[index : index + piece_length] = torch.nn.functional.one_hot(
                input[1] % 12,
                num_classes=VOCAB,
                out=len(KINDS),
            )
        elif piece == "note_octave":
            output[index : index + piece_length] = torch.nn.functional.one_hot(
                input[1] // 12,
                num_classes=VOCAB,
                out=len(KINDS),
            )
        elif piece == "time":
            for seq_index in range(input.shape[0]):
                time_parsing.time_uint8_to_support(
                    input[seq_index, 2],
                    output[seq_index, index : index + piece_length],
                )
        index += piece_length
    assert index == output.shape[0], (index, output.shape[0])
    return output


def piece_range(piece_type: PieceType) -> Tuple[int, int]:
    index = 0
    for piece, length in PIECE_LENGTHS.items():
        if piece == piece_type:
            return index, index + length
        index += length
    raise ValueError(piece)
