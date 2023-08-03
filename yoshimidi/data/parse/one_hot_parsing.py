from typing import Tuple, cast

import numpy as np
import torch
from jaxtyping import Float, UInt8

from yoshimidi.data.parse import time_parsing
from yoshimidi.data.parse.token_parsing import KINDS, TOKEN_FIELDS, TokenFields

TOKEN_FIELD_LENGTHS: dict[TokenFields, int] = {
    "kind": len(KINDS),
    "note_on": 2**7,
    "note_off": 2**7,
    "time": time_parsing.NUM_TIME_SUPPORTS,
}
VOCAB = sum(TOKEN_FIELD_LENGTHS.values())  # 38

# jaxtyping
seq, token, vocab = None, None, None


def from_tokens(
    input: UInt8[np.ndarray, "seq token"],  # noqa: F722
    device: torch.device,
    dtype: torch.dtype,
) -> Float[torch.Tensor, "seq vocab"]:  # noqa: F722
    input_tensor = torch.tensor(input, device=device, dtype=torch.int64)
    output = torch.zeros((input.shape[0], VOCAB), device=device, dtype=dtype)
    start, end = piece_range("kind")
    output[:, start:end] = torch.nn.functional.one_hot(
        input_tensor[:, TOKEN_FIELDS.index("kind")],
        num_classes=TOKEN_FIELD_LENGTHS["kind"],
    )
    start, end = piece_range("note_on")
    output[:, start:end] = torch.nn.functional.one_hot(
        input_tensor[:, TOKEN_FIELDS.index("note_on")],
        num_classes=TOKEN_FIELD_LENGTHS["note_on"],
    )
    start, end = piece_range("note_off")
    output[:, start:end] = torch.nn.functional.one_hot(
        input_tensor[:, TOKEN_FIELDS.index("note_off")],
        num_classes=TOKEN_FIELD_LENGTHS["note_off"],
    )
    for seq_index in range(input_tensor.shape[0]):
        time_parsing.time_uint8_to_support(
            cast(int, input_tensor[seq_index, TOKEN_FIELDS.index("time")].item()),
            output[seq_index, start:end],
        )
    return output


def piece_range(token_field: TokenFields) -> Tuple[int, int]:
    index = 0
    for tf, length in TOKEN_FIELD_LENGTHS.items():
        if tf == token_field:
            return index, index + length
        index += length
    raise ValueError(token_field)
