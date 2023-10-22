from typing import Literal, Optional, cast

import numpy as np
import torch
from jaxtyping import Float, UInt8

from yoshimidi.data.parse import time_parsing, token_parsing
from yoshimidi.data.parse.token_parsing import (
    KINDS,
    TOKEN_FIELDS,
    KindType,
    TokenFields,
)

OneHotRange = Literal["pause", "note_on", "note_off", "end", "key_signature"]
ONE_HOT_RANGE_LENGTHS: dict[OneHotRange, int] = {
    # Put pause first so we can one-hot the rest of the fields in one go.
    "pause": time_parsing.NUM_TIME_SUPPORTS,
    "note_on": 2**7,
    "note_off": 2**7,
    "end": 1,
    "key_signature": len(token_parsing.KEY_SIGNATURES),
}
VOCAB = sum(ONE_HOT_RANGE_LENGTHS.values())

# jaxtyping
seq, token, vocab = None, None, None


def from_tokens(
    input: UInt8[np.ndarray, "seq token"],  # noqa: F722
    device: torch.device,
    dtype: torch.dtype,
) -> Float[torch.Tensor, "seq vocab"]:  # noqa: F722
    input_tensor = torch.tensor(input, device=device, dtype=torch.int64)
    output = torch.zeros((input.shape[0], VOCAB), device=device, dtype=dtype)
    pause_start, pause_end = piece_range("pause")
    assert pause_start == 0, pause_start
    for seq_index in range(input_tensor.shape[0]):
        if input_tensor[seq_index, TOKEN_FIELDS.index("kind")] != KINDS.index("pause"):
            continue
        time_parsing.time_uint8_to_support(
            cast(int, input_tensor[seq_index, TOKEN_FIELDS.index("time")].item()),
            output[seq_index, pause_start:pause_end],
        )
    output[:, ONE_HOT_RANGE_LENGTHS["pause"] :] = _fill_non_pause_fields(
        input_tensor, device
    )
    return output


def _fill_non_pause_fields(
    input: UInt8[torch.Tensor, "seq token"],  # noqa: F722
    device: torch.device,
) -> torch.Tensor:
    non_pause_tokens = torch.zeros((input.shape[0]), device=device, dtype=torch.int64)
    indices_used = 0

    mappings: list[tuple[KindType, OneHotRange, Optional[TokenFields]]] = [
        ("on", "note_on", "note_on"),
        ("off", "note_off", "note_off"),
        ("end", "end", None),
        ("key_signature", "key_signature", "key_signature"),
    ]
    for kind, one_hot_range, token_field in mappings:
        mask = input[:, TOKEN_FIELDS.index("kind")] == KINDS.index(kind)
        if token_field is not None:
            non_pause_tokens[mask] = (
                input[mask, TOKEN_FIELDS.index(token_field)] + indices_used
            )
        else:
            non_pause_tokens[mask] = indices_used
        indices_used += ONE_HOT_RANGE_LENGTHS[one_hot_range]
    assert indices_used == VOCAB - ONE_HOT_RANGE_LENGTHS["pause"], indices_used
    non_pause_tokens = torch.nn.functional.one_hot(
        non_pause_tokens,
        num_classes=indices_used,
    )
    pause_mask: torch.Tensor = input[:, TOKEN_FIELDS.index("kind")] == KINDS.index(
        "pause"
    )
    non_pause_tokens[pause_mask] = 0
    return non_pause_tokens


def piece_range(one_hot_range: OneHotRange) -> tuple[int, int]:
    index = 0
    for ohr, length in ONE_HOT_RANGE_LENGTHS.items():
        if ohr == one_hot_range:
            return index, index + length
        index += length
    raise ValueError(one_hot_range)
