from typing import Tuple, cast

import numpy as np
import torch
from jaxtyping import Bool, Float, UInt8

from yoshimidi.data.parse import time_parsing, token_parsing
from yoshimidi.data.parse.token_parsing import (
    KINDS,
    TOKEN_FIELDS,
    KindType,
    TokenFields,
)

TOKEN_FIELD_LENGTHS: dict[TokenFields, int] = {
    "kind": len(KINDS),
    "note_on": 2**7 + 1,
    "note_off": 2**7 + 1,
    "time": time_parsing.NUM_TIME_SUPPORTS + 1,
    "key_signature": len(token_parsing.KEY_SIGNATURES) + 1,
}
VOCAB = sum(TOKEN_FIELD_LENGTHS.values())

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
        _get_classes_with_none(input_tensor, "on", "note_on"),
        num_classes=TOKEN_FIELD_LENGTHS["note_on"],
    )

    start, end = piece_range("note_off")
    output[:, start:end] = torch.nn.functional.one_hot(
        _get_classes_with_none(input_tensor, "off", "note_off"),
        num_classes=TOKEN_FIELD_LENGTHS["note_off"],
    )

    start, end = piece_range("time")
    for seq_index in range(input_tensor.shape[0]):
        if input_tensor[seq_index, TOKEN_FIELDS.index("kind")] == KINDS.index("pause"):
            # If we're not in a pause field, we "one hot" by setting the first bit.
            # Otherwise, we bias predictions towards zero.
            output[seq_index, start] = 1
            continue
        time_parsing.time_uint8_to_support(
            cast(int, input_tensor[seq_index, TOKEN_FIELDS.index("time")].item()),
            output[seq_index, start + 1 : end],
        )

    start, end = piece_range("key_signature")
    output[:, start:end] = torch.nn.functional.one_hot(
        _get_classes_with_none(input_tensor, "key_signature", "key_signature"),
        num_classes=TOKEN_FIELD_LENGTHS["key_signature"],
    )
    return output


def _get_classes_with_none(
    input_tensor: UInt8[torch.Tensor, "seq token"],  # noqa: F722
    kind: KindType,
    token_field: TokenFields,
) -> torch.Tensor:
    """
    Gets the classes for a given kind.

    Used for classes that don't always exist: for example, we don't have a note-on field
    for a pause. Therefore, when the kind is not occupied, we return a class of zero.
    """
    is_kind: Bool[torch.Tensor, "seq"] = input_tensor[
        :, TOKEN_FIELDS.index("kind")
    ] == KINDS.index(kind)
    kind_values = input_tensor[:, TOKEN_FIELDS.index(token_field)]
    return is_kind.to(dtype=torch.uint8) + kind_values


def piece_range(token_field: TokenFields) -> Tuple[int, int]:
    index = 0
    for tf, length in TOKEN_FIELD_LENGTHS.items():
        if tf == token_field:
            return index, index + length
        index += length
    raise ValueError(token_field)
