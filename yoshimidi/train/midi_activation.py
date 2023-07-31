import torch
from jaxtyping import Float
from torch import Tensor

from yoshimidi.data.parse.one_hot_parsing import PIECE_LENGTHS


def midi_activation(
    logits: Float[Tensor, "batch seq vocab"]  # noqa: F722
) -> Float[Tensor, "batch seq vocab"]:  # noqa: F722
    results = []
    index = 0
    for piece_length in PIECE_LENGTHS.values():
        results.append(
            torch.nn.functional.softmax(
                logits[:, :, index : index + piece_length], dim=2
            )
        )
        index += piece_length
    assert index == logits.size(2)
    result = torch.cat(results, dim=2)
    assert result.shape == logits.shape, (result.shape, logits.shape)
    return result
