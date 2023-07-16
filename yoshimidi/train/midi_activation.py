import torch
from jaxtyping import Float
from torch import Tensor

from yoshimidi.data.token_format import PIECE_LENGTHS


def midi_activation(
    logits: Float[Tensor, "batch seq vocab"]  # noqa: F722
) -> Float[Tensor, "batch seq vocab"]:  # noqa: F722
    results = []
    index = 0
    for piece, piece_length in PIECE_LENGTHS.items():
        if piece in ["kind", "note_key", "note_octave"]:
            results.append(
                torch.nn.functional.softmax(
                    logits[:, :, index : index + piece_length], dim=2
                )
            )
        elif piece == "time":
            assert piece_length == 1
            results.append(logits[:, :, index : index + piece_length])
        else:
            raise ValueError(piece)
        index += piece_length

    assert index == logits.size(2)
    result = torch.cat(results, dim=2)
    assert result.shape == logits.shape, (result.shape, logits.shape)
    return result
