from dataclasses import dataclass

import torch
from jaxtyping import Float

from yoshimidi.data.token_format import PIECE_LENGTHS


@dataclass
class LossValues:
    loss: torch.Tensor
    kind_loss: torch.Tensor
    note_key_loss: torch.Tensor
    note_octave_loss: torch.Tensor
    time_loss: torch.Tensor


def autoregressive_midi_loss(
    *,
    batch: Float[torch.Tensor, "batch seq vocab"],  # noqa: F722
    logits: Float[torch.Tensor, "batch seq vocab"],  # noqa: F722
) -> LossValues:
    labels = torch.flatten(batch[:, 1:, :], start_dim=0, end_dim=1)
    logits = torch.flatten(logits[:, :-1, :], start_dim=0, end_dim=1)

    index = 0
    kind_loss, note_key_loss, note_octave_loss, time_loss = None, None, None, None

    for piece, piece_length in PIECE_LENGTHS.items():
        if piece == "kind":
            assert piece_length == 4
            kind_loss = torch.nn.functional.cross_entropy(
                logits[:, index : index + 4],
                labels[:, index : index + 4],
            )

        elif piece == "note_key":
            assert piece_length == 12
            note_key_loss = torch.nn.functional.cross_entropy(
                logits[:, index : index + 12],
                labels[:, index : index + 12],
            )

        elif piece == "note_octave":
            assert piece_length == 11
            note_octave_loss = torch.nn.functional.cross_entropy(
                logits[:, index : index + 11],
                labels[:, index : index + 11],
            )

        elif piece == "time":
            assert piece_length == 1
            time_loss = torch.nn.functional.mse_loss(
                logits[:, index],
                labels[:, index],
            )

        index += piece_length

    assert index == batch.size(2)
    assert kind_loss is not None
    assert note_key_loss is not None
    assert note_octave_loss is not None
    assert time_loss is not None

    return LossValues(
        loss=kind_loss + note_key_loss + note_octave_loss + time_loss,
        kind_loss=kind_loss,
        note_key_loss=note_key_loss,
        note_octave_loss=note_octave_loss,
        time_loss=time_loss,
    )
