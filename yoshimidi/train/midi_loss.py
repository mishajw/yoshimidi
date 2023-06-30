from dataclasses import dataclass

import torch
from jaxtyping import Float


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
    outputs: Float[torch.Tensor, "batch seq vocab"],  # noqa: F722
) -> LossValues:
    labels = batch[:, 1:, :]
    outputs = outputs[:, :-1, :]
    index = 0

    print(outputs[0, 1, index : index + 4])
    kind_loss = torch.nn.functional.cross_entropy(
        labels[:, :, index : index + 4],
        outputs[:, :, index : index + 4],
    )
    index += 4

    note_key_loss = torch.nn.functional.cross_entropy(
        labels[:, :, index : index + 12],
        outputs[:, :, index : index + 12],
    )
    index += 12

    note_octave_loss = torch.nn.functional.cross_entropy(
        labels[:, :, index : index + 11],
        outputs[:, :, index : index + 11],
    )
    index += 11

    time_loss = torch.nn.functional.mse_loss(labels[:, :, index], outputs[:, :, index])
    index += 1

    assert index == batch.size(2)
    return LossValues(
        loss=kind_loss + note_key_loss + note_octave_loss + time_loss,
        kind_loss=kind_loss,
        note_key_loss=note_key_loss,
        note_octave_loss=note_octave_loss,
        time_loss=time_loss,
    )
