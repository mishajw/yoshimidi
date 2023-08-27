from dataclasses import dataclass

import torch
from jaxtyping import Float

from yoshimidi.data.parse import one_hot_parsing
from yoshimidi.data.parse.token_parsing import TOKEN_FIELDS


@dataclass
class LossValues:
    loss: torch.Tensor
    kind_loss: torch.Tensor
    note_on_loss: torch.Tensor
    note_off_loss: torch.Tensor
    time_loss: torch.Tensor
    key_signature_loss: torch.Tensor


def autoregressive_midi_loss(
    *,
    batch: Float[torch.Tensor, "batch seq vocab"],  # noqa: F722
    logits: Float[torch.Tensor, "batch seq vocab"],  # noqa: F722
) -> LossValues:
    labels = torch.flatten(batch[:, 1:, :], start_dim=0, end_dim=1)
    logits = torch.flatten(logits[:, :-1, :], start_dim=0, end_dim=1)

    kind_loss = None
    note_on_loss = None
    note_off_loss = None
    time_loss = None
    key_signature_loss = None

    for token_field in TOKEN_FIELDS:
        start, end = one_hot_parsing.piece_range(token_field)
        loss = torch.nn.functional.cross_entropy(
            logits[:, start:end],
            labels[:, start:end],
        )
        if token_field == "kind":
            kind_loss = loss
        elif token_field == "note_on":
            note_on_loss = loss
        elif token_field == "note_off":
            note_off_loss = loss
        elif token_field == "time":
            time_loss = loss
        elif token_field == "key_signature":
            key_signature_loss = loss
        else:
            raise ValueError(token_field)

    assert kind_loss is not None
    assert note_on_loss is not None
    assert note_off_loss is not None
    assert time_loss is not None
    assert key_signature_loss is not None

    return LossValues(
        loss=kind_loss + note_on_loss + note_off_loss + time_loss + key_signature_loss,
        kind_loss=kind_loss or torch.zeros(()),
        note_on_loss=note_on_loss or torch.zeros(()),
        note_off_loss=note_off_loss or torch.zeros(()),
        time_loss=time_loss or torch.zeros(()),
        key_signature_loss=key_signature_loss or torch.zeros(()),
    )
