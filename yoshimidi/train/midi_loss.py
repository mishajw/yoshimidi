from dataclasses import dataclass

import torch
from jaxtyping import Float

from yoshimidi.data.parse import one_hot_parsing


@dataclass
class LossValues:
    loss: torch.Tensor
    time_loss: torch.Tensor
    note_on_loss: torch.Tensor
    note_off_loss: torch.Tensor
    end_loss: torch.Tensor
    key_signature_loss: torch.Tensor


def autoregressive_midi_loss(
    *,
    batch: Float[torch.Tensor, "batch seq vocab"],  # noqa: F722
    logits: Float[torch.Tensor, "batch seq vocab"],  # noqa: F722
) -> LossValues:
    labels = torch.flatten(batch[:, 1:, :], start_dim=0, end_dim=1)
    logits = torch.flatten(logits[:, :-1, :], start_dim=0, end_dim=1)

    pause_loss = None
    note_on_loss = None
    note_off_loss = None
    end_loss = None
    key_signature_loss = None

    for one_hot_range in one_hot_parsing.ONE_HOT_RANGE_LENGTHS:
        start, end = one_hot_parsing.piece_range(one_hot_range)
        range_loss = torch.nn.functional.cross_entropy(
            _reduce_to_range(logits, start, end),
            _reduce_to_range(labels, start, end),
        )
        if one_hot_range == "pause":
            pause_loss = range_loss
        elif one_hot_range == "note_on":
            note_on_loss = range_loss
        elif one_hot_range == "note_off":
            note_off_loss = range_loss
        elif one_hot_range == "end":
            end_loss = range_loss
        elif one_hot_range == "key_signature":
            key_signature_loss = range_loss
        else:
            raise ValueError(one_hot_range)

    assert pause_loss is not None
    assert note_on_loss is not None
    assert note_off_loss is not None
    assert end_loss is not None
    assert key_signature_loss is not None

    loss = torch.nn.functional.cross_entropy(
        logits,
        labels,
    )
    return LossValues(
        loss=loss,
        time_loss=pause_loss,
        note_on_loss=note_on_loss,
        note_off_loss=note_off_loss,
        end_loss=end_loss,
        key_signature_loss=key_signature_loss,
    )


def _reduce_to_range(t: torch.Tensor, start: int, end: int) -> torch.Tensor:
    non_range_sum = t[:, end:].sum(dim=1) + t[:, :start].sum(dim=1)
    return torch.concat(
        [t[:, start:end], non_range_sum.unsqueeze(1)],
        dim=1,
    )
