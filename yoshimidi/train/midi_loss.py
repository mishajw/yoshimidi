from dataclasses import dataclass

import torch
from jaxtyping import Float

from yoshimidi.data.parse import one_hot_parsing


@dataclass
class LossAndStats:
    @dataclass
    class LossStat:
        value: torch.Tensor
        entropy: torch.Tensor
        target_entropy: torch.Tensor
        num_predicted: torch.Tensor
        num_target: torch.Tensor

    loss: torch.Tensor
    range_stats: dict[one_hot_parsing.OneHotRange, LossStat]
    # time: LossStat
    # note_on: LossStat
    # note_off: LossStat
    # end: LossStat
    # key_signature: LossStat


def autoregressive_midi_loss(
    *,
    batch: Float[torch.Tensor, "batch seq vocab"],  # noqa: F722
    logits: Float[torch.Tensor, "batch seq vocab"],  # noqa: F722
) -> LossAndStats:
    labels = batch[:, 1:, :].transpose(1, 2)
    logits = logits[:, :-1, :].transpose(1, 2)

    range_stats: dict[one_hot_parsing.OneHotRange, LossAndStats.LossStat] = dict()
    for one_hot_range in one_hot_parsing.ONE_HOT_RANGE_LENGTHS:
        start, end = one_hot_parsing.piece_range(one_hot_range)
        loss_value = torch.nn.functional.cross_entropy(
            _reduce_to_range(logits, start, end),
            _reduce_to_range(labels, start, end),
        )
        loss_entropy = _entropy(logits[:, :, start:end])
        loss_target_entropy = _entropy(labels[:, :, start:end])
        range_stats[one_hot_range] = LossAndStats.LossStat(
            value=loss_value,
            entropy=loss_entropy,
            target_entropy=loss_target_entropy,
            num_predicted=_num_in_range(logits, start, end),
            num_target=_num_in_range(labels, start, end),
        )

    loss = torch.nn.functional.cross_entropy(
        logits,
        labels,
    )
    return LossAndStats(
        loss=loss,
        range_stats=range_stats,
    )


def _reduce_to_range(t: torch.Tensor, start: int, end: int) -> torch.Tensor:
    non_range_sum = t[:, end:, :].sum(dim=1) + t[:, :start, :].sum(dim=1)
    return torch.concat(
        [t[:, start:end, :], non_range_sum.unsqueeze(1)],
        dim=1,
    )


def _entropy(logits: torch.Tensor) -> torch.Tensor:
    mean_probs = torch.nn.functional.softmax(logits, dim=1).mean(dim=(0, 2))
    return -torch.sum(mean_probs * torch.log(mean_probs + 1e-9))


def _num_in_range(t: torch.Tensor, start: int, end: int) -> torch.Tensor:
    in_range = (t.argmax(dim=1) >= start) & (t.argmax(dim=1) < end)
    return in_range.float().mean()
