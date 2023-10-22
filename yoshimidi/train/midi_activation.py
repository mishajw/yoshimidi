import torch
from jaxtyping import Float
from torch import Tensor


def midi_activation(
    logits: Float[Tensor, "batch seq vocab"]  # noqa: F722
) -> Float[Tensor, "batch seq vocab"]:  # noqa: F722
    return torch.nn.functional.softmax(logits, dim=2)
