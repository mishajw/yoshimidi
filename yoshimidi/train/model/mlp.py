import torch
from jaxtyping import Float

from yoshimidi.train.model.transformer_config import TransformerConfig


class Mlp(torch.nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.linear_1 = torch.nn.Linear(
            in_features=config.residual_stream_size,
            out_features=config.feed_forward_size,
        )
        self.linear_2 = torch.nn.Linear(
            in_features=config.feed_forward_size,
            out_features=config.residual_stream_size,
        )

    def forward(
        self,
        residual_stream: Float[torch.Tensor, "batch seq resid"],  # noqa: F722
    ) -> Float[torch.Tensor, "batch seq resid"]:  # noqa: F722
        residual_stream = self.linear_1(residual_stream)
        residual_stream = torch.nn.functional.relu(residual_stream)
        residual_stream = self.linear_2(residual_stream)
        return residual_stream
