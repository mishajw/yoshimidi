import torch
from jaxtyping import Float

from yoshimidi.data.parse.one_hot_parsing import VOCAB
from yoshimidi.train.model.mlp import Mlp
from yoshimidi.train.model.multi_head_attention import MultiHeadAttention
from yoshimidi.train.model.positional_encoding import PositionalEncoding
from yoshimidi.train.model.transformer_config import TransformerConfig


class Gpt(torch.nn.Module):
    """GPT implementation.

    GPT uses the original Transformer architecture.

    TODO: Use GELU activation.

    # ruff: noqa: E501
    Architecture links:
    - GPT: https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf
    - Transformer: https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.token_embeddings = torch.nn.Parameter(
            torch.randn((VOCAB, config.residual_stream_size)),
            requires_grad=True,
        )
        self.token_unembeddings = torch.nn.Parameter(
            torch.randn((config.residual_stream_size, VOCAB)),
            requires_grad=True,
        )
        self.blocks = [_TransformerBlock(config) for _ in range(config.num_layers)]
        self.positional_encoding = PositionalEncoding(
            config.residual_stream_size, config.context_window
        )
        for i, block in enumerate(self.blocks):
            self.add_module(f"block_{i}", block)

    def forward(
        self,
        tokens: Float[torch.Tensor, "batch seq vocab"],  # noqa: F722
    ) -> Float[torch.Tensor, "batch seq vocab"]:  # noqa: F722
        residual_stream = tokens @ self.token_embeddings
        residual_stream = self.positional_encoding(residual_stream)
        for block in self.blocks:
            residual_stream = block(residual_stream)
        return residual_stream @ self.token_unembeddings


class _TransformerBlock(torch.nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.mlp = Mlp(config)
        self.layer_norm_attention = torch.nn.LayerNorm(config.residual_stream_size)
        self.layer_norm_mlp = torch.nn.LayerNorm(config.residual_stream_size)

    def forward(
        self,
        residual_stream: Float[torch.Tensor, "batch seq resid"],  # noqa: F722
    ) -> Float[torch.Tensor, "batch seq resid"]:  # noqa: F722
        residual_stream = self.layer_norm_attention(
            residual_stream + self.attention(residual_stream),
        )
        residual_stream = self.layer_norm_mlp(
            residual_stream + self.mlp(residual_stream),
        )
        return residual_stream
