from dataclasses import dataclass

import torch
from jaxtyping import Float

from yoshimidi.data.parse.token_parsing import VOCAB


class Transformer(torch.nn.Module):
    def __init__(self, config: "TransformerConfig"):
        super().__init__()
        self.input_embeddings = torch.nn.Linear(
            in_features=VOCAB,
            out_features=config.residual_stream_size,
        )
        self.blocks = [_TransformerBlock(config) for _ in range(config.num_layers)]
        self.output_embeddings = torch.nn.Linear(
            in_features=config.residual_stream_size,
            out_features=VOCAB,
        )

    def forward(
        self,
        tokens: Float[torch.Tensor, "batch seq vocab"],  # noqa: F722
    ) -> Float[torch.Tensor, "batch seq vocab"]:  # noqa: F722
        residual_stream = self.input_embeddings(tokens)
        for block in self.blocks:
            residual_stream = block(residual_stream)
        outputs = self.output_embeddings(residual_stream)
        # TODO: MIDI softmax!
        return torch.nn.functional.softmax(outputs, dim=2)


class _TransformerBlock(torch.nn.Module):
    def __init__(self, config: "TransformerConfig"):
        super().__init__()
        self.attention = _MultiHeadAttention(config)
        self.mlp = _Mlp(config)
        self.layer_norm = torch.nn.LayerNorm(config.residual_stream_size)

    def forward(
        self,
        residual_stream: Float[torch.Tensor, "batch seq resid"],  # noqa: F722
    ) -> Float[torch.Tensor, "batch seq resid"]:  # noqa: F722
        residual_stream = self.layer_norm(residual_stream)
        return (
            residual_stream
            + self.attention(residual_stream)
            + self.mlp(residual_stream)
        )


class _MultiHeadAttention(torch.nn.Module):
    def __init__(self, config: "TransformerConfig"):
        super().__init__()
        self.attention_heads = [
            _AttentionHead(config) for _ in range(config.num_attention_heads)
        ]
        self.output_layer = torch.nn.Linear(
            in_features=config.residual_stream_size,
            out_features=config.residual_stream_size,
        )

    def forward(
        self,
        residual_stream: Float[torch.Tensor, "batch seq resid"],  # noqa: F722
    ) -> Float[torch.Tensor, "batch seq resid"]:  # noqa: F722
        outputs = torch.concat(
            [
                attention_head(residual_stream)
                for attention_head in self.attention_heads
            ],
            dim=2,
        )
        return self.output_layer(outputs)


class _AttentionHead(torch.nn.Module):
    def __init__(self, config: "TransformerConfig"):
        super().__init__()
        self.attention_head_size = config.attention_head_size
        self.qkv_layer = torch.nn.Linear(
            in_features=config.residual_stream_size,
            out_features=(self.attention_head_size * 3),
        )

    def forward(
        self,
        head_stream: Float[torch.Tensor, "batch seq head"],  # noqa: F722
    ) -> Float[torch.Tensor, "batch seq head"]:  # noqa: F722
        qkv = self.qkv_layer(head_stream)
        queries = qkv[:, :, 0 : self.attention_head_size]
        keys = qkv[:, :, self.attention_head_size : self.attention_head_size * 2]
        values = qkv[:, :, self.attention_head_size * 2 :]
        attention = torch.nn.functional.softmax(
            (queries @ keys.transpose(2, 1)) / (self.attention_head_size**0.5), dim=2
        )
        return attention @ values


class _Mlp(torch.nn.Module):
    def __init__(self, config: "TransformerConfig"):
        super().__init__()
        self.linear_1 = torch.nn.Linear(
            in_features=config.residual_stream_size,
            out_features=config.residual_stream_size * 4,
        )
        self.linear_2 = torch.nn.Linear(
            in_features=config.residual_stream_size * 4,
            out_features=config.residual_stream_size,
        )

    def forward(
        self,
        residual_stream: Float[torch.Tensor, "batch seq resid"],  # noqa: F722
    ) -> Float[torch.Tensor, "batch seq resid"]:  # noqa: F722
        residual_stream = self.linear_1(residual_stream)
        residual_stream = torch.nn.functional.gelu(residual_stream)
        residual_stream = self.linear_2(residual_stream)
        return residual_stream


@dataclass
class TransformerConfig:
    residual_stream_size: int = 128
    attention_head_size: int = 32
    num_attention_heads: int = 128 // 32
    num_layers: int = 3
