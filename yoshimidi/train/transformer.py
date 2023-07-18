import torch
from jaxtyping import Float

from yoshimidi.data.token_format import VOCAB
from yoshimidi.train.transformer_config import TransformerConfig


class Transformer(torch.nn.Module):
    """GPT-J implementation.

    - GPT-J uses GPT-3 architecture, but with:
      - Rotary Position Embeddings (RoPE).
      - Dense attention.
    - GPT-3 uses GPT-2 architecture, but with sparse attention (which we drop a la
    GPT-J).
    - GPT-2 uses GPT architecture, but with:
      - Layer normalization is moved to the input of each sub-block.
      - Layer normalization is added after the final self-attention block.
      - Weights of residual layers are scaled by a factor of 1/sqrt(N) where N is the
      number of residual layers.
    - GPT uses the original Transformer architecture.

    TODO: Implement GPT-2/3/J architectures, currently only base GPT/Transformers are
    implemented.

    # ruff: noqa: E501
    Architecture links:
    - GPT-J: https://en.wikipedia.org/wiki/GPT-J#Architecture
    - GPT-3: https://arxiv.org/pdf/2005.14165.pdf#page=8
    - GPT-2: https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf#page=4
    - GPT: https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf
    - Transformer: https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.token_embeddings = torch.nn.Parameter(
            torch.randn((VOCAB, config.residual_stream_size)), requires_grad=True
        )
        self.blocks = [_TransformerBlock(config) for _ in range(config.num_layers)]
        self.positional_encoding = _PositionalEncoding(config.residual_stream_size)
        for i, block in enumerate(self.blocks):
            self.add_module(f"block_{i}", block)

    def forward(
        self,
        tokens: Float[torch.Tensor, "batch seq vocab"],  # noqa: F722
    ) -> Float[torch.Tensor, "batch seq vocab"]:  # noqa: F722
        residual_stream = tokens @ (
            self.token_embeddings * self.config.residual_stream_size**0.5
        )
        residual_stream = self.positional_encoding(residual_stream)
        for block in self.blocks:
            residual_stream = block(residual_stream)
        return residual_stream @ self.token_embeddings.T


class _TransformerBlock(torch.nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attention = _MultiHeadAttention(config)
        self.mlp = _Mlp(config)
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


class _MultiHeadAttention(torch.nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attention_heads = [
            _AttentionHead(config) for _ in range(config.num_attention_heads)
        ]
        for i, attention_head in enumerate(self.attention_heads):
            self.add_module(f"attention_head_{i}", attention_head)
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
    def __init__(self, config: TransformerConfig):
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
        _, seq_len, _ = head_stream.shape
        qkv = self.qkv_layer(head_stream)
        queries = qkv[:, :, 0 : self.attention_head_size]
        keys = qkv[:, :, self.attention_head_size : self.attention_head_size * 2]  # ()
        values = qkv[:, :, self.attention_head_size * 2 :]
        attention_logits = queries @ keys.transpose(2, 1)
        attention_mask = torch.triu(
            torch.full(
                (seq_len, seq_len),
                fill_value=-1e6,
                dtype=head_stream.dtype,
            ),
            diagonal=1,
        )
        attention = torch.nn.functional.softmax(
            (attention_logits * attention_mask) / (self.attention_head_size**0.5),
            dim=2,
        )
        return attention @ values


class _Mlp(torch.nn.Module):
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


class _PositionalEncoding(torch.nn.Module):
    def __init__(self, residual_stream_size: int):
        super().__init__()
        self.encodings: torch.Tensor | None = None
        self.residual_stream_size = residual_stream_size

    def forward(self, x: torch.Tensor, seq_dim: int = -2) -> torch.Tensor:
        if (
            self.encodings is None
            or self.encodings.shape[0] != x.shape[seq_dim]
            or self.encodings.dtype != x.dtype
        ):
            self.encodings = self._generate_encodings(x, seq_dim, x.dtype)
        return x + self.encodings

    def _generate_encodings(
        self, x: torch.Tensor, seq_dim: int, dtype: torch.dtype
    ) -> torch.Tensor:
        result = torch.tensor(
            [
                [
                    pos / 10000 ** (2 * i / x.shape[seq_dim])
                    for i in range(self.residual_stream_size)
                ]
                for pos in range(x.shape[seq_dim])
            ],
            dtype=dtype,
        )
        result[:, 0::2] = torch.sin(result[:, 0::2])
        result[:, 1::2] = torch.cos(result[:, 1::2])
        return result
