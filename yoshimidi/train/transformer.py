import torch
from jaxtyping import Float

from yoshimidi.data.parse.one_hot_parsing import VOCAB
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
            torch.randn((VOCAB, config.residual_stream_size)),
            requires_grad=True,
        )
        self.blocks = [_TransformerBlock(config) for _ in range(config.num_layers)]
        self.positional_encoding = _PositionalEncoding(
            config.residual_stream_size, config.context_window
        )
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
        self.qkv_projection = torch.nn.Linear(
            in_features=config.residual_stream_size,
            out_features=config.residual_stream_size * 3,
        )
        self.output_layer = torch.nn.Linear(
            in_features=config.residual_stream_size,
            out_features=config.residual_stream_size,
        )
        self.residual_stream_size = config.residual_stream_size
        self.attention_head_size = config.attention_head_size
        self.num_attention_heads = config.num_attention_heads

    def forward(
        self,
        residual_stream: Float[torch.Tensor, "batch seq resid"],  # noqa: F722
    ) -> Float[torch.Tensor, "batch seq resid"]:  # noqa: F722
        batch, seq, _ = residual_stream.shape
        # shape=(batch, seq, resid * 3)
        qkv = self.qkv_projection(residual_stream)
        # shape=(batch, seq, resid)
        q, k, v = qkv.split(self.residual_stream_size, dim=2)
        # We want the last two dimensions to be sequence length & embedding size, as
        # this is what scaled_dot_product_attention expects. So we translate to:
        # shape=(batch, attn_head, seq, attn_head_size)
        q = q.view(batch, seq, self.num_attention_heads, self.attention_head_size)
        q = q.transpose(1, 2)
        k = k.view(batch, seq, self.num_attention_heads, self.attention_head_size)
        k = k.transpose(1, 2)
        v = v.view(batch, seq, self.num_attention_heads, self.attention_head_size)
        v = v.transpose(1, 2)
        result = torch.nn.functional.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            is_causal=True,
        )
        # shape=(batch, seq, resid)
        result = (
            result.transpose(1, 2)
            .contiguous()
            .view(batch, seq, self.residual_stream_size)
        )
        return self.output_layer(result)


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
    def __init__(self, residual_stream_size: int, context_window: int):
        super().__init__()
        self.encodings: torch.Tensor | None = None
        self.residual_stream_size = residual_stream_size
        self.context_window = context_window

    def forward(self, x: torch.Tensor, seq_dim: int = -2) -> torch.Tensor:
        if (
            self.encodings is None
            or self.encodings.shape[0] != x.shape[seq_dim]
            or self.encodings.dtype != x.dtype
        ):
            self.encodings = self._generate_encodings(x, seq_dim, x.dtype)
        # TODO: Am I handling the encodings var correctly?
        return x + self.encodings.to(device=x.device)

    def _generate_encodings(
        self, x: torch.Tensor, seq_dim: int, dtype: torch.dtype
    ) -> torch.Tensor:
        result = torch.tensor(
            [
                [
                    pos / 10000 ** (2 * i / self.context_window)
                    for i in range(self.residual_stream_size)
                ]
                for pos in range(x.shape[seq_dim])
            ],
            dtype=dtype,
        )
        result[:, 0::2] = torch.sin(result[:, 0::2])
        result[:, 1::2] = torch.cos(result[:, 1::2])
        return result
