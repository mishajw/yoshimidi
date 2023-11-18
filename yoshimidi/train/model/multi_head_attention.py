import torch
from jaxtyping import Float

from yoshimidi.train.model.transformer_config import TransformerConfig


class MultiHeadAttention(torch.nn.Module):
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
