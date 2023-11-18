from typing import Literal

from pydantic import BaseModel


class TransformerConfig(BaseModel, extra="forbid"):
    type: Literal["gpt", "gpt2"]
    num_layers: int
    residual_stream_size: int  # d_model
    num_attention_heads: int  # h
    context_window: int

    @property
    def attention_head_size(self) -> int:
        assert self.residual_stream_size % self.num_attention_heads == 0, self
        return self.residual_stream_size // self.num_attention_heads

    @property
    def feed_forward_size(self) -> int:
        return self.residual_stream_size * 4
