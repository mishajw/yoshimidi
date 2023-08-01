from pydantic import BaseModel


class TransformerConfig(BaseModel, extra="forbid"):
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


GPT_CONFIG = TransformerConfig(
    num_layers=6,
    residual_stream_size=512,
    num_attention_heads=8,
    context_window=1024,
)
