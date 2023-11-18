import torch


class PositionalEncoding(torch.nn.Module):
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
