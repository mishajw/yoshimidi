import pathlib
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from yoshimidi.data.parse import one_hot_parsing, token_parsing


@dataclass
class MidiDataset(Dataset):
    context_window: int
    device: torch.device
    dtype: torch.dtype
    end_indices: List[List[int]]
    memmap: List[np.ndarray]
    memmap_cum_tokens: List[int]

    @classmethod
    def from_path(
        cls,
        path: pathlib.Path,
        context_window: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "MidiDataset":
        all_end_indices = []
        all_memmaps = []
        memmap_cum_tokens = [0]
        end_indices_paths = sorted(path.glob("end_indices_*.npy"))
        assert len(end_indices_paths) > 0, f"No end indices found in {path}"
        for end_indices_path in end_indices_paths:
            end_indices = np.fromfile(end_indices_path, dtype=np.int32).tolist()
            index = end_indices_path.stem[len("end_indices_") :]
            memmap = np.memmap(
                path / f"tokens_{index}.npy", dtype=token_parsing.DTYPE, mode="r"
            ).reshape((-1, token_parsing.TOKEN_DIM))
            assert memmap.shape == (2**22, token_parsing.TOKEN_DIM), memmap.shape
            memmap = memmap[: end_indices[-1]]
            all_end_indices.append(end_indices)
            all_memmaps.append(memmap)
            memmap_cum_tokens.append(end_indices[-1] + memmap_cum_tokens[-1])
        return MidiDataset(
            context_window=context_window,
            device=device,
            dtype=dtype,
            end_indices=all_end_indices,
            memmap=all_memmaps,
            memmap_cum_tokens=memmap_cum_tokens,
        )

    def __len__(self) -> int:
        num_token = sum(end_indices[-1] for end_indices in self.end_indices)
        return num_token // self.context_window

    def __getitem__(self, index: int) -> Tensor:
        return one_hot_parsing.from_tokens(
            self._get_tokens(index), self.device, self.dtype
        )

    def _get_tokens(self, index: int) -> np.ndarray:
        token_start = index * self.context_window
        token_end = (index + 1) * self.context_window
        first_bigger_tokens_idx = next(
            idx
            for idx, cum_tokens in enumerate(self.memmap_cum_tokens)
            if cum_tokens > token_start
        )
        memmap_idx = first_bigger_tokens_idx - 1
        cum_tokens = self.memmap_cum_tokens[memmap_idx]
        result = self.memmap[memmap_idx][
            token_start - cum_tokens : token_end - cum_tokens
        ]
        # TODO: Instead of padding, combine with next memmap.
        result = np.pad(result, [(0, self.context_window - result.shape[0]), (0, 0)])
        assert result.shape == (
            self.context_window,
            token_parsing.TOKEN_DIM,
        ), result.shape
        return result
