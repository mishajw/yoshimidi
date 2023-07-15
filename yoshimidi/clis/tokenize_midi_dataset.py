#!/usr/bin/env python

import pathlib
from dataclasses import dataclass
from typing import Iterable

import fire
import msgspec
import numpy as np
import tqdm
from jaxtyping import Float

from yoshimidi.data import token_parsing
from yoshimidi.data.tracks import Channel, Track


@dataclass
class _TokenizeState:
    index: int
    written_lines: int
    memmap: Float[np.memmap, "seq vocab"] | None  # noqa: F722
    end_indices: list[int]
    output_dir: pathlib.Path
    lines_per_file: int

    def open_mmap(self) -> None:
        self.memmap = np.memmap(
            self.output_dir / f"tokens_{self.index:04d}.npy",
            dtype=np.float32,
            shape=(self.lines_per_file, token_parsing.VOCAB),
            mode="w+",
        )

    def write_end_indicies(self) -> None:
        with (self.output_dir / f"end_indicies_{self.index:04d}.npy").open("wb") as f:
            np.array(self.end_indices, dtype=np.int32).tofile(f)

    def get_slice(self, num_lines: int) -> np.ndarray:
        assert self.memmap is not None
        if self.written_lines + num_lines > self.lines_per_file:
            self.next_index()
            return self.get_slice(num_lines)
        return self.memmap[self.written_lines : self.written_lines + num_lines]

    def register_lines_written(self, num_lines: int) -> None:
        self.written_lines += num_lines
        self.end_indices.append(self.written_lines)

    def next_index(self) -> None:
        self.write_end_indicies()
        self.index += 1
        self.written_lines = 0
        self.end_indices = []
        self.open_mmap()


def main(
    input_file: str,
    output_dir: str,
    lines_per_file: int = 2**22,
):
    input_file: pathlib.Path = pathlib.Path(input_file).expanduser()
    output_dir: pathlib.Path = pathlib.Path(output_dir).expanduser()
    assert input_file.exists(), f"input_file does not exist: {input_file}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_file.open("r") as f:
        _tokenize(
            channels=(
                channel
                for line in f
                for track in msgspec.json.decode(line, type=list[Track])
                for channel in track.channels.values()
                if len(channel.notes) > 0
            ),
            output_dir=output_dir,
            lines_per_file=lines_per_file,
        )


def _tokenize(
    channels: Iterable[Channel], output_dir: pathlib.Path, lines_per_file: int
):
    state = _TokenizeState(
        output_dir=output_dir,
        index=0,
        written_lines=0,
        memmap=None,
        end_indices=[],
        lines_per_file=lines_per_file,
    )
    state.open_mmap()
    pbar = tqdm.tqdm(channels, desc="Tokenizing channels")
    for channel in pbar:
        channel_lines, _ = token_parsing.get_buffer_size(channel)
        token_parsing.from_channel_to_buffer(channel, state.get_slice(channel_lines))
        state.register_lines_written(channel_lines)
        pbar.set_postfix(idx=state.index, lines=state.written_lines)
    state.write_end_indicies()


if __name__ == "__main__":
    fire.Fire(main)
