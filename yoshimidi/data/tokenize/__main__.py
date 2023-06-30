import pathlib
from dataclasses import dataclass
from typing import List, Literal, Optional

import fire
import msgspec
import numpy as np
import tqdm
from jaxtyping import Float

from yoshimidi.data.track import Channel, Track

# jaxtyping
seq, vocab = None, None

VOCAB = 28  # 4 + 12 + 11 + 1
_LINES_PER_FILE = 2**22


@dataclass
class _TokenizeState:
    index: int
    written_lines: int
    mmap: Optional[Float[np.memmap, "seq vocab"]]  # noqa: F722
    end_indices: List[int]
    output_dir: pathlib.Path

    def open_mmap(self) -> None:
        self.mmap = np.memmap(
            self.output_dir / f"tokens_{self.index:04d}.npy",
            dtype=np.float32,
            shape=(_LINES_PER_FILE, VOCAB),
            mode="w+",
        )

    def write_end_indicies(self) -> None:
        with (self.output_dir / f"end_indicies_{self.index:04d}.npy").open("wb") as f:
            np.array(self.end_indices, dtype=np.int32).tofile(f)

    def get_slice(self, num_lines: int) -> np.memmap:
        if self.written_lines + num_lines > _LINES_PER_FILE:
            self.next_index()
            return self.get_slice(num_lines)
        return self.mmap[self.written_lines : self.written_lines + num_lines]

    def register_lines_written(self, num_lines: int) -> None:
        self.written_lines += num_lines
        self.end_indices.append(self.written_lines)

    def next_index(self) -> None:
        self.index += 1
        self.written_lines = 0
        self.write_end_indicies()
        self.end_indices = []
        self.open_mmap()


def main(
    input_file: str,
    output_dir: str,
):
    input_file: pathlib.Path = pathlib.Path(input_file).expanduser()
    output_dir: pathlib.Path = pathlib.Path(output_dir).expanduser()
    assert input_file.exists(), f"input_file does not exist: {input_file}"
    output_dir.mkdir(parents=True, exist_ok=True)

    state = _TokenizeState(
        output_dir=output_dir, index=0, written_lines=0, mmap=None, end_indices=[]
    )
    state.open_mmap()

    with input_file.open("r") as f:
        channel_iter = (
            channel
            for line in f
            for track in msgspec.json.decode(line, type=List[Track])
            for channel in track.channels.values()
            if len(channel.notes) > 0
        )
        pbar = tqdm.tqdm(channel_iter, desc="Tokenizing channels")
        for channel in pbar:
            channel_lines = len(channel.notes) * 2
            _tokenize_channel(channel, state.get_slice(channel_lines))
            state.register_lines_written(channel_lines)
            pbar.set_postfix(idx=state.index, lines=state.written_lines)

    state.write_end_indicies()


def _tokenize_channel(
    channel: Channel, mmap: Float[np.memmap, "seq vocab"]  # noqa: F722
):
    index = 0
    for note_index in range(len(channel.notes)):
        note = channel.notes[note_index]
        _create_token(mmap[index], kind=note.kind, note=note.note)
        index += 1
        if note_index < len(channel.notes) - 1:
            _create_token(
                mmap[index],
                kind="pause",
                time=channel.notes[note_index + 1].time_secs - note.time_secs,
            )
            index += 1
        else:
            _create_token(mmap[index], kind="end")
            index += 1
    assert index == mmap.shape[0]


def _create_token(
    mmap: Float[np.memmap, "vocab"],
    kind: Literal["on", "off", "pause", "end"],
    *,
    note: Optional[int] = None,
    time: Optional[float] = None,
) -> Float[np.ndarray, "vocab"]:
    index = 0

    if kind == "on":
        mmap[0] = 1
    elif kind == "off":
        mmap[1] = 1
    elif kind == "pause":
        mmap[2] = 1
    elif kind == "end":
        mmap[3] = 1
    index += 4

    if note is not None:
        assert 0 <= note < 128, note
        mmap[index + note % 12] = 1
    index += 12
    if note is not None:
        mmap[index + note // 12] = 1
    index += 11

    if time is not None:
        mmap[index] = time
    index += 1

    assert index == mmap.shape[0], (index, mmap.shape[0])


if __name__ == "__main__":
    fire.Fire(main)
