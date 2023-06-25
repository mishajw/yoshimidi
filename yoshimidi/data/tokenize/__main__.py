import pathlib
from typing import List, Literal, Optional

import fire
import msgspec
import numpy as np
import tqdm
from jaxtyping import Float

from yoshimidi.data.track import Channel, Track

# jaxtyping
seq, vocab = None, None

_VOCAB = 28  # 4 + 12 + 11 + 1


def main(
    input_file: str,
    output_dir: str,
):
    input_file: pathlib.Path = pathlib.Path(input_file).expanduser()
    output_dir: pathlib.Path = pathlib.Path(output_dir).expanduser()
    assert input_file.exists(), f"input_file does not exist: {input_file}"
    output_dir.mkdir(parents=True, exist_ok=True)

    lines_per_file = 2**22
    mmap_index = -1
    mmap_lines = lines_per_file
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
            if mmap_lines + channel_lines > lines_per_file:
                mmap_index += 1
                mmap_lines = 0
                mmap = np.memmap(
                    output_dir / f"tokens_{mmap_index:04d}.npy",
                    dtype=np.float32,
                    shape=(lines_per_file, _VOCAB),
                    mode="w+",
                )
            _tokenize_channel(channel, mmap[mmap_lines : mmap_lines + channel_lines])
            mmap_lines += channel_lines
            pbar.set_postfix(idx=mmap_index, lines=mmap_lines)


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
