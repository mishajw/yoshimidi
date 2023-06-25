import itertools
import pathlib
import re
from typing import List, Literal, Optional

import fire
import msgspec
import numpy as np
from jaxtyping import Float, Int

from yoshimidi.data.track import Channel, Track

_NOTE_REGEX = re.compile(r"([A-G][b#]?)(\d+)")
_NOTES = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]


# jaxtyping
seq, vocab = None, None


def main(
    input_file: str,
    output_dir: str,
):
    input_file: pathlib.Path = pathlib.Path(input_file).expanduser()
    output_dir: pathlib.Path = pathlib.Path(output_dir).expanduser()
    assert input_file.exists(), f"input_file does not exist: {input_file}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_file.open("r") as f:
        for line in itertools.islice(f, 100):
            tracks = msgspec.json.decode(line, type=List[Track])
            for track in tracks:
                for channel in track.channels.values():
                    _tokenize_channel(channel)


def _tokenize_channel(channel: Channel) -> Int[np.ndarray, "seq vocab"]:  # noqa: F722
    result = []
    for note_index in range(len(channel.notes)):
        note = channel.notes[note_index]
        result.append(_create_token(kind=note.kind, note=note.note))
        if note_index == len(channel.notes) - 1:
            continue
        pause_time = channel.notes[note_index + 1].start - note.start
        if pause_time > 0:
            result.append(_create_token(kind="pause", time=pause_time))
    return np.stack(result)


def _create_token(
    kind: Literal["on", "off", "pause"],
    note: Optional[str] = None,
    time: Optional[float] = None,
) -> Float[np.ndarray, "vocab"]:
    kind_array = np.array(
        [kind == "on", kind == "off", kind == "pause"],
        dtype=np.float16,
    )

    note_array = np.zeros((12,), dtype=np.float16)
    octave_array = np.zeros((11,), dtype=np.float16)
    if note is not None:
        match = _NOTE_REGEX.match(note)
        assert match is not None
        note_index = _NOTES.index(match[1])
        note_octave = int(match[2])
        assert 3 <= note_octave <= 10, note
        if note_octave > 10:
            print(note)
            note_octave = 10
        note_array[note_index] = 1
        octave_array[note_octave - 3] = 1

    time_array = np.zeros((1,), dtype=np.float16)
    if time is not None:
        time_array[0] = np.log(time)

    return np.concatenate([kind_array, note_array, octave_array, time_array])


if __name__ == "__main__":
    fire.Fire(main)
