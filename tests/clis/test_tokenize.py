import pathlib
from tempfile import TemporaryDirectory

import numpy as np

from yoshimidi.clis.tokenize_midi_dataset import _tokenize
from yoshimidi.data.parse import token_parsing
from yoshimidi.data.parse.tracks import Channel, Note


def test_single_file() -> None:
    with TemporaryDirectory() as temp_dir:
        root = pathlib.Path(temp_dir)
        (root / "output").mkdir()
        _tokenize(
            [
                Channel(
                    notes=[
                        Note(1, "on", 0, 0),
                        Note(1, "off", 0, 1),
                        Note(2, "on", 0, 1),
                        Note(2, "off", 0, 2),
                        Note(3, "on", 0, 4),
                        Note(3, "off", 0, 4.5),
                    ],
                    program_nums=[],
                ),
                Channel(
                    notes=[
                        Note(3, "on", 0, 0),
                        Note(3, "off", 0, 1),
                        Note(4, "on", 0, 1),
                        Note(4, "off", 0, 2),
                    ],
                    program_nums=[],
                ),
            ],
            output_dir=root / "output",
            lines_per_file=256,
        )

        end_indices = np.fromfile(
            root / "output" / "end_indices_0000.npy", dtype=np.uint32
        ).tolist()
        assert end_indices == [6 + 1, 6 + 1 + 4 + 1]

        memmap = np.memmap(
            root / "output" / "tokens_0000.npy", dtype=token_parsing.DTYPE
        ).reshape(-1, token_parsing.TOKEN_DIM)
        assert memmap[: 6 + 1, 0].tolist() == [0, 1, 0, 1, 0, 1, 2]
        assert memmap[6 + 1 : 6 + 1 + 4 + 1, 0].tolist() == [0, 1, 0, 1, 2]
        assert np.all(memmap[6 + 1 + 4 + 1 :, 0] == 0)


def test_multiple_files() -> None:
    with TemporaryDirectory() as temp_dir:
        root = pathlib.Path(temp_dir)
        (root / "output").mkdir()
        _tokenize(
            [
                Channel(
                    notes=[
                        Note(1, "on", 0, 0),
                        Note(1, "off", 0, 1),
                        Note(2, "on", 0, 1),
                        Note(2, "off", 0, 2),
                        Note(3, "on", 0, 4),
                        Note(3, "off", 0, 4.5),
                    ],
                    program_nums=[],
                ),
                Channel(
                    notes=[
                        Note(3, "on", 0, 0),
                        Note(3, "off", 0, 1),
                        Note(4, "on", 0, 1),
                        Note(4, "off", 0, 2),
                    ],
                    program_nums=[],
                ),
            ],
            output_dir=root / "output",
            lines_per_file=8,
        )

        assert {p.name for p in (root / "output").iterdir()} == {
            "tokens_0000.npy",
            "tokens_0001.npy",
            "end_indices_0000.npy",
            "end_indices_0001.npy",
        }

        end_indices = np.fromfile(
            root / "output" / "end_indices_0000.npy", dtype=np.int32
        ).tolist()
        assert end_indices == [6 + 1]

        memmap = np.memmap(
            root / "output" / "tokens_0000.npy", dtype=token_parsing.DTYPE
        ).reshape(-1, token_parsing.TOKEN_DIM)
        assert memmap[: 6 + 1, 0].tolist() == [0, 1, 0, 1, 0, 1, 2]
        assert memmap[6 + 1 :, 0].tolist() == [0]

        end_indices = np.fromfile(
            root / "output" / "end_indices_0001.npy", dtype=np.int32
        ).tolist()
        assert end_indices == [4 + 1]

        memmap = np.memmap(
            root / "output" / "tokens_0001.npy", dtype=token_parsing.DTYPE
        ).reshape(-1, token_parsing.TOKEN_DIM)
        assert memmap[: 4 + 1, 0].tolist() == [0, 1, 0, 1, 2]
        assert memmap[4 + 1 :, 0].tolist() == [0, 0, 0]
