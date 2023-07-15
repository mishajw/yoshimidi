import pathlib
from tempfile import TemporaryDirectory

import numpy as np
from scripts.tokenize import _tokenize

from yoshimidi.data.token_parsing import VOCAB
from yoshimidi.data.tracks import Channel, Note


def test_single_file():
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
            root / "output" / "end_indicies_0000.npy", dtype=np.int32
        ).tolist()
        assert end_indices == [6 * 2, 6 * 2 + 4 * 2]

        memmap = np.memmap(
            root / "output" / "tokens_0000.npy", dtype=np.float32
        ).reshape(-1, VOCAB)

        assert memmap[0, 0:4].tolist() == [1, 0, 0, 0]  # on
        assert memmap[1, 0:4].tolist() == [0, 0, 1, 0]  # pause
        assert memmap[2, 0:4].tolist() == [0, 1, 0, 0]  # off
        assert memmap[3, 0:4].tolist() == [0, 0, 1, 0]  # pause
        assert memmap[4, 0:4].tolist() == [1, 0, 0, 0]  # on
        assert memmap[5, 0:4].tolist() == [0, 0, 1, 0]  # pause
        assert memmap[6, 0:4].tolist() == [0, 1, 0, 0]  # off
        assert memmap[7, 0:4].tolist() == [0, 0, 1, 0]  # pause
        assert memmap[8, 0:4].tolist() == [1, 0, 0, 0]  # on
        assert memmap[9, 0:4].tolist() == [0, 0, 1, 0]  # pause
        assert memmap[10, 0:4].tolist() == [0, 1, 0, 0]  # off
        assert memmap[11, 0:4].tolist() == [0, 0, 0, 1]  # end

        assert memmap[12, 0:4].tolist() == [1, 0, 0, 0]  # on
        assert memmap[13, 0:4].tolist() == [0, 0, 1, 0]  # pause
        assert memmap[14, 0:4].tolist() == [0, 1, 0, 0]  # off
        assert memmap[15, 0:4].tolist() == [0, 0, 1, 0]  # pause
        assert memmap[16, 0:4].tolist() == [1, 0, 0, 0]  # on
        assert memmap[17, 0:4].tolist() == [0, 0, 1, 0]  # pause
        assert memmap[18, 0:4].tolist() == [0, 1, 0, 0]  # off
        assert memmap[19, 0:4].tolist() == [0, 0, 0, 1]  # end

        assert np.all(memmap[20:] == 0)


def test_multiple_files():
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
            lines_per_file=13,
        )

        assert {p.name for p in (root / "output").iterdir()} == {
            "tokens_0000.npy",
            "tokens_0001.npy",
            "end_indicies_0000.npy",
            "end_indicies_0001.npy",
        }

        end_indices = np.fromfile(
            root / "output" / "end_indicies_0000.npy", dtype=np.int32
        ).tolist()
        assert end_indices == [6 * 2]

        memmap = np.memmap(
            root / "output" / "tokens_0000.npy", dtype=np.float32
        ).reshape(-1, VOCAB)
        assert memmap[0, 0:4].tolist() == [1, 0, 0, 0]  # on
        assert memmap[1, 0:4].tolist() == [0, 0, 1, 0]  # pause
        assert memmap[2, 0:4].tolist() == [0, 1, 0, 0]  # off
        assert memmap[3, 0:4].tolist() == [0, 0, 1, 0]  # pause
        assert memmap[4, 0:4].tolist() == [1, 0, 0, 0]  # on
        assert memmap[5, 0:4].tolist() == [0, 0, 1, 0]  # pause
        assert memmap[6, 0:4].tolist() == [0, 1, 0, 0]  # off
        assert memmap[7, 0:4].tolist() == [0, 0, 1, 0]  # pause
        assert memmap[8, 0:4].tolist() == [1, 0, 0, 0]  # on
        assert memmap[9, 0:4].tolist() == [0, 0, 1, 0]  # pause
        assert memmap[10, 0:4].tolist() == [0, 1, 0, 0]  # off
        assert memmap[11, 0:4].tolist() == [0, 0, 0, 1]  # end

        end_indices = np.fromfile(
            root / "output" / "end_indicies_0001.npy", dtype=np.int32
        ).tolist()
        assert end_indices == [4 * 2]

        memmap = np.memmap(
            root / "output" / "tokens_0001.npy", dtype=np.float32
        ).reshape(-1, VOCAB)
        assert memmap[0, 0:4].tolist() == [1, 0, 0, 0]  # on
        assert memmap[1, 0:4].tolist() == [0, 0, 1, 0]  # pause
        assert memmap[2, 0:4].tolist() == [0, 1, 0, 0]  # off
        assert memmap[3, 0:4].tolist() == [0, 0, 1, 0]  # pause
        assert memmap[4, 0:4].tolist() == [1, 0, 0, 0]  # on
        assert memmap[5, 0:4].tolist() == [0, 0, 1, 0]  # pause
        assert memmap[6, 0:4].tolist() == [0, 1, 0, 0]  # off
        assert memmap[7, 0:4].tolist() == [0, 0, 0, 1]  # end

        assert np.all(memmap[20:] == 0)
