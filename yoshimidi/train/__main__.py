import pathlib

import fire

from yoshimidi.data.midi_dataset import MidiDataset


def main(dataset_path: str = "~/Downloads/yoshimidi/tokenized"):
    dataset_path: pathlib.Path = pathlib.Path(dataset_path).expanduser()
    dataset = MidiDataset.from_path(dataset_path, context_window=1024)
    for i in range(len(dataset)):
        assert dataset[i].sum() > 0


if __name__ == "__main__":
    fire.Fire(main)
