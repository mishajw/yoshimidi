import pathlib

import fire
import tqdm
from loguru import logger
from torch.utils.data import DataLoader

from yoshimidi.data.midi_dataset import MidiDataset
from yoshimidi.train.transformer import Transformer, TransformerConfig


def main(dataset_path: str = "~/Downloads/yoshimidi/tokenized"):
    dataset_path: pathlib.Path = pathlib.Path(dataset_path).expanduser()

    logger.info("Initializing model...")
    model = Transformer(TransformerConfig())
    logger.info("Initializing dataset...")
    dataset = MidiDataset.from_path(dataset_path, context_window=1024)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    logger.info("Starting training...")
    for batch in tqdm.tqdm(data_loader):
        model(batch)


if __name__ == "__main__":
    fire.Fire(main)
