import pathlib

import dotenv
import fire
import torch
import tqdm
from loguru import logger
from torch.utils.data import DataLoader

import wandb
from yoshimidi.data.midi_dataset import MidiDataset
from yoshimidi.train.midi_loss import autoregressive_midi_loss
from yoshimidi.train.transformer import Transformer, TransformerConfig

dotenv.load_dotenv()


def main(dataset_path: str):
    dataset_path: pathlib.Path = pathlib.Path(dataset_path).expanduser()

    logger.info("Initializing WandB...")
    wandb.login()
    wandb.init(project="yoshimidi", name="2023-07-15_v1")
    logger.info("Initializing model...")
    model = Transformer(TransformerConfig())
    logger.info("Initializing dataset...")
    dataset = MidiDataset.from_path(dataset_path, context_window=1024)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    logger.info("Initializing optimizer...")
    optimizer = torch.optim.Adam(model.parameters())

    logger.info("Starting training...")
    bar = tqdm.tqdm(data_loader)
    for batch in bar:
        outputs = model(batch)
        loss_values = autoregressive_midi_loss(batch=batch, outputs=outputs)
        loss_values.loss.backward()
        optimizer.step()
        metrics = dict(
            loss=loss_values.loss.item(),
            kind_loss=loss_values.kind_loss.item(),
            note_key_loss=loss_values.note_key_loss.item(),
            note_octave_loss=loss_values.note_octave_loss.item(),
            time_loss=loss_values.time_loss.item(),
        )

        bar.set_postfix(metrics)
        wandb.log(metrics)


if __name__ == "__main__":
    fire.Fire(main)
