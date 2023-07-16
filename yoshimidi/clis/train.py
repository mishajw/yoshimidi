import dataclasses
import json
import pathlib
from datetime import datetime

import dotenv
import fire
import torch
import tqdm
import wandb
from loguru import logger
from torch.utils.data import DataLoader

from yoshimidi.data.midi_dataset import MidiDataset
from yoshimidi.train.flops import calculate_flops, calculate_num_parameters
from yoshimidi.train.midi_loss import autoregressive_midi_loss
from yoshimidi.train.training_config import TrainingConfig
from yoshimidi.train.transformer import Transformer
from yoshimidi.train.transformer_config import GPT_CONFIG

dotenv.load_dotenv()


def main(dataset_path: str):
    dataset_path: pathlib.Path = pathlib.Path(dataset_path).expanduser()
    transformer_config = GPT_CONFIG
    training_config = TrainingConfig()
    logger.info("Starting training")
    logger.info(
        "Training config: " + json.dumps(dataclasses.asdict(training_config), indent=2)
    )
    logger.info(
        "Transformer config: "
        + json.dumps(dataclasses.asdict(transformer_config), indent=2)
    )
    logger.info(f"Num parameters: {calculate_num_parameters(transformer_config):.2E}")

    logger.info("Initializing WandB...")
    wandb.login()
    wandb.init(project="yoshimidi", name="2023-07-15_v1", dir=".wandb")
    logger.info("Initializing model...")
    model = Transformer(transformer_config)
    logger.info("Initializing dataset...")
    dataset = MidiDataset.from_path(
        dataset_path, context_window=training_config.context_window
    )
    data_loader = DataLoader(
        dataset, batch_size=training_config.batch_size, shuffle=True
    )
    logger.info("Initializing optimizer...")
    optimizer = torch.optim.Adam(model.parameters())

    logger.info("Starting training...")
    bar = tqdm.tqdm(data_loader)
    for batch in bar:
        start_time = datetime.now()
        outputs = model(batch)
        loss_values = autoregressive_midi_loss(batch=batch, outputs=outputs)
        loss_values.loss.backward()
        optimizer.step()
        time_per_batch_secs = (datetime.now() - start_time).total_seconds()
        flops = calculate_flops(
            transformer_config, training_config, time_per_batch_secs
        )
        metrics = {
            "loss/loss": loss_values.loss.item(),
            "loss/kind": loss_values.kind_loss.item(),
            "loss/note_key": loss_values.note_key_loss.item(),
            "loss/note_octave": loss_values.note_octave_loss.item(),
            "loss/time": loss_values.time_loss.item(),
            "perf/time_per_batch_secs": time_per_batch_secs,
            "perf/flops": flops,
        }
        bar.set_postfix(metrics)
        wandb.log(metrics)


if __name__ == "__main__":
    fire.Fire(main)
