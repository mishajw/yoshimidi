from datetime import datetime

import dotenv
import fire
import torch
import tqdm
import wandb
from loguru import logger
from pydantic import BaseModel
from torch.utils.data import DataLoader

from yoshimidi.data.midi_dataset import MidiDataset
from yoshimidi.output_config import OutputConfig
from yoshimidi.train import checkpoints
from yoshimidi.train.checkpoints import CheckpointConfig
from yoshimidi.train.flops import calculate_flops, calculate_num_parameters
from yoshimidi.train.midi_loss import autoregressive_midi_loss
from yoshimidi.train.training_config import TrainingConfig
from yoshimidi.train.transformer import Transformer
from yoshimidi.train.transformer_config import TransformerConfig

dotenv.load_dotenv()


class Config(BaseModel, extra="forbid"):
    tag: str = "2023-07-16_v1"
    output: OutputConfig = OutputConfig()
    transformer: TransformerConfig = TransformerConfig(
        num_layers=3,
        residual_stream_size=128,
        num_attention_heads=4,
    )
    training: TrainingConfig = TrainingConfig(
        context_window=1024,
        batch_size=32,
    )
    checkpoint: CheckpointConfig = CheckpointConfig(
        every_n_steps=100,
    )
    use_wandb: bool = False


def main():
    config = Config()
    logger.info("Starting training")
    logger.info("Config: {}", config.model_dump_json(indent=2))
    logger.info(f"Num parameters: {calculate_num_parameters(config.transformer):.2E}")

    if config.use_wandb:
        wandb.login()
        wandb.init(project="yoshimidi", name=config.tag, dir=".wandb")

    logger.debug("Loading model")
    model = Transformer(config.transformer)
    optimizer = torch.optim.Adam(model.parameters())
    logger.debug(
        f"Num loaded parameters: {sum(p.numel() for p in model.parameters()):.2E}"
    )

    logger.debug("Loading dataset")
    dataset = MidiDataset.from_path(
        config.output.dataset_tokenized, context_window=config.training.context_window
    )
    data_loader = DataLoader(
        dataset, batch_size=config.training.batch_size, shuffle=True
    )
    logger.debug(f"Num tokens: {len(dataset) * config.training.context_window:.2E}")
    logger.debug(f"Num rows: {len(dataset):.2E}")
    logger.debug(f"Num batches: {len(data_loader):.2E}")

    bar = tqdm.tqdm(data_loader, desc="Training")
    for step, batch in enumerate(bar):
        optimizer.zero_grad()
        start_time = datetime.now()
        logits = model(batch)
        loss_values = autoregressive_midi_loss(batch=batch, logits=logits)
        loss_values.loss.backward()
        optimizer.step()
        time_per_batch_secs = (datetime.now() - start_time).total_seconds()
        flops = calculate_flops(
            config.transformer, config.training, time_per_batch_secs
        )
        metrics = {
            "loss/loss": loss_values.loss.item(),
            "loss/kind": loss_values.kind_loss.item(),
            "loss/note_key": loss_values.note_key_loss.item(),
            "loss/note_octave": loss_values.note_octave_loss.item(),
            "loss/time": loss_values.time_loss.item(),
            "perf/time_per_batch_secs": time_per_batch_secs,
            "perf/flops": flops,
            **{
                f"norms/{name}": param.norm().item()
                for name, param in model.named_parameters()
            },
            **{
                f"grad_norms/{name}": param.grad.norm().item()
                if param.grad is not None
                else -1
                for name, param in model.named_parameters()
            },
        }
        bar.set_postfix(loss=metrics["loss/loss"], flops=metrics["perf/flops"])
        if config.use_wandb:
            wandb.log(metrics)

        if config.checkpoint.should_save(step=step, max_steps=len(data_loader)):
            checkpoints.save_checkpoint(
                tag=config.tag,
                step=step,
                model=model,
                optimizer=optimizer,
                output_config=config.output,
            )


if __name__ == "__main__":
    fire.Fire(main)
