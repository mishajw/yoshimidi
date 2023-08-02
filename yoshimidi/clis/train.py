from datetime import datetime

import dotenv
import fire
import toml
import torch
import tqdm
import wandb
from loguru import logger
from pydantic import BaseModel
from torch.utils.data import DataLoader

from yoshimidi.data.midi_dataset import MidiDataset
from yoshimidi.output_config import OutputConfig
from yoshimidi.train import checkpoints, evals
from yoshimidi.train.checkpoints import CheckpointConfig
from yoshimidi.train.evals import EvalConfig
from yoshimidi.train.flops import calculate_flops, calculate_num_parameters
from yoshimidi.train.midi_loss import autoregressive_midi_loss
from yoshimidi.train.training_config import TrainingConfig
from yoshimidi.train.transformer import Transformer
from yoshimidi.train.transformer_config import TransformerConfig

dotenv.load_dotenv()


class Config(BaseModel, extra="forbid"):
    tag: str
    output: OutputConfig = OutputConfig()
    transformer: TransformerConfig
    training: TrainingConfig
    checkpoint: CheckpointConfig
    eval: EvalConfig
    use_wandb: bool = True


def main(config_path: str) -> None:
    with open(config_path) as f:
        config = Config.model_validate(toml.load(f))
    logger.info("Starting training")
    logger.info("Config: {}", config.model_dump_json(indent=2))
    logger.info(f"Num parameters: {calculate_num_parameters(config.transformer):.2E}")
    assert not config.output.has_checkpoints(
        tag=config.tag
    ), f"Checkpoints already exist for tag: {config.tag}"

    if config.use_wandb:
        logger.info("Setting up WandB")
        wandb.login()
        wandb.init(
            project="yoshimidi",
            name=config.tag,
            dir=".wandb",
            config=config.model_dump(),
        )

    logger.debug("Loading model")
    model = Transformer(config.transformer).to(
        device=config.training.torch_device(), dtype=config.training.torch_dtype()
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    logger.debug(
        f"Num loaded parameters: {sum(p.numel() for p in model.parameters()):.2E}"
    )

    logger.debug("Loading dataset")
    dataset = MidiDataset.from_path(
        config.output.dataset_tokenized,
        context_window=config.transformer.context_window,
        device=config.training.torch_device(),
        dtype=config.training.torch_dtype(),
    )
    # TODO: We shouldn't split on the batch-level, as the same song could be split
    # between eval and train. Instead, we should split while parsing.
    dataset_eval, dataset_train = torch.utils.data.random_split(
        dataset, [config.eval.split, 1 - config.eval.split]
    )
    data_loader_train = DataLoader(
        dataset_train, batch_size=config.training.batch_size, shuffle=True
    )
    data_loader_eval = DataLoader(
        dataset_eval, batch_size=config.eval.batch_size, shuffle=True
    )
    logger.debug(f"Num tokens: {len(dataset) * config.transformer.context_window:.2E}")
    logger.debug(f"Num rows: {len(dataset):.2E}")
    logger.debug(f"Num batches: {len(data_loader_train):.2E}")
    logger.debug(f"Num batches (eval): {len(data_loader_eval):.2E}")

    bar = tqdm.tqdm(data_loader_train, desc="Training")
    for step, batch in enumerate(bar):
        model.train()
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

        if config.checkpoint.schedule.should_run(
            step=step, max_steps=len(data_loader_train)
        ):
            checkpoints.save_checkpoint(
                tag=config.tag,
                step=step,
                model=model,
                optimizer=optimizer,
                output_config=config.output,
            )

        if config.eval.schedule.should_run(step=step, max_steps=len(data_loader_train)):
            eval_loss = evals.evaluate(
                tag=config.tag,
                step=step,
                model=model,
                output_config=config.output,
                data_loader_eval=data_loader_eval,
            )
            bar.set_postfix(eval=eval_loss.loss.item())
            if config.use_wandb:
                wandb.log(
                    {
                        "evals/loss/loss": eval_loss.loss.item(),
                        "evals/loss/kind": eval_loss.kind_loss.item(),
                        "evals/loss/note_key": eval_loss.note_key_loss.item(),
                        "evals/loss/note_octave": eval_loss.note_octave_loss.item(),
                        "evals/loss/time": eval_loss.time_loss.item(),
                    }
                )


if __name__ == "__main__":
    fire.Fire(main)
