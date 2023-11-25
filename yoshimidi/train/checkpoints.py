import shutil
import tempfile
from pathlib import Path
from typing import Literal

import toml
import torch
from loguru import logger
from pydantic import BaseModel

from yoshimidi.output_config import OutputConfig
from yoshimidi.train.model import transformer
from yoshimidi.train.model.transformer_config import TransformerConfig
from yoshimidi.train.step_schedule import StepSchedule
from yoshimidi.train.training_config import TrainingConfig


class CheckpointConfig(BaseModel, extra="forbid"):
    schedule: StepSchedule
    rolling_schedule: StepSchedule


def maybe_save_checkpoints(
    tag: str,
    step: int,
    max_steps: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    transformer_config: TransformerConfig,
    training_config: TrainingConfig,
    checkpoint_config: CheckpointConfig,
    output_config: OutputConfig,
) -> None:
    if checkpoint_config.schedule.should_run(step=step, max_steps=max_steps):
        _save_checkpoint_using_tmp(
            tag=tag,
            step=step,
            model=model,
            optimizer=optimizer,
            transformer_config=transformer_config,
            training_config=training_config,
            output_config=output_config,
            check_exists=True,
        )
    if checkpoint_config.rolling_schedule.should_run(step=step, max_steps=max_steps):
        _save_checkpoint_using_tmp(
            tag=tag,
            step="rolling",
            model=model,
            optimizer=optimizer,
            transformer_config=transformer_config,
            training_config=training_config,
            output_config=output_config,
            check_exists=False,
        )


def load_checkpoint(
    tag: str,
    step: int | Literal["latest"],
    output_config: OutputConfig,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    if step == "latest":
        checkpoint_dir = output_config.get_latest_checkpoint(tag=tag)
    else:
        checkpoint_dir = output_config.get_checkpoint(tag=tag, step=step)
    logger.info("Loading checkpoint: {}", checkpoint_dir)

    transformer_config_path = checkpoint_dir / "transformer_config.toml"
    with open(transformer_config_path, "r") as f:
        transformer_config = TransformerConfig.model_validate(toml.load(f))
    model = transformer.load_model(transformer_config)
    model.load_state_dict(torch.load(checkpoint_dir / "model.pt", map_location=device))

    training_config_path = checkpoint_dir / "training_config.toml"
    with open(training_config_path, "r") as f:
        training_config = TrainingConfig.model_validate(toml.load(f))
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)
    optimizer.load_state_dict(
        torch.load(checkpoint_dir / "optimizer.pt", map_location=device)
    )

    return model, optimizer


def _save_checkpoint_using_tmp(
    tag: str,
    step: int | Literal["rolling"],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    transformer_config: TransformerConfig,
    training_config: TrainingConfig,
    output_config: OutputConfig,
    check_exists: bool,
) -> None:
    checkpoint_dir = output_config.get_checkpoint(tag=tag, step=step)
    logger.info("Saving checkpoint: {}", checkpoint_dir)
    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=checkpoint_dir.parent) as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        _save_checkpoint(
            output_dir=temp_dir,
            model=model,
            optimizer=optimizer,
            transformer_config=transformer_config,
            training_config=training_config,
        )
        if check_exists:
            assert (
                not checkpoint_dir.exists()
            ), f"Checkpoint directory already exists: {checkpoint_dir}"
        elif checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        temp_dir.rename(checkpoint_dir)
        logger.info(f"Checkpoint saved: {checkpoint_dir}")


def _save_checkpoint(
    output_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    transformer_config: TransformerConfig,
    training_config: TrainingConfig,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")
    torch.save(optimizer.state_dict(), output_dir / "optimizer.pt")
    with open(output_dir / "transformer_config.toml", "w") as f:
        toml.dump(transformer_config.model_dump(), f)
    with open(output_dir / "training_config.toml", "w") as f:
        toml.dump(training_config.model_dump(), f)
