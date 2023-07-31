from typing import Literal

import torch
from loguru import logger
from pydantic import BaseModel

from yoshimidi.output_config import OutputConfig
from yoshimidi.train.step_schedule import StepSchedule
from yoshimidi.train.transformer import Transformer


class CheckpointConfig(BaseModel, extra="forbid"):
    schedule: StepSchedule


def save_checkpoint(
    tag: str,
    step: int,
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    output_config: OutputConfig,
):
    checkpoint_path = output_config.get_checkpoint(tag=tag, step=step)
    logger.info("Saving checkpoint: {}", checkpoint_path)
    assert not checkpoint_path.exists(), f"Checkpoint already exists: {checkpoint_path}"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )


def load_checkpoint(
    tag: str,
    step: int | Literal["latest"],
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    output_config: OutputConfig,
):
    if step == "latest":
        checkpoint_path = output_config.get_latest_checkpoint(tag=tag)
    else:
        checkpoint_path = output_config.get_checkpoint(tag=tag, step=step)
    logger.info("Loading checkpoint: {}", checkpoint_path)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, optimizer
