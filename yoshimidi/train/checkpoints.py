import shutil
from typing import Literal

import toml
import torch
from loguru import logger
from pydantic import BaseModel

from yoshimidi.output_config import OutputConfig
from yoshimidi.train.model.transformer import Transformer
from yoshimidi.train.model.transformer_config import TransformerConfig
from yoshimidi.train.step_schedule import StepSchedule
from yoshimidi.train.training_config import TrainingConfig


class CheckpointConfig(BaseModel, extra="forbid"):
    schedule: StepSchedule
    # If set, we delete checkpoints that are older than this many steps.
    rolling: int | None = 5


def save_checkpoint(
    tag: str,
    step: int,
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    transformer_config: TransformerConfig,
    training_config: TrainingConfig,
    checkpoint_config: CheckpointConfig,
    output_config: OutputConfig,
) -> None:
    checkpoint_dir = output_config.get_checkpoint(tag=tag, step=step)
    assert (
        not checkpoint_dir.exists()
    ), f"Checkpoint directory already exists: {checkpoint_dir}"
    logger.info("Saving checkpoint: {}", checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), checkpoint_dir / "model.pt")
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    with open(checkpoint_dir / "transformer_config.toml", "w") as f:
        toml.dump(transformer_config.model_dump(), f)
    with open(checkpoint_dir / "training_config.toml", "w") as f:
        toml.dump(training_config.model_dump(), f)

    _handle_rolling_checkpoints(
        checkpoint_config=checkpoint_config,
        output_config=output_config,
        tag=tag,
    )


def load_checkpoint(
    tag: str,
    step: int | Literal["latest"],
    output_config: OutputConfig,
    device: torch.device,
) -> tuple[Transformer, torch.optim.Optimizer]:
    if step == "latest":
        checkpoint_dir = output_config.get_latest_checkpoint(tag=tag)
    else:
        checkpoint_dir = output_config.get_checkpoint(tag=tag, step=step)
    logger.info("Loading checkpoint: {}", checkpoint_dir)

    transformer_config_path = checkpoint_dir / "transformer_config.toml"
    with open(transformer_config_path, "r") as f:
        transformer_config = TransformerConfig.model_validate(toml.load(f))
    model = Transformer(transformer_config)
    model.load_state_dict(torch.load(checkpoint_dir / "model.pt", map_location=device))

    training_config_path = checkpoint_dir / "training_config.toml"
    with open(training_config_path, "r") as f:
        training_config = TrainingConfig.model_validate(toml.load(f))
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)
    optimizer.load_state_dict(
        torch.load(checkpoint_dir / "optimizer.pt", map_location=device)
    )

    return model, optimizer


def _handle_rolling_checkpoints(
    checkpoint_config: CheckpointConfig, output_config: OutputConfig, tag: str
) -> None:
    if checkpoint_config.rolling is None:
        return

    checkpoint_dirs = output_config.get_all_checkpoints(tag=tag)
    checkpoint_dirs.sort(key=lambda c: c.index, reverse=True)
    checkpoints_to_delete = checkpoint_dirs[: -checkpoint_config.rolling]

    for checkpoint in checkpoints_to_delete:
        logger.info("Deleting checkpoint: {}", checkpoint.path)
        shutil.rmtree(checkpoint.path)
