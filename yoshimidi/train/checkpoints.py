import torch
from loguru import logger
from pydantic import BaseModel

from yoshimidi.output_config import OutputConfig
from yoshimidi.train.transformer import Transformer


class CheckpointConfig(BaseModel, extra="forbid"):
    every_n_steps: int
    at_begin: bool = True
    at_end: bool = True

    def should_save(self, step: int, max_steps: int) -> bool:
        if self.at_begin and step == 0:
            return True
        if self.at_end and step == (max_steps - 1):
            return True
        return step > 0 and step % self.every_n_steps == 0


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
    step: int,
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    output_config: OutputConfig,
):
    checkpoint = torch.load(
        output_config.get_checkpoint(tag=tag, step=step),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, optimizer
