import torch
import tqdm
from pydantic import BaseModel
from torch.utils.data import DataLoader

from yoshimidi.output_config import OutputConfig
from yoshimidi.train.midi_loss import LossValues, autoregressive_midi_loss
from yoshimidi.train.step_schedule import StepSchedule
from yoshimidi.train.transformer import Transformer


class EvalConfig(BaseModel, extra="forbid"):
    schedule: StepSchedule
    split: float
    batch_size: int


@torch.no_grad()
def evaluate(
    tag: str,
    step: int,
    model: Transformer,
    output_config: OutputConfig,
    *,
    data_loader_eval: DataLoader[torch.Tensor],
) -> LossValues:
    model.eval()
    losses = []
    for batch in tqdm.tqdm(data_loader_eval, desc="Evaluating"):
        logits = model(batch)
        losses.append(autoregressive_midi_loss(batch=batch, logits=logits))
    return LossValues(
        loss=torch.stack([loss.loss for loss in losses]).mean(),
        kind_loss=torch.stack([loss.kind_loss for loss in losses]).mean(),
        note_on_loss=torch.stack([loss.note_on_loss for loss in losses]).mean(),
        note_off_loss=torch.stack([loss.note_off_loss for loss in losses]).mean(),
        time_loss=torch.stack([loss.time_loss for loss in losses]).mean(),
    )
