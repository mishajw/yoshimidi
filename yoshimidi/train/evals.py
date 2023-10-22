import torch
import tqdm
from pydantic import BaseModel
from torch.utils.data import DataLoader

from yoshimidi.data.parse import one_hot_parsing
from yoshimidi.train.midi_loss import LossAndStats, autoregressive_midi_loss
from yoshimidi.train.step_schedule import StepSchedule
from yoshimidi.train.transformer import Transformer


class EvalConfig(BaseModel, extra="forbid"):
    schedule: StepSchedule
    split: float
    batch_size: int


@torch.no_grad()
def evaluate(
    model: Transformer,
    *,
    data_loader_eval: DataLoader[torch.Tensor],
) -> LossAndStats:
    model.eval()
    losses: list[LossAndStats] = []
    for batch in tqdm.tqdm(data_loader_eval, desc="Evaluating"):
        logits = model(batch)
        losses.append(autoregressive_midi_loss(batch=batch, logits=logits))
    return LossAndStats(
        loss=torch.stack([loss.loss for loss in losses]).mean(),
        range_stats={
            one_hot_range: LossAndStats.LossStat(
                value=torch.stack(
                    [loss.range_stats[one_hot_range].value for loss in losses]
                ).mean(),
                entropy=torch.stack(
                    [loss.range_stats[one_hot_range].entropy for loss in losses]
                ).mean(),
                target_entropy=torch.stack(
                    [loss.range_stats[one_hot_range].target_entropy for loss in losses]
                ).mean(),
                num_predicted=torch.stack(
                    [loss.range_stats[one_hot_range].num_predicted for loss in losses]
                ).mean(),
                num_target=torch.stack(
                    [loss.range_stats[one_hot_range].num_target for loss in losses]
                ).mean(),
            )
            for one_hot_range in one_hot_parsing.ONE_HOT_RANGE_LENGTHS
        },
    )
