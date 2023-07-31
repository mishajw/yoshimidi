from pydantic import BaseModel
from torch.utils.data import DataLoader

from yoshimidi.output_config import OutputConfig
from yoshimidi.train.step_schedule import StepSchedule
from yoshimidi.train.transformer import Transformer


class EvalConfig(BaseModel, extra="forbid"):
    schedule: StepSchedule
    split: float = 0.1


def evaluate(
    tag: str,
    step: int,
    model: Transformer,
    output_config: OutputConfig,
    *,
    data_loader_eval: DataLoader,
) -> None:
    pass
