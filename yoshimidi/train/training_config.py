import torch
from pydantic import BaseModel, validator

from yoshimidi.train.step_schedule import StepSchedule


class TrainingConfig(BaseModel, extra="forbid"):
    batch_size: int
    learning_rate: float
    device: str
    dtype: str
    metrics_schedule: StepSchedule

    def torch_device(self) -> torch.device:
        return torch.device(self.device)

    def torch_dtype(self) -> torch.dtype:
        if self.dtype == "float32":
            return torch.float32
        elif self.dtype == "float16":
            return torch.float16
        elif self.dtype == "bfloat16":
            return torch.bfloat16
        else:
            raise ValueError(self.dtype)

    @validator("device", pre=True)
    def _check_device(cls, v: str) -> str:
        torch.device(v)
        return v

    @validator("dtype", pre=True)
    def _check_dtype(cls, v: str) -> str:
        assert v in {"float32", "float16", "bfloat16"}, v
        return v
