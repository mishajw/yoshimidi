from pydantic import BaseModel


class TrainingConfig(BaseModel, extra="forbid"):
    context_window: int
    batch_size: int
