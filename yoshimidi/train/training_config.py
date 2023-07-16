from dataclasses import dataclass


@dataclass
class TrainingConfig:
    context_window: int
    batch_size: int
