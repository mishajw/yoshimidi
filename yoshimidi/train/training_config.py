from dataclasses import dataclass


@dataclass
class TrainingConfig:
    context_window: int = 1024
    batch_size: int = 32
