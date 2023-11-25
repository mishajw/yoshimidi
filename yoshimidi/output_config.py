from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import toml
from pydantic import BaseModel


@dataclass
class CheckpointInfo:
    step: int
    path: Path


class OutputConfig(BaseModel, extra="forbid"):
    root: Path = Path("out")

    dataset_dir: Path = root / "dataset"
    dataset_raw_compressed: Path = dataset_dir / "00_raw.tar.gz"
    dataset_raw: Path = dataset_dir / "01_raw"
    dataset_parsed: Path = dataset_dir / "02_parsed.jsonl"
    dataset_parsed_metadata: Path = dataset_dir / "02_parsed.jsonl.metadata.json"
    dataset_tokenized: Path = dataset_dir / "03_tokenized"

    checkpoints: Path = root / "checkpoints"

    def has_checkpoints(self, tag: str) -> bool:
        return (self.checkpoints / tag).exists()

    def get_checkpoint(
        self, tag: str, step: int | Literal["rolling", "latest"]
    ) -> Path:
        if step == "latest":
            return self.get_all_checkpoints(tag=tag)[-1].path
        elif step == "rolling":
            return self.checkpoints / tag / "rolling"
        else:
            return self.checkpoints / tag / f"step_{step:06d}"

    def get_all_checkpoints(self, tag: str) -> list[CheckpointInfo]:
        batch_paths = list((self.checkpoints / tag).iterdir())
        assert len(batch_paths) > 0, (self, tag)
        checkpoints: list[CheckpointInfo] = []
        for p in batch_paths:
            checkpoint_info_file = p / "checkpoint_info.toml"
            with open(checkpoint_info_file, "r") as f:
                checkpoint_info = toml.load(f)
            step = checkpoint_info["step"]
            assert isinstance(step, int), "Step field is not an integer"
            checkpoint = CheckpointInfo(step=step, path=p)
            checkpoints.append(checkpoint)
        checkpoints.sort(key=lambda x: x.step)
        return checkpoints
