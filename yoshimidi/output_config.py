import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

_CHECKPOINT_NAME_REGEX = re.compile(r"step_(\d+)")


@dataclass
class CheckpointInfo:
    index: int
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

    def get_checkpoint(self, tag: str, step: int) -> Path:
        return self.checkpoints / tag / f"step_{step:06d}"

    def get_all_checkpoints(self, tag: str) -> list[CheckpointInfo]:
        batch_paths = list((self.checkpoints / tag).iterdir())
        assert len(batch_paths) > 0, (self, tag)
        checkpoints: list[CheckpointInfo] = []
        for p in batch_paths:
            match = _CHECKPOINT_NAME_REGEX.fullmatch(p.name)
            assert match is not None, p
            index = int(match.group(1))
            checkpoint = CheckpointInfo(index=index, path=p)
            checkpoints.append(checkpoint)
        checkpoints.sort(key=lambda x: x.index)
        return checkpoints

    def get_latest_checkpoint(self, tag: str) -> Path:
        return self.get_all_checkpoints(tag=tag)[-1].path
