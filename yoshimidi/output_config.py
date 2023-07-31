import re
from pathlib import Path

from pydantic import BaseModel

_CHECKPOINT_NAME_REGEX = re.compile(r"step_(\d+)\.pt")


class OutputConfig(BaseModel, extra="forbid"):
    root: Path = Path("out")

    dataset_dir: Path = root / "dataset"
    dataset_raw_compressed: Path = dataset_dir / "00_raw.tar.gz"
    dataset_raw: Path = dataset_dir / "01_raw"
    dataset_parsed: Path = dataset_dir / "02_parsed.jsonl"
    dataset_tokenized: Path = dataset_dir / "03_tokenized"

    checkpoints: Path = root / "checkpoints"

    def has_checkpoints(self, tag: str) -> bool:
        return (self.checkpoints / tag).exists()

    def get_checkpoint(self, tag: str, step: int) -> Path:
        return self.checkpoints / tag / f"step_{step:06d}.pt"

    def get_latest_checkpoint(self, tag: str) -> Path:
        batch_paths = list((self.checkpoints / tag).iterdir())
        assert len(batch_paths) > 0, (self, tag)
        assert all(
            _CHECKPOINT_NAME_REGEX.fullmatch(p.name) is not None for p in batch_paths
        ), batch_paths
        return sorted(
            batch_paths,
            key=lambda p: int(
                _CHECKPOINT_NAME_REGEX.fullmatch(p.name).group(1),  # type: ignore
            ),
            reverse=True,
        )[0]
