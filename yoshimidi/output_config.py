from pathlib import Path

from pydantic import BaseModel


class OutputConfig(BaseModel, extra="forbid"):
    root: Path = Path("out")

    dataset_dir: Path = root / "dataset"
    dataset_parsed: Path = dataset_dir / "dataset_parsed"
    dataset_raw: Path = dataset_dir / "dataset_raw"
    dataset_raw_compressed: Path = dataset_dir / "dataset_raw.tar.gz"

    dataset_tokenized: Path = root / "dataset_tokenized"

    checkpoints: Path = root / "checkpoints"

    def has_checkpoints(self, tag: str) -> Path:
        return (self.checkpoints / tag).exists()

    def get_checkpoint(self, tag: str, step: int) -> Path:
        return self.checkpoints / tag / f"step_{step:06d}.pt"

    def get_latest_checkpoint(self, tag: str) -> Path:
        batch_paths = list((self.checkpoints / tag).iterdir())
        assert len(batch_paths) > 0, (self, tag)
        assert all(p.name.isdecimal() for p in batch_paths), batch_paths
        return sorted(batch_paths, key=lambda p: int(p.name), reverse=True)[0]
