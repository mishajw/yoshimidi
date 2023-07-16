from pathlib import Path

from pydantic import BaseModel


class OutputConfig(BaseModel, extra="forbid"):
    root: Path = Path("out")

    dataset_dir: Path = root / "dataset"
    dataset_parsed: Path = dataset_dir / "dataset_parsed"
    dataset_raw: Path = dataset_dir / "dataset_raw"
    dataset_raw_compressed: Path = dataset_dir / "dataset_raw.tar.gz"

    dataset_tokenized: Path = root / "dataset_tokenized"
