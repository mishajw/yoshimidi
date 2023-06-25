import multiprocessing
import pathlib
import shutil
from typing import DefaultDict

import fire
import msgspec
import requests
import tqdm
from loguru import logger

from yoshimidi.data import midi_parser

_LAKH_MIDI_DATASET_URL = "http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz"


def main(output_path: str):
    output_path: pathlib.Path = pathlib.Path(output_path).expanduser()

    logger.info("Starting")
    logger.info("output_path: {}", output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Stage 1: Downloading data")
    download_path = output_path / "dataset_raw.tar.gz"
    if not download_path.exists():
        with requests.get(_LAKH_MIDI_DATASET_URL, stream=True) as r:
            with open(download_path, "wb") as f:
                shutil.copyfileobj(r.raw, f)

    logger.info("Stage 2: Extracting data")
    extract_path = output_path / "dataset_raw"
    if not extract_path.exists():
        shutil.unpack_archive(download_path, extract_path)

    logger.info("Stage 3: Parsing data")
    parse_path = output_path / "dataset_parsed.jsonl"
    if not parse_path.exists():
        _parse_data_multiprocessing(extract_path, parse_path)


def _parse_data_multiprocessing(
    input_dir: pathlib.Path,
    output_file: pathlib.Path,
):
    counters: DefaultDict[str, int] = DefaultDict(int)
    midi_files = list(input_dir.rglob("*.mid"))
    with multiprocessing.Pool(processes=4) as pool, output_file.open("wb") as f:
        results = pool.imap_unordered(midi_parser.parse, midi_files)
        pbar = tqdm.tqdm(results, total=len(midi_files))
        for result in pbar:
            f.write(msgspec.json.encode(result.tracks) + b"\n")
            for counter_name, counter_value in result.counters.items():
                counters[counter_name] += counter_value
            pbar.set_postfix(
                file=counters["successful_files"],
                track=counters["successful_tracks"],
                note=counters["successful_notes"],
            )
    logger.info("Counters:")
    for counter_name, counter_value in counters.items():
        logger.info("{}: {}", counter_name, counter_value)


if __name__ == "__main__":
    fire.Fire(main)
