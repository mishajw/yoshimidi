#!/usr/bin/env python

import json
import multiprocessing
import pathlib
import shutil
from dataclasses import dataclass
from typing import DefaultDict

import fire
import msgspec
import requests
import tqdm
from loguru import logger
from mido import MidiFile

from yoshimidi.data.parse import track_parsing
from yoshimidi.data.parse.tracks import KeySignature, Track
from yoshimidi.output_config import OutputConfig

_LAKH_MIDI_DATASET_URL = "http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz"


def main() -> None:
    config = OutputConfig()

    logger.info("Starting")
    logger.info(
        "output_config: {}", config.model_dump_json(indent=2, exclude_none=True)
    )
    config.dataset_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Stage 1: Downloading data")
    if not config.dataset_raw_compressed.exists():
        with requests.get(_LAKH_MIDI_DATASET_URL, stream=True) as r:
            with open(config.dataset_raw_compressed, "wb") as f:
                shutil.copyfileobj(r.raw, f)

    logger.info("Stage 2: Extracting data")
    if not config.dataset_raw.exists():
        shutil.unpack_archive(config.dataset_raw_compressed, config.dataset_raw)

    logger.info("Stage 3: Parsing data")
    if not config.dataset_parsed.exists():
        _parse_data_multiprocessing(
            input_dir=config.dataset_raw,
            output_file=config.dataset_parsed,
            output_metadata_file=config.dataset_parsed_metadata,
        )


def _parse_data_multiprocessing(
    *,
    input_dir: pathlib.Path,
    output_file: pathlib.Path,
    output_metadata_file: pathlib.Path,
) -> None:
    counters: DefaultDict[str, int] = DefaultDict(int)
    midi_files = list(input_dir.rglob("*.mid"))
    with multiprocessing.Pool() as pool, output_file.open("wb") as f:
        results = pool.imap_unordered(_parse_midi_path, midi_files)
        pbar = tqdm.tqdm(results, desc="Parsing files", total=len(midi_files))
        for result in pbar:
            f.write(msgspec.json.encode(result.track) + b"\n")
            for counter_name, counter_value in result.counters.items():
                counters[counter_name] += counter_value
            pbar.set_postfix(
                file=counters["successful_files"],
                chans=counters["successful_channels"],
                note=counters["successful_notes"],
                sigs=counters["successful_key_signatures"],
            )
    logger.info("Counters:")
    for counter_name, counter_value in counters.items():
        logger.info("{}: {}", counter_name, counter_value)
    with output_metadata_file.open("w") as f:
        json.dump(counters, f)


def _parse_midi_path(path: pathlib.Path) -> "_ParseResult":
    counters: DefaultDict[str, int] = DefaultDict(int)
    try:
        midi_file = MidiFile(path)
    except Exception:
        counters["bad_file"] += 1
        return _ParseResult(track=None, counters=counters)
    track = track_parsing.from_midi(midi_file, log_warnings=False)
    if track is None:
        counters["bad_track"] += 1
        return _ParseResult(track=None, counters=counters)
    counters["successful_files"] += 1
    counters["successful_channels"] += len(track.channels)
    counters["successful_notes"] += sum(
        len(channel.notes) for channel in track.channels.values()
    )
    counters["successful_key_signatures"] += sum(
        1
        for channel in track.channels.values()
        for note in channel.notes
        if isinstance(note, KeySignature)
    )
    return _ParseResult(track=track, counters=counters)


@dataclass
class _ParseResult:
    track: Track | None
    counters: DefaultDict[str, int]


if __name__ == "__main__":
    fire.Fire(main)
