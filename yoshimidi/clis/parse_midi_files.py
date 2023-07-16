#!/usr/bin/env python

import dataclasses
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
from yoshimidi.data.parse.tracks import Track
from yoshimidi.output_config import OutputConfig

_LAKH_MIDI_DATASET_URL = "http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz"


def main():
    config = OutputConfig()

    logger.info("Starting")
    logger.info("output_config: {}", dataclasses.asdict(config))
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
        _parse_data_multiprocessing(config.dataset_raw, config.dataset_parsed)


def _parse_data_multiprocessing(
    input_dir: pathlib.Path,
    output_file: pathlib.Path,
):
    counters: DefaultDict[str, int] = DefaultDict(int)
    midi_files = list(input_dir.rglob("*.mid"))
    with multiprocessing.Pool(processes=4) as pool, output_file.open("wb") as f:
        results = pool.imap_unordered(_parse_midi_path, midi_files)
        pbar = tqdm.tqdm(results, desc="Parsing files", total=len(midi_files))
        for result in pbar:
            f.write(msgspec.json.encode(result.tracks) + b"\n")
            for counter_name, counter_value in result.counters.items():
                counters[counter_name] += counter_value
            pbar.set_postfix(
                file=counters["successful_files"],
                track=counters["successful_tracks"],
                chans=counters["successful_channels"],
                note=counters["successful_notes"],
            )
    logger.info("Counters:")
    for counter_name, counter_value in counters.items():
        logger.info("{}: {}", counter_name, counter_value)


def _parse_midi_path(path: pathlib.Path) -> "_ParseResult":
    counters: DefaultDict[str, int] = DefaultDict(int)
    try:
        midi_file = MidiFile(path)
    except Exception:
        counters["bad_file"] += 1
        return _ParseResult(tracks=[], counters=counters)
    tracks_with_failures: list[Track | None] = [
        track_parsing.from_midi(
            midi_track, ticks_per_beat=midi_file.ticks_per_beat, log_warnings=False
        )
        for midi_track in midi_file.tracks
    ]
    tracks: list[Track] = [track for track in tracks_with_failures if track is not None]
    counters["successful_tracks"] += len(tracks)
    counters["successful_channels"] += sum(len(track.channels) for track in tracks)
    counters["successful_notes"] += sum(
        len(channel.notes) for track in tracks for channel in track.channels.values()
    )
    counters["failed_tracks"] += len(tracks_with_failures) - len(tracks)
    if len(tracks) == 0:
        counters["bad_empty_files"] += 1
        return _ParseResult(tracks=[], counters=counters)
    counters["successful_files"] += 1
    return _ParseResult(tracks=tracks, counters=counters)


@dataclass
class _ParseResult:
    tracks: list[Track]
    counters: DefaultDict[str, int]


if __name__ == "__main__":
    fire.Fire(main)
