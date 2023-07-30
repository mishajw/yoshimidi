from collections import defaultdict
from typing import Optional

import mido
import numpy as np
from loguru import logger
from mido import Message, MidiTrack

from yoshimidi.data.parse import token_parsing
from yoshimidi.data.parse.tracks import Channel, Note, Track, TrackMetadata
from yoshimidi.data.token_format import VOCAB

DEFAULT_TICKS_PER_BEAT = 120
DEFAULT_TEMPO = 447761


def from_midi(
    midi_track: MidiTrack,
    *,
    ticks_per_beat: float,
    log_warnings: bool = True,
) -> Optional[Track]:
    track = Track(
        channels=defaultdict(lambda: Channel(notes=[], program_nums=[])),
        metadata=TrackMetadata(),
    )

    tempos = {message.tempo for message in midi_track if message.type == "set_tempo"}
    if len(tempos) > 1:
        if log_warnings:
            logger.warning(f"Multiple tempos found: {tempos}")
        return None
    tempo = list(tempos)[0] if tempos else 500000

    message: Message
    for message in midi_track:
        if message.type == "stop":
            if log_warnings:
                logger.warning(f"Found stop message: {message}")
            return None
        note = _parse_note(
            message,
            ticks_per_beat=ticks_per_beat,
            tempo=tempo,
        )
        if note is not None:
            track.channels[message.channel].notes.append(note)

    track.channels = {
        channel_num: channel
        for channel_num, channel in track.channels.items()
        if len(channel.notes) > 0
    }
    return track


def _parse_note(
    message: Message, *, ticks_per_beat: float, tempo: float
) -> Optional[Note]:
    if message.channel == 9:
        return None  # Skip drum channel.
    if message.is_meta or message.type == "sysex":
        return None
    time_secs = mido.tick2second(
        tick=message.time, ticks_per_beat=ticks_per_beat, tempo=tempo
    )
    if message.type == "note_on" and message.velocity > 0:
        return Note(
            note=message.note,
            kind="on",
            velocity=message.velocity,
            time_delta_secs=time_secs,
        )
    elif (
        message.type == "note_off"
        or message.type == "note_on"
        and message.velocity == 0
    ):
        return Note(
            note=message.note,
            kind="off",
            velocity=message.velocity,
            time_delta_secs=time_secs,
        )
    else:
        return None


def from_tokens(channel_tokens: list[np.ndarray]) -> Track:
    channels = []
    for channel in channel_tokens:
        assert channel.shape[1] == VOCAB
        notes = []

        idx = 0
        while idx < channel.shape[0]:
            kind = token_parsing.get_kind(channel[idx])
            if kind == "end":
                assert idx == channel.shape[0] - 1
                break
            assert kind == "pause", kind
            time_secs = token_parsing.get_time_secs(channel[idx])
            idx += 1

            kind = token_parsing.get_kind(channel[idx])
            assert kind == "on" or kind == "off", kind
            note = token_parsing.get_note(channel[idx])
            idx += 1

            notes.append(
                Note(
                    note=note,
                    kind=kind,
                    velocity=127,
                    time_delta_secs=time_secs,
                )
            )
        channels.append(Channel(notes=notes, program_nums=[]))
    return Track(
        channels={i: c for i, c in enumerate(channels)},
        metadata=TrackMetadata(),
    )
