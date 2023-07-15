from collections import defaultdict
from typing import Optional

import mido
import numpy as np
from loguru import logger
from mido import Message, MidiTrack

from yoshimidi.data import token_parsing
from yoshimidi.data.tracks import Channel, Note, Track, TrackMetadata

# DEFAULT_TICKS_PER_BEAT = 480
# DEFAULT_TEMPO = 500000
DEFAULT_TICKS_PER_BEAT = 120
DEFAULT_TEMPO = 447761


def from_midi(midi_track: MidiTrack, *, ticks_per_beat: float) -> Optional[Track]:
    track = Track(
        channels=defaultdict(lambda: Channel(notes=[], program_nums=[])),
        metadata=TrackMetadata(),
    )

    tempos = {message.tempo for message in midi_track if message.type == "set_tempo"}
    if len(tempos) > 1:
        logger.warning(f"Multiple tempos found: {tempos}")
        return None
    tempo = list(tempos)[0] if tempos else 500000
    print(tempo, ticks_per_beat)

    message: Message
    for message in midi_track:
        if message.is_meta or message.type == "sysex":
            continue
        if message.type == "stop":
            return None
        if message.channel == 9:
            # Skip drum channel.
            continue
        _parse_event(
            message,
            track.channels[message.channel],
            ticks_per_beat=ticks_per_beat,
            tempo=tempo,
        )

    track.channels = {
        channel_num: channel
        for channel_num, channel in track.channels.items()
        if len(channel.notes) > 0
    }
    return track


def _parse_event(
    message: Message, channel: Channel, *, ticks_per_beat: float, tempo: float
) -> None:
    time_secs = mido.tick2second(
        tick=message.time, ticks_per_beat=ticks_per_beat, tempo=tempo
    )
    if message.type == "note_on" and message.velocity > 0:
        channel.notes.append(
            Note(
                note=message.note,
                kind="on",
                velocity=message.velocity,
                time_delta_secs=time_secs,
            )
        )
    elif (
        message.type == "note_off"
        or message.type == "note_on"
        and message.velocity == 0
    ):
        channel.notes.append(
            Note(
                note=message.note,
                kind="off",
                velocity=message.velocity,
                time_delta_secs=time_secs,
            )
        )


def from_tokens(channel_tokens: list[np.ndarray]) -> Track:
    channels = []
    for channel in channel_tokens:
        assert channel.shape[1] == 28
        notes = []

        idx = 0
        while idx < channel.shape[0]:
            kind = token_parsing.get_kind(channel[idx])
            assert kind == "on" or kind == "off", kind
            note = token_parsing.get_note(channel[idx])
            idx += 1

            pause_kind = token_parsing.get_kind(channel[idx])
            if pause_kind == "end":
                assert idx == channel.shape[0] - 1
                break
            assert pause_kind == "pause", pause_kind
            time_secs = token_parsing.get_time_secs(channel[idx])
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
