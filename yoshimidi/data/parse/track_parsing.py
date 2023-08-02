from collections import defaultdict

import mido
import msgspec
import numpy as np
from loguru import logger
from mido import Message, MidiTrack

from yoshimidi.data.parse import token_parsing
from yoshimidi.data.parse.tracks import Channel, Note, Track, TrackMetadata

DEFAULT_TICKS_PER_BEAT = 120
DEFAULT_TEMPO = 447761


def from_midi(
    midi_track: MidiTrack,
    *,
    tempo: int,
    ticks_per_beat: float,
    log_warnings: bool = True,
) -> Track | None:
    track = Track(
        channels=defaultdict(lambda: Channel(notes=[], program_nums=[])),
        metadata=TrackMetadata(),
    )

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
        channel_num: msgspec.structs.replace(
            channel, notes=_shift_time_deltas(channel.notes)
        )
        for channel_num, channel in track.channels.items()
        if len(channel.notes) > 0
    }
    return track


def parse_tempo(midi_file: mido.MidiFile, log_warnings: bool = True) -> int | None:
    tempos = {
        message.tempo
        for track in midi_file.tracks
        for message in track
        if isinstance(message, mido.MetaMessage) and message.type == "set_tempo"
    }
    if len(tempos) != 1:
        if log_warnings:
            logger.warning(f"Expected exactly one tempo: {tempos}")
        return None
    (tempo,) = tempos
    return tempo


def _parse_note(
    message: Message, *, ticks_per_beat: float, tempo: float
) -> Note | None:
    """
    N.B.: This function must be combnied with _shift_time_deltas() to get the correct
    time deltas.
    """
    if message.is_meta or message.type == "sysex":
        return None
    if message.channel == 9:
        return None  # Skip drum channel.
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


def _shift_time_deltas(notes: list[Note]) -> list[Note]:
    shifted_notes = []
    for note1, note2 in zip(notes[:-1], notes[1:], strict=True):
        shifted_notes.append(
            Note(
                note=note1.note,
                kind=note1.kind,
                velocity=note1.velocity,
                time_delta_secs=note2.time_delta_secs,
            )
        )
    shifted_notes.append(
        Note(
            note=notes[-1].note,
            kind=notes[-1].kind,
            velocity=notes[-1].velocity,
            time_delta_secs=0,
        )
    )
    return shifted_notes


def from_tokens(channel_tokens: list[np.ndarray]) -> Track:
    channels = []
    for channel in channel_tokens:
        assert channel.shape[1] == token_parsing.TOKEN_DIM
        notes: list[Note] = []

        for index in range(channel.shape[0]):
            kind = token_parsing.get_kind(channel[index])
            if kind == "end":
                assert index == channel.shape[0] - 1
                break
            note = token_parsing.get_note(channel[index])
            time_secs = token_parsing.get_time_secs(channel[index])
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
