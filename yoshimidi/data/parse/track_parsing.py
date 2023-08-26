from collections import defaultdict

import msgspec
import numpy as np
from loguru import logger
from mido import Message, MidiFile

from yoshimidi.data.parse import token_parsing
from yoshimidi.data.parse.tracks import Channel, Note, Track, TrackMetadata


def from_midi(
    midi_file: MidiFile,
    *,
    log_warnings: bool = True,
) -> Track | None:
    assert midi_file.type in [0, 1], midi_file.type
    channels: dict[int, Channel] = defaultdict(
        lambda: Channel(notes=[], program_nums=[])
    )
    cum_secs: float = 0.0
    channel_cum_secs: dict[int, float] = defaultdict(lambda: 0.0)
    for message in midi_file:
        if message.type == "stop":
            if log_warnings:
                logger.warning(f"Found stop message: {message}")
            return None
        cum_secs += message.time
        if not hasattr(message, "channel"):
            continue
        message_channel_delta_secs = cum_secs - channel_cum_secs[message.channel]
        channel_cum_secs[message.channel] = cum_secs
        note = _parse_note(message, time_secs=message_channel_delta_secs)
        if note is None:
            continue
        channels[message.channel].notes.append(note)
    track = Track(
        channels={
            channel_num: msgspec.structs.replace(
                # The above time deltas are the deltas *before* the note was played.
                # We want the deltas to be after, so we shift them.
                channel,
                notes=_shift_time_deltas(channel.notes),
            )
            for channel_num, channel in channels.items()
        },
        metadata=TrackMetadata(),
    )
    return track


def _parse_note(message: Message, *, time_secs: float) -> Note | None:
    """
    N.B.: This function must be combnied with _shift_time_deltas() to get the correct
    time deltas.
    """
    if message.is_meta or message.type == "sysex":
        return None
    if message.channel == 9:
        return None  # Skip drum channel.
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

        note_buffer = None
        for index in range(channel.shape[0]):
            kind = token_parsing.get_kind(channel[index])

            if note_buffer is not None and kind != "pause":
                notes.append(note_buffer)
                note_buffer = None

            if kind == "end":
                assert index == channel.shape[0] - 1
                break

            elif kind == "on":
                note = token_parsing.get_note_on(channel[index])
                note_buffer = Note(
                    note=note,
                    kind="on",
                    velocity=127,
                    time_delta_secs=0,
                )

            elif kind == "off":
                note = token_parsing.get_note_off(channel[index])
                note_buffer = Note(
                    note=note,
                    kind="off",
                    velocity=127,
                    time_delta_secs=0,
                )

            elif kind == "pause":
                time_delta_secs = token_parsing.get_time_secs(channel[index])
                if note_buffer is not None:
                    if note_buffer.time_delta_secs != 0:
                        logger.warning("Found multiple pause tokens in a row")
                    note_buffer.time_delta_secs += time_delta_secs
                else:
                    logger.warning("Found pause without preceding note")

        if note_buffer is not None:
            notes.append(note_buffer)
        channels.append(Channel(notes=notes, program_nums=[]))
    return Track(
        channels={i: c for i, c in enumerate(channels)},
        metadata=TrackMetadata(),
    )


def _split_channels(messages: list[Message]) -> dict[int, list[Message]]:
    channels = list(
        set(message.channel for message in messages if hasattr(message, "channel"))
    )
    if channels == []:
        channels = [0]
    time_buffers = {channel: 0.0 for channel in channels}
    result: dict[int, list[Message]] = {channel: [] for channel in channels}
    for message in messages:
        message_channel = getattr(message, "channel", channels[0])
        for channel in channels:
            if channel != message_channel:
                time_buffers[channel] += message.time
        message.time += time_buffers[message_channel]
        time_buffers[message_channel] = 0.0
        result[message_channel].append(message)
    return result
