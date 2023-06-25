import pathlib
from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, List, Optional

import mido
from mido import Message, MidiFile, MidiTrack
from mido.midifiles.meta import KeySignatureError

from yoshimidi.data.track import Channel, Note, Track, TrackMetadata


@dataclass
class ParseResult:
    tracks: List[Track]
    counters: DefaultDict[str, int]


def parse(path: pathlib.Path) -> ParseResult:
    counters: DefaultDict[str, int] = DefaultDict(int)
    try:
        midi_file = MidiFile(path)
    except (OSError, ValueError) as e:
        if "MThd not found" in str(e):
            counters["bad_header_chunk"] += 1
            return ParseResult(tracks=[], counters=counters)
        elif "data byte must be in range" in str(e):
            counters["bad_data_byte"] += 1
            return ParseResult(tracks=[], counters=counters)
        elif "running status without last_status" in str(e):
            counters["bad_status"] += 1
            return ParseResult(tracks=[], counters=counters)
        else:
            raise e
    except IndexError:
        counters["bad_list_index"] += 1
        return ParseResult(tracks=[], counters=counters)
    except KeySignatureError:
        counters["bad_key_signature"] += 1
        return ParseResult(tracks=[], counters=counters)
    except EOFError:
        counters["bad_eof"] += 1
        return ParseResult(tracks=[], counters=counters)
    tracks_with_failures: List[Optional[Track]] = [
        _parse_track(midi_track, counters, ticks_per_beat=midi_file.ticks_per_beat)
        for midi_track in midi_file.tracks
    ]
    tracks: List[Track] = [track for track in tracks_with_failures if track is not None]
    if len(tracks) == 0:
        counters["bad_empty_files"] += 1
        return ParseResult(tracks=[], counters=counters)
    counters["successful_files"] += 1
    return ParseResult(tracks=tracks, counters=counters)


def _parse_track(
    midi_track: MidiTrack, counters: DefaultDict[str, int], *, ticks_per_beat: float
) -> Optional[Track]:
    track = Track(
        channels=defaultdict(lambda: Channel(notes=[], program_nums=[])),
        metadata=TrackMetadata(),
    )

    tempos = [message.tempo for message in midi_track if message.type == "set_tempo"]
    if len(tempos) > 1:
        counters["bad_multiple_tempos"] += 1
        return None
    tempo = tempos[0] if tempos else 500000

    message: Message
    for message in midi_track:
        if message.is_meta or message.type == "sysex":
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
    counters["successful_tracks"] += 1
    counters["successful_channels"] += len(track.channels)
    counters["successful_notes"] += sum(
        len(channel.notes) for channel in track.channels.values()
    )
    return track


def _parse_event(
    message: Message, channel: Channel, *, ticks_per_beat: float, tempo: float
) -> None:
    time_secs = mido.tick2second(
        tick=message.time, ticks_per_beat=ticks_per_beat, tempo=tempo
    )
    if message.type == "note_on":
        channel.notes.append(
            Note(
                note=str(message.note),
                kind="on",
                velocity=message.velocity,
                time_secs=time_secs,
            )
        )
    elif message.type == "note_off":
        channel.notes.append(
            Note(
                note=str(message.note),
                kind="off",
                velocity=message.velocity,
                time_secs=time_secs,
            )
        )
