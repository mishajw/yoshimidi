from dataclasses import dataclass
import pathlib
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional

from MIDI import MIDIFile
from MIDI.chunks.track import Track as MIDITrack
from MIDI.Events import MIDIEvent
from MIDI.Events.meta import MetaEvent, MetaEventKinds
from loguru import logger
from yoshimidi.data import utils

from yoshimidi.data.track import Channel, Note, Track, TrackMetadata


@dataclass
class ParseResult:
    tracks: List[Track]
    counters: DefaultDict[str, int]


def parse(path: pathlib.Path) -> List[Track]:
    counters = DefaultDict(int)
    midi_file = MIDIFile(path)
    try:
        with utils.capture_output() as output:
            midi_file.parse()
        if len(output.getvalue()) > 0:
            counters["file_stdout_stderr_written_to"] += 1
            return ParseResult(tracks=[], counters=counters)
        tracks: List[Optional[Track]] = [_parse_track(midi_track, counters) for midi_track in midi_file]
        tracks: List[Track] = [track for track in tracks if track is not None]
        counters["successful_files"] += 1
        return ParseResult(tracks=tracks, counters=counters)
    except UnicodeDecodeError:
        counters["unicode_decode_error"] += 1
        return ParseResult(tracks=[], counters=counters)
    except Exception as e:
        if "Header chunk must have length" not in str(e):
            raise e
        counters["bad_header_chunk"] += 1
        return ParseResult(tracks=[], counters=counters)


def _parse_track(midi_track: MIDITrack, counters: DefaultDict[str, int]) -> Optional[Track]:
    with utils.capture_output() as output:
        midi_track.parse()
    if len(output.getvalue()) > 0:
        counters["track_stdout_stderr_written_to"] += 1
        return None
    track = Track(
        channels=defaultdict(lambda: Channel(notes=[], program_nums=[])),
        metadata=TrackMetadata(),
    )
    for midi_event in midi_track:
        if isinstance(midi_event, MetaEvent):
            _parse_metadata(midi_event, track.metadata)
        elif isinstance(midi_event, MIDIEvent):
            _parse_event(midi_event, track.channels[midi_event.channel])
        else:
            raise ValueError(f"Unrecognized event type: {type(midi_event)}")
    track.channels = {
        channel_num: channel
        for channel_num, channel in track.channels.items()
        if len(channel.notes) > 0
    }
    counters["successful_tracks"] += 1
    counters["successful_channels"] += len(track.channels)
    counters["successful_notes"] += sum(len(channel.notes) for channel in track.channels.values())
    return track


def _parse_metadata(event: MetaEvent, output: TrackMetadata) -> None:
    if event.type == MetaEventKinds.Set_Tempo.value:
        output.bpm = event.attributes["bpm"]
    elif event.type == MetaEventKinds.Time_Signature.value:
        output.time_signature = f"{event.attributes['numerator']}/{event.attributes['denominator']}"
    elif event.type == MetaEventKinds.Key_Signature.value:
        output.key = f"{event.attributes['key']} {event.attributes['mode']}"


def _parse_event(event: MIDIEvent, channel: Channel) -> None:
    command_str = MIDIEvent.commands.get(event.command, None)
    if command_str == "NOTE_ON":
        channel.notes.append(
            Note(
                note=str(event.message.note),
                kind="on",
                velocity=event.message.velocity,
                start=event.time,
            )
        )
    elif command_str == "NOTE_OFF":
        channel.notes.append(
            Note(
                note=str(event.message.note),
                kind="off",
                velocity=event.message.velocity,
                start=event.time,
            )
        )
    elif command_str == "PROGRAM_CHANGE":
        channel.program_nums.append(
            event.message.value,
        )
