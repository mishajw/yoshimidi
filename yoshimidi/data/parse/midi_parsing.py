import mido
from mido import Message, MetaMessage, MidiFile, MidiTrack

from yoshimidi.data.parse.track_parsing import DEFAULT_TEMPO, DEFAULT_TICKS_PER_BEAT
from yoshimidi.data.parse.tracks import Note, Track


def from_tracks(tracks: list[Track]) -> MidiFile:
    midi_file = MidiFile()
    midi_file.ticks_per_beat = DEFAULT_TICKS_PER_BEAT
    for track in tracks:
        assert (
            len(track.channels) == 1
        ), f"Only supports single-channel tracks, found {len(track.channels)}"
        midi_track = MidiTrack()
        midi_file.tracks.append(midi_track)
        # TODO: Set program_nums.
        midi_track.append(MetaMessage("set_tempo", tempo=DEFAULT_TEMPO, time=0))

        for channel_num, channel in track.channels.items():
            midi_track.append(
                _parse_message(channel.notes[0], channel_num=channel_num, time=0)
            )
            for previous_note, note in zip(
                channel.notes[:-1], channel.notes[1:], strict=True
            ):
                time = int(
                    mido.second2tick(
                        previous_note.time_delta_secs,
                        ticks_per_beat=DEFAULT_TICKS_PER_BEAT,
                        tempo=DEFAULT_TEMPO,
                    )
                )
                midi_track.append(
                    _parse_message(note, channel_num=channel_num, time=time)
                )
    return midi_file


def _parse_message(note: Note, *, channel_num: int, time: float) -> Message:
    if note.kind == "on":
        message_type = "note_on"
    elif note.kind == "off":
        message_type = "note_off"
    else:
        raise ValueError(note.kind)
    return Message(
        message_type,
        note=note.note,
        velocity=note.velocity,
        time=time,
        channel=channel_num,
    )
