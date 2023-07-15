import mido
from mido import Message, MetaMessage, MidiFile, MidiTrack

from yoshimidi.data.parse.track_parsing import DEFAULT_TEMPO, DEFAULT_TICKS_PER_BEAT
from yoshimidi.data.parse.tracks import Track


def from_tracks(tracks: list[Track]) -> MidiFile:
    midi_file = MidiFile()
    midi_file.ticks_per_beat = DEFAULT_TICKS_PER_BEAT
    for track in tracks:
        midi_track = MidiTrack()
        midi_file.tracks.append(midi_track)
        # TODO: Set program_nums.
        midi_track.append(MetaMessage("set_tempo", tempo=DEFAULT_TEMPO, time=0))

        notes_with_channels = [
            (
                note,
                channel_num,
            )
            for channel_num, channel in track.channels.items()
            for note in channel.notes
        ]

        for note, channel_num in notes_with_channels:
            if note.kind == "on":
                message_type = "note_on"
            elif note.kind == "off":
                message_type = "note_off"
            else:
                raise ValueError(note.kind)

            time = int(
                mido.second2tick(
                    note.time_delta_secs,
                    ticks_per_beat=DEFAULT_TICKS_PER_BEAT,
                    tempo=DEFAULT_TEMPO,
                )
            )
            midi_track.append(
                Message(
                    message_type,
                    note=note.note,
                    velocity=note.velocity,
                    time=time,
                    channel=channel_num,
                )
            )
    return midi_file
