from typing import cast

import mido
from mido import Message, MetaMessage, MidiFile, MidiTrack

from yoshimidi.data.parse.tracks import KeySignature, Note, Track

_DEFAULT_TICKS_PER_BEAT = 120
_DEFAULT_TEMPO = 447761


def from_tracks(tracks: list[Track]) -> MidiFile:
    midi_file = MidiFile()
    midi_file.ticks_per_beat = _DEFAULT_TICKS_PER_BEAT
    for track in tracks:
        assert (
            len(track.channels) == 1
        ), f"Only supports single-channel tracks, found {len(track.channels)}"
        midi_track = MidiTrack()
        midi_file.tracks.append(midi_track)
        # TODO: Set program_nums.
        midi_track.append(MetaMessage("set_tempo", tempo=_DEFAULT_TEMPO, time=0))

        for channel_num, channel in track.channels.items():
            for note in _shift_time_deltas(channel.notes):
                if isinstance(note, Note):
                    time = int(
                        mido.second2tick(
                            note.time_delta_secs,
                            ticks_per_beat=_DEFAULT_TICKS_PER_BEAT,
                            tempo=_DEFAULT_TEMPO,
                        )
                    )
                    midi_track.append(
                        _parse_message(note, channel_num=channel_num, time=time)
                    )
                elif isinstance(note, KeySignature):
                    midi_track.append(
                        MetaMessage(
                            "key_signature",
                            key=note.key,
                            time=0,
                        )
                    )
                else:
                    raise ValueError(note)
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


def _shift_time_deltas(notes: list[Note | KeySignature]) -> list[Note | KeySignature]:
    shifted_notes = [*notes]
    note_indices = [i for i, note in enumerate(notes) if isinstance(note, Note)]
    first_note = cast(Note, notes[note_indices[0]])
    shifted_notes[note_indices[0]] = Note(
        note=first_note.note,
        kind=first_note.kind,
        velocity=first_note.velocity,
        time_delta_secs=0.0,
    )
    for note1_index, note2_index in zip(
        note_indices[:-1], note_indices[1:], strict=True
    ):
        note1 = cast(Note, notes[note1_index])
        note2 = cast(Note, notes[note2_index])
        shifted_notes[note2_index] = Note(
            note=note2.note,
            kind=note2.kind,
            velocity=note2.velocity,
            time_delta_secs=note1.time_delta_secs,
        )
    return shifted_notes
