from yoshimidi.data.parse.track_parsing import _shift_time_deltas
from yoshimidi.data.parse.tracks import KeySignature, Note


def test_shift_time_deltas() -> None:
    notes: list[Note | KeySignature] = [
        Note(note=60, kind="on", velocity=100, time_delta_secs=0.1),
        Note(note=62, kind="off", velocity=0, time_delta_secs=0.2),
        Note(note=64, kind="on", velocity=100, time_delta_secs=0.3),
        Note(note=67, kind="off", velocity=0, time_delta_secs=0.4),
        Note(note=69, kind="on", velocity=100, time_delta_secs=0.5),
        Note(note=72, kind="off", velocity=0, time_delta_secs=0.6),
    ]
    expected_notes: list[Note | KeySignature] = [
        Note(note=60, kind="on", velocity=100, time_delta_secs=0.2),
        Note(note=62, kind="off", velocity=0, time_delta_secs=0.3),
        Note(note=64, kind="on", velocity=100, time_delta_secs=0.4),
        Note(note=67, kind="off", velocity=0, time_delta_secs=0.5),
        Note(note=69, kind="on", velocity=100, time_delta_secs=0.6),
        Note(note=72, kind="off", velocity=0, time_delta_secs=0.0),
    ]
    assert _shift_time_deltas(notes) == expected_notes


def test_shift_time_deltas_with_key_signatures() -> None:
    notes: list[Note | KeySignature] = [
        KeySignature(key="A"),
        Note(note=60, kind="on", velocity=100, time_delta_secs=0.1),
        Note(note=62, kind="off", velocity=0, time_delta_secs=0.2),
        KeySignature(key="B"),
        Note(note=64, kind="on", velocity=100, time_delta_secs=0.3),
        Note(note=67, kind="off", velocity=0, time_delta_secs=0.4),
        KeySignature(key="C"),
        Note(note=69, kind="on", velocity=100, time_delta_secs=0.5),
        Note(note=72, kind="off", velocity=0, time_delta_secs=0.6),
        KeySignature(key="D"),
    ]
    expected_notes: list[Note | KeySignature] = [
        KeySignature(key="A"),
        Note(note=60, kind="on", velocity=100, time_delta_secs=0.2),
        Note(note=62, kind="off", velocity=0, time_delta_secs=0.3),
        KeySignature(key="B"),
        Note(note=64, kind="on", velocity=100, time_delta_secs=0.4),
        Note(note=67, kind="off", velocity=0, time_delta_secs=0.5),
        KeySignature(key="C"),
        Note(note=69, kind="on", velocity=100, time_delta_secs=0.6),
        Note(note=72, kind="off", velocity=0, time_delta_secs=0.0),
        KeySignature(key="D"),
    ]
    assert _shift_time_deltas(notes) == expected_notes
