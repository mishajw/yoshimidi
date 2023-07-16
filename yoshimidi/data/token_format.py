from typing import Literal, TypeAlias

KindType: TypeAlias = Literal["on", "off", "pause", "end"]
PieceType: TypeAlias = Literal["kind", "note_key", "note_octave", "time"]

VOCAB = 28  # 4 + 12 + 11 + 1
PIECE_LENGTHS: dict[PieceType, int] = {
    "kind": 4,
    "note_key": 12,
    "note_octave": 11,
    "time": 1,
}
