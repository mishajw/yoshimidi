from typing import Literal, Tuple, TypeAlias

PieceType: TypeAlias = Literal["kind", "note_key", "note_octave", "time"]
KindType: TypeAlias = Literal["on", "off", "pause", "end"]

KINDS: list[KindType] = ["on", "off", "pause", "end"]
VOCAB = 28  # 4 + 12 + 11 + 1
PIECE_LENGTHS: dict[PieceType, int] = {
    "kind": 4,
    "note_key": 12,
    "note_octave": 11,
    "time": 1,
}


def piece_range(piece_type: PieceType) -> Tuple[int, int]:
    index = 0
    for piece, length in PIECE_LENGTHS.items():
        if piece == piece_type:
            return index, index + length
        index += length
    raise ValueError(piece)
