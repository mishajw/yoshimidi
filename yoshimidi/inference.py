import itertools
from typing import Generator, Literal

import torch
import tqdm

from yoshimidi.data import token_format
from yoshimidi.data.parse import token_parsing
from yoshimidi.data.parse.tracks import Channel, Note
from yoshimidi.train.midi_activation import midi_activation
from yoshimidi.train.transformer import Transformer


@torch.inference_mode()
def run_inference(
    model: Transformer,
    prompt: Channel,
    max_new_tokens: int,
) -> Generator[Note, None, None]:
    # TODO: Remove [:-1] hack.
    tokens = torch.tensor(token_parsing.from_channel(prompt, include_end=False)[:-1])
    prompt_length = tokens.size(0)
    current_note: int | None = None
    current_note_kind: Literal["on", "off"] | None = None
    for _ in tqdm.tqdm(itertools.count(), desc="Generating tokens"):
        logits = model(tokens.unsqueeze(0))
        activations = midi_activation(logits)[0, -1, :]

        lower, upper = token_format.piece_range("kind")
        # activations[upper - 1] = 0
        kind = token_format.KINDS[_sample(activations[lower:upper])]

        if kind == "on" or kind == "off":
            assert current_note is None
            assert current_note_kind is None
            lower, upper = token_format.piece_range("note_key")
            note_key = _sample(activations[lower:upper])
            lower, upper = token_format.piece_range("note_octave")
            note_octave = _sample(activations[lower:upper])
            current_note = note_key + 12 * note_octave
            current_note_kind = kind
            next_token = token_parsing.create_torch_token(kind, note=current_note)

        if kind == "pause":
            assert current_note is not None
            assert current_note_kind is not None
            lower, upper = token_format.piece_range("time")
            assert lower + 1 == upper
            time_delta = max(0, activations[lower].item())
            yield Note(
                note=current_note,
                kind=current_note_kind,
                velocity=127,
                time_delta_secs=time_delta,
            )
            current_note = None
            current_note_kind = None
            next_token = token_parsing.create_torch_token("pause", time=time_delta)

        if kind == "end":
            assert current_note is not None
            assert current_note_kind is not None
            yield Note(
                note=current_note,
                kind=current_note_kind,
                velocity=128,
                time_delta_secs=0,
            )
            break

        tokens = torch.cat([tokens, next_token.unsqueeze(0)])

        if tokens.size(0) - prompt_length >= max_new_tokens:
            break


def _sample(probabilities: torch.Tensor) -> int:
    return int(probabilities.multinomial(num_samples=1, replacement=True).item())
