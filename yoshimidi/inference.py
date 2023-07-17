import itertools
from typing import Generator

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
    tokens = torch.tensor(token_parsing.from_channel(prompt, include_end=False))
    prompt_length = tokens.size(0)
    current_time_delta_secs: float | None = None
    for _ in tqdm.tqdm(itertools.count(), desc="Generating tokens"):
        logits = model(tokens.unsqueeze(0))
        activations = midi_activation(logits)[0, -1, :]

        lower, upper = token_format.piece_range("kind")
        kind = token_format.KINDS[_sample(activations[lower:upper])]

        if kind == "pause":
            assert current_time_delta_secs is None
            lower, upper = token_format.piece_range("time")
            time_support = activations[lower:upper].numpy()
            current_time_delta_secs = token_format.support_to_time(time_support)
            next_token = token_parsing.create_torch_token(
                "pause", time=current_time_delta_secs
            )

        if kind == "on" or kind == "off":
            assert current_time_delta_secs is not None
            lower, upper = token_format.piece_range("note_key")
            note_key = _sample(activations[lower:upper])
            lower, upper = token_format.piece_range("note_octave")
            note_octave = _sample(activations[lower:upper])
            note = note_key + 12 * note_octave
            yield Note(
                note=note,
                kind=kind,
                velocity=127,
                time_delta_secs=current_time_delta_secs,
            )
            current_time_delta_secs = None
            next_token = token_parsing.create_torch_token(kind, note=note)

        if kind == "end":
            assert current_time_delta_secs is None
            break

        tokens = torch.cat([tokens, next_token.unsqueeze(0)])

        if tokens.size(0) - prompt_length >= max_new_tokens:
            break


def _sample(probabilities: torch.Tensor) -> int:
    return int(probabilities.multinomial(num_samples=1, replacement=True).item())
