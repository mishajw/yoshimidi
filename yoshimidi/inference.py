import itertools
from typing import Generator

import numpy as np
import torch
import tqdm

from yoshimidi.data.parse import one_hot_parsing, time_parsing, token_parsing
from yoshimidi.data.parse.tracks import Channel, Note
from yoshimidi.train.midi_activation import midi_activation
from yoshimidi.train.transformer import Transformer


@torch.inference_mode()
def run_inference(
    model: Transformer,
    prompt: Channel,
    device: torch.device,
    dtype: torch.dtype,
    max_new_tokens: int,
    temperature: float = 1,
) -> Generator[Note, None, None]:
    tokens = one_hot_parsing.from_tokens(
        token_parsing.from_channel(prompt, include_end=False), device, dtype
    )
    prompt_length = tokens.size(0)
    for _ in tqdm.tqdm(itertools.count(), desc="Generating tokens"):
        logits = model(tokens.unsqueeze(0).float())
        activations = midi_activation(logits)[0, -1, :]

        lower, upper = one_hot_parsing.piece_range("kind")
        kind = one_hot_parsing.KINDS[
            _sample(activations[lower:upper], temperature=temperature)
        ]

        if kind == "end":
            break

        lower, upper = one_hot_parsing.piece_range("time")
        time_support = activations[lower:upper]
        time_uint8 = time_parsing.time_uint8_from_support(time_support)
        time_delta_secs = time_parsing.time_from_uint8(time_uint8)

        lower, upper = one_hot_parsing.piece_range("note_key")
        note_key = _sample(activations[lower:upper], temperature=temperature)
        lower, upper = one_hot_parsing.piece_range("note_octave")
        note_octave = _sample(activations[lower:upper], temperature=temperature)
        note = note_key + 12 * note_octave

        yield Note(
            note=note,
            kind=kind,
            velocity=127,
            time_delta_secs=time_delta_secs,
        )

        next_token = one_hot_parsing.from_tokens(
            np.expand_dims(
                token_parsing.create_token(
                    kind=kind,
                    note=note,
                    time_delta_secs=time_delta_secs,
                ),
                axis=0,
            ),
            device,
            dtype,
        )
        tokens = torch.cat([tokens, next_token], dim=0)

        if tokens.size(0) - prompt_length >= max_new_tokens:
            break


def _sample(
    probabilities: torch.Tensor,
    temperature: float,
) -> int:
    if temperature == 0:
        return int(probabilities.argmax().item())
    probabilities /= temperature
    return int(probabilities.multinomial(num_samples=1, replacement=True).item())
