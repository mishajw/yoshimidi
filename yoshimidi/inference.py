import itertools
from typing import Generator

import numpy as np
import torch
import tqdm
from loguru import logger

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
    note_buffer = None
    for _ in tqdm.tqdm(itertools.count(), desc="Generating tokens"):
        logits = model(tokens.unsqueeze(0).float())
        assert temperature > 0, "TODO: Support temperature=0."
        logits /= temperature
        activations = midi_activation(logits)[0, -1, :]

        lower, upper = one_hot_parsing.piece_range("kind")
        kind = token_parsing.KINDS[_sample(activations[lower:upper])]
        next_token = np.zeros(token_parsing.TOKEN_DIM, dtype=token_parsing.DTYPE)

        if note_buffer is not None and kind != "pause":
            yield note_buffer
            note_buffer = None

        if kind == "end":
            break

        elif kind == "on":
            lower, upper = one_hot_parsing.piece_range("note_on")
            note = _sample(activations[lower:upper])
            note_buffer = Note(
                note=note,
                kind="on",
                velocity=127,
                time_delta_secs=0,
            )
            token_parsing.create_note_on_token(next_token, note)

        elif kind == "off":
            lower, upper = one_hot_parsing.piece_range("note_off")
            note = _sample(activations[lower:upper])
            note_buffer = Note(
                note=note,
                kind="off",
                velocity=127,
                time_delta_secs=0,
            )
            token_parsing.create_note_off_token(next_token, note)

        elif kind == "pause":
            lower, upper = one_hot_parsing.piece_range("time")
            time_support = activations[lower:upper]
            # TODO: How can we best sample from the time distribution?
            time_support_one_hot = torch.zeros_like(time_support)
            time_support_one_hot[_sample(time_support)] = 1
            time_uint8 = time_parsing.time_uint8_from_support(time_support_one_hot)
            time_delta_secs = time_parsing.time_from_uint8(time_uint8)
            if note_buffer is not None:
                if note_buffer.time_delta_secs != 0:
                    logger.warning("Found multiple pause tokens in a row")
                note_buffer.time_delta_secs += time_delta_secs
            else:
                logger.warning("Found pause without preceding note")
            token_parsing.create_pause_token(next_token, time_delta_secs)

        else:
            raise ValueError(kind)

        if note_buffer is not None:
            yield note_buffer

        next_token_tensor = one_hot_parsing.from_tokens(
            np.expand_dims(next_token, axis=0),
            device,
            dtype,
        )
        tokens = torch.cat([tokens, next_token_tensor], dim=0)

        if tokens.size(0) - prompt_length >= max_new_tokens:
            break


def _sample(
    probabilities: torch.Tensor,
) -> int:
    return int(probabilities.multinomial(num_samples=1, replacement=True).item())
