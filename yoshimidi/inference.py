import itertools
from typing import Generator

import numpy as np
import torch
import tqdm
from loguru import logger

from yoshimidi.data.parse import one_hot_parsing, token_parsing
from yoshimidi.data.parse.tracks import Channel, Note
from yoshimidi.train.midi_activation import midi_activation
from yoshimidi.train.model.transformer import Transformer


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
    for _ in tqdm.tqdm(
        itertools.count(), desc="Generating tokens", total=max_new_tokens
    ):
        logits = model(tokens.unsqueeze(0).float())
        assert temperature > 0, "TODO: Support temperature=0."
        logits /= temperature
        activations = midi_activation(logits)[0, -1, :]

        token = one_hot_parsing.to_token(_sample(activations), activations)
        kind = token_parsing.get_kind(token)

        if note_buffer is not None and kind != "pause":
            yield note_buffer
            note_buffer = None

        if kind == "end":
            break

        elif kind == "on":
            note = token_parsing.get_note_on(token)
            note_buffer = Note(
                note=note,
                kind="on",
                velocity=127,
                time_delta_secs=0,
            )

        elif kind == "off":
            note = token_parsing.get_note_off(token)
            note_buffer = Note(
                note=note,
                kind="off",
                velocity=127,
                time_delta_secs=0,
            )

        elif kind == "pause":
            time_delta_secs = token_parsing.get_time_secs(token)
            if note_buffer is not None:
                if note_buffer.time_delta_secs != 0:
                    logger.warning("Found multiple pause tokens in a row")
                note_buffer.time_delta_secs += time_delta_secs
            else:
                logger.warning("Found pause without preceding note")

        else:
            raise ValueError(kind)

        if note_buffer is not None:
            yield note_buffer

        next_token_tensor = one_hot_parsing.from_tokens(
            np.expand_dims(token, axis=0),
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
