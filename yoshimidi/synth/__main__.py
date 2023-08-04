import dataclasses
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator, Literal

import fire
import fluidsynth
import numpy as np
import pygame
import pygame.locals
import scipy
import torch
from jaxtyping import Float

from yoshimidi.data.parse import one_hot_parsing, token_parsing
from yoshimidi.data.parse.tracks import Channel, Note
from yoshimidi.output_config import OutputConfig
from yoshimidi.train import checkpoints
from yoshimidi.train.midi_activation import midi_activation
from yoshimidi.train.transformer import Transformer
from yoshimidi.train.transformer_config import TransformerConfig

SYNTH_KEYS = [
    pygame.locals.K_q,
    pygame.locals.K_a,
    pygame.locals.K_w,
    pygame.locals.K_s,
    pygame.locals.K_e,
    pygame.locals.K_d,
    pygame.locals.K_r,
    pygame.locals.K_f,
    pygame.locals.K_t,
    pygame.locals.K_g,
    pygame.locals.K_y,
    pygame.locals.K_h,
    pygame.locals.K_u,
    pygame.locals.K_j,
    pygame.locals.K_i,
    pygame.locals.K_k,
    pygame.locals.K_o,
    pygame.locals.K_l,
    pygame.locals.K_p,
    pygame.locals.K_SEMICOLON,
    pygame.locals.K_LEFTBRACKET,
    pygame.locals.K_QUOTE,
    pygame.locals.K_RIGHTBRACKET,
]


# jaxtyping
note = None


@torch.inference_mode()
def main(
    *,
    model_tag: str,
    soundfont_path: str,
) -> None:
    pygame.init()
    pygame.display.set_mode((500, 500))

    fs = fluidsynth.Synth()
    fs.start()
    sfid = fs.sfload(soundfont_path)
    fs.program_select(0, sfid, 0, 0)
    fs.setting("synth.gain", 0.6)

    model = Transformer(
        TransformerConfig(
            num_layers=6,
            residual_stream_size=512,
            num_attention_heads=16,
            context_window=1024,
        )
    )
    optimizer = torch.optim.Adam(model.parameters())
    model, optimizer = checkpoints.load_checkpoint(
        tag=model_tag,
        step="latest",
        model=model,
        optimizer=optimizer,
        output_config=OutputConfig(),
        device=torch.device("cpu"),
    )

    resolved_key_presses: list[ResolvedKeyPress] = []
    pygame.event.clear()
    while True:
        event = pygame.event.wait()
        key_press = None
        now = datetime.now()
        if event.type == pygame.locals.QUIT:
            pygame.quit()
            return
        elif event.type == pygame.locals.KEYDOWN and event.key in SYNTH_KEYS:
            key_press = KeyPress(
                key_index=SYNTH_KEYS.index(event.key),
                type="on",
                time=now,
            )
        elif event.type == pygame.locals.KEYUP and event.key in SYNTH_KEYS:
            key_press = KeyPress(
                key_index=SYNTH_KEYS.index(event.key),
                type="off",
                time=now,
            )

        if key_press is None:
            continue

        if key_press.type == "off":
            resolved_note = next(
                kp.resolved_note
                for kp in reversed(resolved_key_presses)
                if kp.key_index == key_press.key_index
            )
            resolved_key_presses.append(
                ResolvedKeyPress(
                    **dataclasses.asdict(key_press),
                    resolved_note=resolved_note,
                )
            )
            fs.noteoff(0, resolved_note)

        elif key_press.type == "on":
            model_distribution = _get_model_distribution(
                model,
                resolved_key_presses,
                now,
            )
            position_distribution = _get_position_distribution(
                key_press=key_press,
                now=now,
                resolved_key_presses=resolved_key_presses,
            )
            currently_playing_distribution = _get_currently_playing_distribution(
                resolved_key_presses=resolved_key_presses,
            )
            distribution = (
                model_distribution
                * position_distribution
                * currently_playing_distribution
            )
            assert torch.all(distribution >= 0)
            distribution /= distribution.sum()
            resolved_note = _sample(distribution)
            resolved_key_presses.append(
                ResolvedKeyPress(
                    **dataclasses.asdict(key_press),
                    resolved_note=resolved_note,
                )
            )
            fs.noteon(0, resolved_note, 100)


@dataclass
class KeyPress:
    key_index: int
    type: Literal["on", "off"]
    time: datetime


@dataclass
class ResolvedKeyPress(KeyPress):
    resolved_note: int


def _key_presses_to_notes(
    key_presses: list[ResolvedKeyPress], now: datetime
) -> Iterator[Note]:
    for i, key_press in enumerate(key_presses):
        if i < len(key_presses) - 1:
            next_time = key_presses[i + 1].time
        else:
            next_time = now
        time_delta_secs = (next_time - key_press.time).total_seconds()
        if time_delta_secs < 0.1:
            time_delta_secs = 0
        yield Note(
            note=key_press.resolved_note,
            kind=key_press.type,
            velocity=127,
            time_delta_secs=time_delta_secs,
        )


def _sample(
    probabilities: torch.Tensor,
) -> int:
    return int(probabilities.multinomial(num_samples=1, replacement=True).item())


def _get_model_distribution(
    model: Transformer,
    resolved_key_presses: list[ResolvedKeyPress],
    now: datetime,
) -> Float[np.ndarray, "note"]:
    notes = list(_key_presses_to_notes(resolved_key_presses, now))
    tokens = token_parsing.from_channel(Channel(notes=notes, program_nums=[]))
    one_hots = one_hot_parsing.from_tokens(
        tokens, device=torch.device("cpu"), dtype=torch.float32
    )
    logits = model(one_hots.unsqueeze(0))
    logits /= 0.1
    lower, upper = one_hot_parsing.piece_range("note_on")
    logits[:, :, lower] = -1e6  # Exclude "no note" token.
    activations = midi_activation(logits)[0, -1, :]
    return activations[lower:upper].numpy()


def _get_position_distribution(
    key_press: KeyPress,
    now: datetime,
    resolved_key_presses: list[ResolvedKeyPress],
) -> np.ndarray:
    num_notes = one_hot_parsing.TOKEN_FIELD_LENGTHS["note_on"]
    if len(resolved_key_presses) == 0:
        return np.ones(num_notes)
    # for resolved_key_press in resolved_key_presses:
    #     if resolved_key_press.type == "off":
    #         continue
    #     diff = (key_press.key_index - resolved_key_press.key_index)
    #     note = resolved_key_press.resolved_note + diff
    #     weight = math.exp(-(now - resolved_key_press.time).total_seconds())
    #     print(resolved_key_press.resolved_note, diff, note, weight, sep="\t")
    #     note_sum += note * weight
    #     weight_sum += weight
    # note = note_sum / weight_sum
    note = key_press.key_index + 40
    distribution = scipy.stats.norm.pdf(list(range(num_notes)), loc=note, scale=10)
    return distribution / distribution.sum()


def _get_currently_playing_distribution(
    resolved_key_presses: list[ResolvedKeyPress],
) -> np.ndarray:
    currently_playing = set()
    for key_press in resolved_key_presses:
        if key_press.type == "on":
            currently_playing.add(key_press.resolved_note)
        elif key_press.type == "off" and key_press.resolved_note in currently_playing:
            currently_playing.remove(key_press.resolved_note)
    currently_playing = list(currently_playing)
    distribution = np.ones(one_hot_parsing.TOKEN_FIELD_LENGTHS["note_on"])
    distribution[currently_playing] = 0
    return distribution / distribution.sum()


if __name__ == "__main__":
    fire.Fire(main)
