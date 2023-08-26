import dataclasses
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator, Literal, cast

import fire
import fluidsynth
import numpy as np
import pygame
import pygame.locals
import scipy
import torch
from jaxtyping import Float
from loguru import logger

from yoshimidi.data.parse import one_hot_parsing, token_parsing
from yoshimidi.data.parse.tracks import Channel, KeySignature, Note
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
NUM_NOTES = one_hot_parsing.TOKEN_FIELD_LENGTHS["note_on"]

# jaxtyping
note = None


@torch.inference_mode()
def main(
    *,
    model_tag: str,
    soundfont_path: str,
    temperature: float = 0.5,
) -> None:
    pygame.init()
    screen = pygame.display.set_mode((500, 500))
    pygame.font.init()
    font = pygame.font.SysFont("Arial", 20)

    synth = fluidsynth.Synth(gain=1.0)
    synth.start()
    synth_soundfont = synth.sfload(soundfont_path)
    synth.program_select(0, synth_soundfont, 0, 0)

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
            break
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
            synth.noteoff(0, resolved_note)

        elif key_press.type == "on":
            model_distribution = _get_model_distribution(
                model,
                resolved_key_presses,
                now,
                temperature=temperature,
            )
            position_distribution = _get_position_distribution(
                key_press=key_press,
            )
            constraint_distribution = _get_constraint_distribution(
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
                * constraint_distribution
                * currently_playing_distribution
            )
            distribution /= distribution.sum()
            chosen_distribution = np.random.multinomial(n=1, pvals=distribution).astype(
                np.float32
            )
            pygame.draw.rect(screen, (0, 0, 0), pygame.Rect(0, 0, 500, 500))
            _draw_distribution(
                screen,
                font,
                model=model_distribution,
                position=position_distribution,
                constraint=constraint_distribution,
                currently_playing=currently_playing_distribution,
                final=distribution,
                chosen=chosen_distribution,
            )
            pygame.display.flip()
            assert np.all(distribution >= 0)
            resolved_note = int(chosen_distribution.argmax())
            resolved_key_presses.append(
                ResolvedKeyPress(
                    **dataclasses.asdict(key_press),
                    resolved_note=resolved_note,
                )
            )
            synth.noteon(0, resolved_note, 100)
    logger.info("Quitting...")
    pygame.quit()
    synth.delete()


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


def _get_model_distribution(
    model: Transformer,
    resolved_key_presses: list[ResolvedKeyPress],
    now: datetime,
    temperature: float,
) -> Float[np.ndarray, "note"]:
    notes = list(_key_presses_to_notes(resolved_key_presses, now))
    tokens = token_parsing.from_channel(
        Channel(notes=cast(list[Note | KeySignature], notes), program_nums=[])
    )
    one_hots = one_hot_parsing.from_tokens(
        tokens, device=torch.device("cpu"), dtype=torch.float32
    )
    logits = model(one_hots.unsqueeze(0))
    logits /= temperature
    lower, upper = one_hot_parsing.piece_range("note_on")
    logits[:, :, lower] = -1e6  # Exclude "no note" token.
    activations = midi_activation(logits)[0, -1, :]
    return activations[lower:upper].numpy()


def _get_position_distribution(
    key_press: KeyPress,
) -> np.ndarray:
    note = key_press.key_index + 40
    distribution = scipy.stats.norm.pdf(list(range(NUM_NOTES)), loc=note, scale=10)
    return distribution / distribution.sum()


def _get_constraint_distribution(
    key_press: KeyPress,
    now: datetime,
    resolved_key_presses: list[ResolvedKeyPress],
) -> np.ndarray:
    distribution = np.ones(NUM_NOTES)
    for resolved_key_press in resolved_key_presses:
        if resolved_key_press.type == "off":
            continue
        time_since_secs = (now - resolved_key_press.time).total_seconds()
        weight = min(1, math.exp(-time_since_secs + 5))
        key_distribution = np.ones(NUM_NOTES)
        if key_press.key_index > resolved_key_press.key_index:
            key_distribution[: resolved_key_press.resolved_note + 1] = 1 - weight
        elif key_press.key_index < resolved_key_press.key_index:
            key_distribution[resolved_key_press.resolved_note :] = 1 - weight
        else:
            key_distribution[:] = 1 - weight
            key_distribution[resolved_key_press.resolved_note] = 1
        key_distribution /= key_distribution.sum()
        distribution *= key_distribution
        distribution /= distribution.sum()
    return distribution


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
    distribution = np.ones(NUM_NOTES)
    distribution[currently_playing] = 0
    return distribution / distribution.sum()


def _draw_distribution(
    screen: pygame.Surface,
    font: pygame.font.Font,
    *,
    model: np.ndarray,
    position: np.ndarray,
    constraint: np.ndarray,
    currently_playing: np.ndarray,
    final: np.ndarray,
    chosen: np.ndarray,
) -> None:
    num_distributions = 6
    width = screen.get_width()
    height = screen.get_height()
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]

    def _draw(
        d: np.ndarray,
        index: int,
        title: str,
    ) -> None:
        start_x = 0
        end_x = width
        start_y = height * index / num_distributions
        end_y = height * (index + 1) / num_distributions
        max_d = d.max()
        d /= max_d
        for i, p in enumerate(d):
            pygame.draw.rect(
                screen,
                colors[index],
                pygame.Rect(
                    start_x + ((end_x - start_x) / len(d)) * i,
                    start_y + (end_y - start_y) * (1 - p),
                    (end_x - start_x) / len(d),
                    (end_y - start_y) * p,
                ),
            )
        title_surface = font.render(
            f"{title} (max={max_d:.3f})", False, (255, 255, 255)
        )
        screen.blit(title_surface, (start_x, start_y))

    _draw(model, 0, title="model")
    _draw(position, 1, title="position")
    _draw(constraint, 2, title="constraint")
    _draw(currently_playing, 3, title="playing")
    _draw(final, 4, title="final")
    _draw(chosen, 5, title="chosen")


if __name__ == "__main__":
    fire.Fire(main)
