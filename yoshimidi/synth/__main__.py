import dataclasses
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator, Literal

import fire
import fluidsynth
import pygame
import pygame.locals
import torch

from yoshimidi.data.parse import one_hot_parsing, token_parsing
from yoshimidi.data.parse.tracks import Channel, Note
from yoshimidi.output_config import OutputConfig
from yoshimidi.train import checkpoints
from yoshimidi.train.midi_activation import midi_activation
from yoshimidi.train.transformer import Transformer
from yoshimidi.train.transformer_config import TransformerConfig

SYNTH_KEYS = [
    pygame.locals.K_a,
    pygame.locals.K_s,
    pygame.locals.K_d,
    pygame.locals.K_f,
    pygame.locals.K_g,
    pygame.locals.K_h,
    pygame.locals.K_j,
    pygame.locals.K_k,
    pygame.locals.K_l,
]


def main(
    *,
    model_tag: str,
    soundfont_path: str,
) -> None:
    pygame.init()
    pygame.display.set_mode((100, 100))

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

    key_presses: list[ResolvedKeyPress] = []
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
                for kp in reversed(key_presses)
                if kp.key_index == key_press.key_index
            )
            key_presses.append(
                ResolvedKeyPress(
                    **dataclasses.asdict(key_press),
                    resolved_note=resolved_note,
                )
            )
            fs.noteoff(0, resolved_note)

        elif key_press.type == "on":
            notes = list(_key_presses_to_notes(key_presses, now))
            tokens = token_parsing.from_channel(Channel(notes=notes, program_nums=[]))
            one_hots = one_hot_parsing.from_tokens(
                tokens, device=torch.device("cpu"), dtype=torch.float32
            )
            logits = model(one_hots.unsqueeze(0))
            lower, upper = one_hot_parsing.piece_range("note_on")
            logits[:, :, lower] = -1e6  # Exclude "no note" token.
            activations = midi_activation(logits)[0, -1, :]
            resolved_note = _sample(activations[lower:upper])
            key_presses.append(
                ResolvedKeyPress(
                    **dataclasses.asdict(key_press),
                    resolved_note=resolved_note,
                )
            )
            fs.noteon(0, resolved_note, 127)


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
        yield Note(
            note=key_press.resolved_note,
            kind=key_press.type,
            velocity=127,
            time_delta_secs=(next_time - key_press.time).total_seconds(),
        )


def _sample(
    probabilities: torch.Tensor,
) -> int:
    return int(probabilities.multinomial(num_samples=1, replacement=True).item())


if __name__ == "__main__":
    fire.Fire(main)
