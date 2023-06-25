import pathlib

import pygame


def play(midi_file: pathlib.Path) -> None:
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
    pygame.mixer.music.set_volume(0.8)
    try:
        _play_music(midi_file)
    except KeyboardInterrupt:
        pygame.mixer.music.stop()
        raise SystemExit


def _play_music(midi_filename):
    clock = pygame.time.Clock()
    pygame.mixer.music.load(midi_filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        clock.tick(30)  # check if playback has finished
