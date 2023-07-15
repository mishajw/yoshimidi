import pathlib

import pygame
from loguru import logger


def play(midi_file: pathlib.Path) -> None:
    logger.info(f"Playing {midi_file}")
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
    pygame.mixer.music.set_volume(0.8)
    try:
        _play_music(midi_file)
        logger.info("Finished playing")
    except KeyboardInterrupt:
        logger.info("Finishing due to interrupt")
        pygame.mixer.music.stop()


def _play_music(midi_filename):
    clock = pygame.time.Clock()
    pygame.mixer.music.load(midi_filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        clock.tick(30)  # check if playback has finished
