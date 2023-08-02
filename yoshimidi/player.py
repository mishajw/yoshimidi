from pathlib import Path
from tempfile import NamedTemporaryFile

import pygame
from loguru import logger

from yoshimidi.data.parse import midi_parsing
from yoshimidi.data.parse.tracks import Channel, Track, TrackMetadata


def play(midi_file: Path) -> None:
    logger.info(f"Playing {midi_file}")
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
    pygame.mixer.music.set_volume(0.8)
    try:
        _play_music(midi_file)
        logger.info("Finished playing")
    except KeyboardInterrupt:
        logger.info("Finishing due to interrupt")
        pygame.mixer.music.stop()


def play_channel(channel: Channel) -> None:
    track = Track(channels={0: channel}, metadata=TrackMetadata())
    midi_track = midi_parsing.from_tracks([track])
    with NamedTemporaryFile() as temp_file:
        midi_track.save(temp_file.name)
        play(Path(temp_file.name))


def _play_music(midi_file: Path) -> None:
    clock = pygame.time.Clock()
    pygame.mixer.music.load(midi_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        clock.tick(30)  # check if playback has finished
