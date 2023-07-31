# YoshiMIDI

Language model trained to generate music, with control tokens a la [Pretraining Language Models with Human Preferences](https://arxiv.org/abs/2302.08582). The control tokens allow us to put a human-in-the-loop for music generation. Now when you press a key on a keyboard, instead of playing the note corresponding to that key, it generates music conditions on that *kind* of key being pressed. By kind, I mean stuff like: this note is a bit higher than the previous note, this note is played alongside three other notes, this note was pressed harshly.

## Usage

```bash
# Download & parse the dataset.
make parse
# Tokenize the dataset.
make tokenize
# Train the model.
make train
```

## Infra
- **Docker**: We build an image from `Dockerfile` which is pushed to [Docker Hub](https://hub.docker.com/repository/docker/mishajw/yoshimidi/).
- **S3**: There's an S3 bucket containing all training data + checkpoints under `s3://yoshimidi`.
- **Vast.ai**: Training is done on 1x RTX 3090 GPU from Vast.ai (~$0.05/hour).

## Milestones

### Milestone 1: Music generation
A language model with no control tokens that produces coherent-ish sounding music.

- [x] Data preparation (parsing, tokenization, etc).
  - [x] Complete data generation run.
  - [ ] Shuffling dataset.
  - [ ] Fix memmap boundaries issue.
- [x] Transformer implementation.
  - [x] Positional encodings.
- [x] Basic training loop.
- [x] Inference loop.
- [ ] Evals:
  - [ ] Eval schedules.
  - [ ] Train/eval split.
  - [ ] Save MIDI files at each eval step.
- [x] Checkpointing.
- [ ] Train a good model.
  - [ ] Figure out why making it larger makes performance worse.
  - [ ] Set up Docker image for training.
  - [ ] Figure out vast.ai for training.
  - [ ] Do a big training run!

### Milestone 2: Control tokens

### Milestone 3: Interactive web app

### Milestone 4: Polish

### Backlog
- [ ] Categorical loss for time, a la MuZero.
- [ ] Implement GPT-2/3/J architectures.
- [ ] Implement better initialization schemes.
- [ ] Better tokenization of dataset.

## Notes
- realised that we have *way* too inefficent storing of dataset: we store it one-hot!
- let's fix that
- while we're there, let's also change the format such that it's (on/off, key pressed, sleep after)
