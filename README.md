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

### Milestone 1: Music generation [done]
A language model with no control tokens that produces coherent-ish sounding music.

- [x] Data preparation (parsing, tokenization, etc).
  - [x] Complete data generation run.
- [x] Transformer implementation.
  - [x] Positional encodings.
- [x] Basic training loop.
- [x] Inference loop.
- [x] Evals:
  - [x] Eval schedules.
  - [x] Train/eval split.
- [x] Checkpointing.
- [x] Train a good model.
  - [x] Set up Docker image for training.
  - [x] Figure out vast.ai for training.
  - [x] Figure out why making it larger makes performance worse.
    - It was a bug in attention.
  - [x] Do a big training run!

### Milestone 2: Control tokens

### Milestone 3: Interactive web app

### Milestone 4: Polish

### Backlog
- [x] Categorical loss for time, a la MuZero.
- [ ] Implement GPT-2/3/J architectures.
- [ ] Implement better initialization schemes.
- [ ] Better tokenization of dataset.
- [ ] Shuffle tokenized dataset.
- [ ] Save MIDI files at each eval step.
- [ ] Fix memmap boundaries issue.

## Notes
- notes are too fast
- perhaps bc the supports are heavily biased towards smaller times?
- let's try with more times.
