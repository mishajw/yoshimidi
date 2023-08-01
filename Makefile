#!make

include .env
export

# Development
# ===========

.PHONY: install
install:
	poetry install

.PHONY: precommit
precommit:
	poetry run pre-commit run --all-files

.PHONY: mypy
mypy:
	poetry run mypy .

.PHONY: test
test:
	poetry run pytest

.PHONY: du
du:
	@echo "> out"
	@du -sh out/*
	@echo "> dataset"
	@du -sh out/dataset/*
	@echo "> checkpoints"
	@du -sh out/checkpoints/*

# Dataset prep
# ============

.PHONY: parse
parse:
	poetry run python yoshimidi/clis/parse_midi_files.py

.PHONY: tokenize
tokenize:
	poetry run python yoshimidi/clis/tokenize_midi_dataset.py

.PHONY: train
train:
	poetry run python yoshimidi/clis/train.py configs/train.toml

# S3
# ==

.PHONY: s3-upload-dataset
s3-upload-dataset:
	s5cmd sync \
		out/dataset/03_tokenized/ \
		's3://yoshimidi-v2/datasets/2023-07-31/'
.PHONY: s3-download-dataset
s3-download-dataset:
	s5cmd sync \
		's3://yoshimidi-v2/datasets/2023-07-31/*' \
		out/dataset/03_tokenized/

.PHONY: s3-upload-checkpoints
s3-upload-checkpoints:
	s5cmd sync \
		out/checkpoints/ \
		's3://yoshimidi-v2/checkpoints/'
.PHONY: s3-download-checkpoints
s3-download-checkpoints:
	s5cmd sync \
		's3://yoshimidi-v2/checkpoints/*' \
		out/checkpoints/

.PHONY: s3-du
s3-du:
	@s5cmd du --humanize \
		's3://yoshimidi-v2/*'

# Vast.ai
# =======

.PHONY: vast-ssh
vast-ssh:
	poetry run python yoshimidi/clis/vastai.py ssh

.PHONY: vast-make
vast-make:
	poetry run python yoshimidi/clis/vastai.py make $(CMD)

.PHONY: vast-rsync
vast-rsync:
	poetry run python yoshimidi/clis/vastai.py rsync
